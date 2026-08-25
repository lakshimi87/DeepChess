#!/usr/bin/env python3
"""DeepChess self-play training pipeline.

Run repeatedly to accumulate training iterations — each invocation
loads the latest checkpoint and continues from where it left off.

    ./train.sh                       # default settings
    ./train.sh --iterations 50       # override iteration count
    ./train.sh --simulations 400     # stronger self-play
"""

import argparse
import math
import os
import random
import signal
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import _ext, perf
from .model import ChessNet
from .paths import CHECKPOINTS_DIR
from .selfplay import SelfPlayPool, fp16_state_dict
from .validate_gt import score_model

# Self-play is CPU-bound, so configure single-threaded torch *before* any
# tensor work happens.  See src/perf.py for the measurements.
perf.configure(num_threads=1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def default_workers(device):
	"""Pick a self-play worker count.

	Each worker needs ~1 core for python-chess work plus a slice of GPU.
	Measured on a 28-core box with an RTX 5070 Ti at 600 sims/move:

	     1 worker : 13.2 moves/s,  GPU  ~6%
	     6 workers: 44.2 moves/s,  GPU ~60%
	    10 workers: 55.9 moves/s,  GPU 95-100%

	So the knee is around 8 — past that the GPU is the wall and extra workers
	only add ~400 MB of CUDA context each.  On CPU-only boxes the GPU is not
	the limit, so scale with cores instead.
	"""
	cores = os.cpu_count() or 4
	if device.type == "cuda":
		free, _total = torch.cuda.mem_get_info()
		# ~900 MB per worker: CUDA context, workspace, and *two* fp16 models —
		# the self-play net plus the frozen arena opponent.  Never more workers
		# than cores minus one for the parent's training step.
		by_mem = max(1, int(free / (900 * 1024 ** 2)) - 1)
		return max(1, min(8, cores - 1, by_mem))
	return max(1, min(8, cores - 1))


def policy_target_stats(examples, sample=4000, rng=random):
	"""Entropy diagnostics for a batch of MCTS policy targets.

	This is *the* number to watch on a self-play run.  Policy cross-entropy can
	only ever fall to the entropy of the targets it is fitting, so once the two
	meet, training has converged and running longer changes nothing no matter
	what the loss curve looks like.  A target entropy that sits flat at close to
	log(legal moves) means the search is spreading its visits nearly uniformly
	and the targets carry almost no information — the loop is then stuck in a
	fixed point where flat priors produce flat targets produce flat priors.
	"""
	if not examples:
		return None
	n = len(examples)
	idx = range(n) if n <= sample else rng.sample(range(n), sample)
	ents = []
	tops = []
	legal = []
	for i in idx:
		p = examples[i][1]
		nz = p[p > 0]
		if nz.size == 0:
			continue
		ents.append(float(-(nz * np.log(nz)).sum()))
		tops.append(float(nz.max()))
		legal.append(int(nz.size))
	if not ents:
		return None
	legal_arr = np.asarray(legal, dtype=np.float64)
	return {
		"entropy": float(np.mean(ents)),
		"uniform_entropy": float(np.mean(np.log(legal_arr))),
		"top1": float(np.mean(tops)),
		"legal": float(np.mean(legal_arr)),
	}


def value_target_stats(examples):
	"""Win/draw/loss split of the value targets in *examples*."""
	if not examples:
		return None
	v = np.array([e[2] for e in examples], dtype=np.float32)
	return {
		"win": float((v > 0.5).mean()),
		"draw": float((np.abs(v) <= 0.5).mean()),
		"loss": float((v < -0.5).mean()),
	}


def weight_decay_groups(model, weight_decay):
	"""Split parameters so only conv/linear weights get L2.

	BatchNorm gammas/betas and every bias are excluded.  Decaying a BN gamma
	pushes it toward zero, and a gamma near zero collapses that channel's
	output to a constant; the previous run finished with BN gammas around
	0.07-0.09 and running variances down at 1e-5, which is exactly what that
	looks like.  There is nothing to regularise in a per-channel scale anyway.
	"""
	decay = []
	no_decay = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if param.ndim <= 1 or name.endswith(".bias"):
			no_decay.append(param)
		else:
			decay.append(param)
	return [
		{"params": decay, "weight_decay": weight_decay},
		{"params": no_decay, "weight_decay": 0.0},
	]


def elo_diff(win_rate):
	"""Elo difference implied by *win_rate*, clamped at +/-800 for 0 and 1."""
	if win_rate <= 0.0:
		return -800.0
	if win_rate >= 1.0:
		return 800.0
	return -400.0 * math.log10(1.0 / win_rate - 1.0)


def _atomic_save(payload, path):
	"""``torch.save`` to *path* without ever exposing a partial file.

	``latest.pt`` is ~90 MB and is rewritten every iteration, so a plain
	``torch.save`` leaves a multi-second window in which the file on disk is
	truncated or half-written.  Anything reading concurrently — play.sh,
	validate_gt.sh, a resume from another shell, or a `git add` — can land in
	that window and get a corrupt checkpoint.  Writing to a temp file in the
	same directory and renaming makes the swap atomic: readers see either the
	old checkpoint or the new one, never a mixture.

	The temp name carries the pid so two training runs sharing a checkpoint
	directory can't clobber each other's partial writes.
	"""
	tmp = f"{path}.tmp.{os.getpid()}"
	try:
		torch.save(payload, tmp)
		os.replace(tmp, path)
	except BaseException:
		# Don't leave debris behind on interrupt or disk-full.
		try:
			os.unlink(tmp)
		except OSError:
			pass
		raise


def save_checkpoint(model, optimizer, scheduler, iteration, checkpoint_dir,
                    num_res_blocks, num_filters, numbered=True, generation=0):
	"""Write ``latest.pt`` and (optionally) ``model_iter_XXXX.pt``.

	``latest.pt`` is always refreshed so play.sh/resume always see the most
	recent weights.  ``numbered`` controls whether a permanent snapshot is
	also emitted — the training loop only does this every N iterations to
	keep disk usage bounded.

	Both writes go through :func:`_atomic_save`, so a reader is never exposed
	to a partially written checkpoint.
	"""
	os.makedirs(checkpoint_dir, exist_ok=True)
	payload = {
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
		"iteration": iteration,
		"num_res_blocks": num_res_blocks,
		"num_filters": num_filters,
		"generation": generation,
	}
	if numbered:
		numbered_path = os.path.join(checkpoint_dir, f"model_iter_{iteration:04d}.pt")
		_atomic_save(payload, numbered_path)
	_atomic_save(payload, os.path.join(checkpoint_dir, "latest.pt"))


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_on_data(model, optimizer, device, replay_buffer,
                  batch_size=256, steps=0, epochs=1, value_weight=1.0,
                  amp=True):
	"""Train on a random sample of the replay buffer.

	*steps* fixes the number of gradient steps directly, so the amount of
	training a position receives is set by the caller (see --sample-reuse)
	instead of falling out of the buffer's size.  One full pass over a large
	buffer that turns over slowly replays each position once per iteration
	for as many iterations as it takes to age out -- with a 200k buffer and
	~9k new positions per iteration that is ~22 high-LR passes over data the
	net has already fitted, which memorises the buffer and throws the weights
	to an arbitrary point in the space of fits every iteration.  Since the net
	also *generates* the next iteration's data, that wandering never averages
	out: the run random-walks instead of converging.

	*steps* <= 0 falls back to *epochs* full passes over the whole buffer.

	*value_weight* scales the MSE value loss relative to the cross-entropy
	policy loss.  Policy logits span 4672 slots and typically produce losses
	~2–5, while the value MSE is <= 1 — without up-weighting, the value head
	barely sees any gradient.

	*amp* runs the convolutions under bf16 autocast.  bf16 rather than fp16
	because it has fp32's exponent range, so no GradScaler and no loss-scale
	tuning is needed; the softmax/MSE reductions stay in fp32 either way since
	autocast keeps them on its fp32 list.
	"""
	n = len(replay_buffer)
	if n < batch_size:
		return None

	data = list(replay_buffer)
	if steps > 0:
		want = steps * batch_size
		if want <= n:
			idx = random.sample(range(n), want)
		else:
			idx = [random.randrange(n) for _ in range(want)]
		data = [data[i] for i in idx]
		epochs = 1
	else:
		random.shuffle(data)

	states = torch.from_numpy(np.array([d[0] for d in data], dtype=np.float32))
	policies = torch.from_numpy(np.array([d[1] for d in data], dtype=np.float32))
	values = torch.from_numpy(np.array([d[2] for d in data], dtype=np.float32))

	dataset = TensorDataset(states, policies, values)
	# pin_memory turns each batch's H2D copy into an async DMA — the policy
	# tensor alone is 4.8 MB per batch of 256, which is slow to move from
	# pageable memory and stalls the GPU between steps.
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
	                    pin_memory=(device.type == "cuda"), drop_last=False)

	use_amp = amp and device.type == "cuda"
	channels_last = device.type == "cuda"

	model.train()
	total_p_loss = 0.0
	total_v_loss = 0.0
	n_batches = 0

	for _epoch in range(epochs):
		for b_states, b_policies, b_values in loader:
			b_states = b_states.to(device, non_blocking=True)
			b_policies = b_policies.to(device, non_blocking=True)
			b_values = b_values.to(device, non_blocking=True)
			if channels_last:
				b_states = b_states.contiguous(memory_format=torch.channels_last)

			with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
				policy_logits, pred_values = model(b_states)
				p_loss = -(b_policies * F.log_softmax(policy_logits.float(), dim=1)
				           ).sum(dim=1).mean()
				v_loss = F.mse_loss(pred_values.float().squeeze(-1), b_values)
				loss = p_loss + value_weight * v_loss

			# set_to_none frees the grad buffers instead of zero-filling them.
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

			total_p_loss += p_loss.item()
			total_v_loss += v_loss.item()
			n_batches += 1

	model.eval()

	return {
		"policy_loss": total_p_loss / n_batches,
		"value_loss": total_v_loss / n_batches,
		"total_loss": (total_p_loss + value_weight * total_v_loss) / n_batches,
	}


def evaluate_examples(model, device, examples, batch_size=512, amp=True):
	"""Policy CE and value MSE on positions the model has *not* trained on.

	Run this on an iteration's fresh self-play positions before the training
	step.  Nothing has reached them yet, so the whole batch is a free held-out
	set and no data has to be withheld from training to get it.

	The gap against the training loss is the only number in the loop that can
	tell learning apart from memorising the buffer -- a training curve alone
	looks healthy either way.  ``value_baseline`` is the MSE of always
	predicting a draw: a value head scoring worse than that is not merely
	uninformative on new positions, it is actively misleading the search.
	"""
	if not examples:
		return None
	model.eval()
	use_amp = amp and device.type == "cuda"
	p_sum = v_sum = 0.0
	n = 0
	with torch.no_grad():
		for i in range(0, len(examples), batch_size):
			chunk = examples[i:i + batch_size]
			s = torch.from_numpy(np.array([e[0] for e in chunk],
			                              dtype=np.float32)).to(device)
			p = torch.from_numpy(np.array([e[1] for e in chunk],
			                              dtype=np.float32)).to(device)
			v = torch.from_numpy(np.array([e[2] for e in chunk],
			                              dtype=np.float32)).to(device)
			if device.type == "cuda":
				s = s.contiguous(memory_format=torch.channels_last)
			with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
				logits, pred = model(s)
			p_sum += float(-(p * F.log_softmax(logits.float(), dim=1)
			                 ).sum(dim=1).sum())
			v_sum += float(F.mse_loss(pred.float().squeeze(-1), v,
			                          reduction="sum"))
			n += len(chunk)
	vals = np.array([e[2] for e in examples], dtype=np.float32)
	return {
		"policy_loss": p_sum / n,
		"value_loss": v_sum / n,
		"value_baseline": float((vals ** 2).mean()),
	}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(
		description="DeepChess — self-play training",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--iterations", type=int, default=100,
	                    help="Number of train iterations to run")
	parser.add_argument("--games-per-iter", type=int, default=200,
	                    help="Self-play games per iteration.  This sets "
	                         "how much genuinely new data each "
	                         "training step sees.  "
	                         "much genuinely new data each training step sees. "
	                         "At 50 games the buffer turned over 4.5%% per "
	                         "iteration, so the net kept re-fitting data it "
	                         "had already fitted; diversity, not target "
	                         "quality, was the binding constraint.")
	parser.add_argument("--simulations", type=int, default=400,
	                    help="MCTS simulations per move during self-play.  This "
	                         "is the main knob on training-target quality: the "
	                         "targets are visit distributions, so too few sims "
	                         "spreads them almost uniformly and they stop "
	                         "carrying information.  Watch the per-iteration "
	                         "target entropy and raise this when it stops "
	                         "falling.  Sims and games-per-iter trade off "
	                         "against each other at fixed wall-clock: spend "
	                         "on games while the run is data-starved, on "
	                         "sims once it is not.")
	parser.add_argument("--mcts-batch", type=int, default=128,
	                    help="MCTS leaf batch size (virtual-loss parallelism). "
	                         "Higher = fewer, larger NN forward passes but less "
	                         "tree diversity.  Measured 83ms/move at 32 vs "
	                         "59ms/move at 128 for 600 sims.")
	parser.add_argument("--workers", type=int, default=0,
	                    help="Self-play worker processes (0 = auto).  Self-play "
	                         "is CPU-bound, so one process leaves the GPU at "
	                         "~6%% utilisation; workers each hold their own fp16 "
	                         "model and play whole games in parallel.")
	parser.add_argument("--seed", type=int, default=1234,
	                    help="Base RNG seed; worker r uses seed + r*7919")
	parser.add_argument("--max-moves", type=int, default=512,
	                    help="Maximum moves per self-play game")
	parser.add_argument("--batch-size", type=int, default=256,
	                    help="Training batch size")
	parser.add_argument("--sample-reuse", type=float, default=3.0,
	                    help="Average number of times each newly generated "
	                         "position is trained on.  Gradient steps per "
	                         "iteration = new_positions * this / batch_size, "
	                         "sampled at random from the whole buffer, so the "
	                         "step budget tracks the rate of new data instead "
	                         "of the buffer size.  0 restores the old "
	                         "full-buffer --epochs behaviour.")
	parser.add_argument("--epochs", type=int, default=1,
	                    help="Training epochs per iteration.  Keep this at 1 "
	                         "unless the buffer is large: with a 50k buffer and "
	                         "5 epochs each position was being replayed ~21 "
	                         "times before ageing out, which fits the buffer "
	                         "rather than the game.  Ignored unless "
	                         "--sample-reuse is 0.")
	parser.add_argument("--lr", type=float, default=0.02,
	                    help="Initial learning rate (SGD+momentum).  Stepped "
	                         "down by --lr-gamma at each --lr-milestones.")
	parser.add_argument("--lr-milestones", type=int, nargs="+",
	                    default=[1500, 3000, 4000],
	                    help="Absolute iteration numbers at which to decay the "
	                         "learning rate by --lr-gamma.  These are absolute, "
	                         "not relative to a resume, so once the last one is "
	                         "behind you every further iteration runs at the "
	                         "fully decayed rate — set them for the whole "
	                         "intended run, not for one invocation.  Set "
	                         "them inside the run: milestones past the "
	                         "iteration count mean a constant LR forever, "
	                         "which keeps the weights moving at full step "
	                         "size long after the targets stop improving.")
	parser.add_argument("--lr-gamma", type=float, default=0.1,
	                    help="Multiplicative LR decay at each milestone.")
	parser.add_argument("--momentum", type=float, default=0.9,
	                    help="SGD momentum")
	parser.add_argument("--weight-decay", type=float, default=1e-4,
	                    help="Weight decay (L2 regularisation)")
	parser.add_argument("--value-weight", type=float, default=4.0,
	                    help="Weight applied to the MSE value loss when summed "
	                         "with the policy cross-entropy.  The two are on "
	                         "very different scales — a converged value MSE "
	                         "around 0.08 against a policy CE around 2.5 leaves "
	                         "the value head with ~3%% of the gradient, which is "
	                         "backwards when the value head is what MCTS needs "
	                         "to tell moves apart.")
	parser.add_argument("--value-discount", type=float, default=1.0,
	                    help="Per-move discount applied to value targets "
	                         "(1.0 = AlphaZero paper; <1 weakens early-game "
	                         "signal where the game outcome is noisier).")
	parser.add_argument("--buffer-size", type=int, default=200000,
	                    help="Replay buffer capacity (positions).  At 50k and "
	                         "~11k new positions per iteration the buffer turned "
	                         "over completely every ~4 iterations and held only "
	                         "~215 distinct games, so consecutive gradient steps "
	                         "saw heavily correlated data.")
	parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINTS_DIR,
	                    help="Directory for model checkpoints")
	parser.add_argument("--checkpoint-every", type=int, default=10,
	                    help="Write a numbered checkpoint every N iterations "
	                         "(latest.pt is refreshed every iteration)")
	parser.add_argument("--res-blocks", type=int, default=16,
	                    help="Residual blocks in the network")
	parser.add_argument("--filters", type=int, default=192,
	                    help="Convolutional filters per layer")
	parser.add_argument("--fpu-reduction", type=float, default=0.25,
	                    help="First-play urgency: an unvisited child's Q is the "
	                         "node's own NN value minus this.  0 gives every "
	                         "unexplored move a flat even-game score, which "
	                         "makes PUCT keep peeling off to fresh moves and "
	                         "flattens the visit-count targets.  Measured sweet "
	                         "spot is 0.25-0.5; above ~1.0 the targets get "
	                         "sharper but the chosen moves get worse.")
	parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
	                    help="Dirichlet concentration for root exploration noise")
	parser.add_argument("--dirichlet-eps", type=float, default=0.25,
	                    help="Weight of the root Dirichlet noise.  This noise is "
	                         "baked into the policy targets, so it costs ~0.1 "
	                         "nats of target sharpness; it buys the self-play "
	                         "diversity that keeps the run from collapsing, so "
	                         "lower it only deliberately.")
	parser.add_argument("--eval-every", type=int, default=25,
	                    help="Run an arena match against the reference net every "
	                         "N iterations (0 disables).  Without this the only "
	                         "signal is training loss, which is precisely the "
	                         "number that still looks healthy at a fixed point.")
	parser.add_argument("--eval-games", type=int, default=200,
	                    help="Games per arena match, colours alternating.  At "
	                         "30 games and the ~85%% draw rate these nets play, "
	                         "one match carries +/-36 Elo of noise at 2 sigma, "
	                         "so a promotion threshold of 55%% fires on chance "
	                         "alone about one match in eight and the generation "
	                         "counter ratchets up on luck.")
	parser.add_argument("--eval-sims", type=int, default=400,
	                    help="MCTS simulations per move during arena games")
	parser.add_argument("--eval-promote", type=float, default=0.55,
	                    help="Score needed to replace the arena reference with "
	                         "the current net and advance a generation")
	parser.add_argument("--gt-every", type=int, default=25,
	                    help="Score the net on the fixed ground-truth suite "
	                         "every N iterations (0 disables).  Unlike the "
	                         "arena this is an absolute yardstick, so it cannot "
	                         "drift with the opponent and has no ratchet.")
	parser.add_argument("--gt-sims", type=int, default=200,
	                    help="MCTS simulations per ground-truth suite position")
	parser.add_argument("--no-amp", dest="amp", action="store_false",
	                    help="Disable bf16 autocast in the training step "
	                         "(kept as an escape hatch; bf16 needs no loss "
	                         "scaling so it should be safe to leave on).")
	parser.add_argument("--from-scratch", action="store_true",
	                    help="Ignore any existing latest.pt and start fresh. "
	                         "Use this after architecture changes so old "
	                         "incompatible checkpoints don't block resume.")
	args = parser.parse_args()

	# Always make sure the checkpoint directory exists.
	os.makedirs(args.checkpoint_dir, exist_ok=True)

	device = get_device()

	if args.workers <= 0:
		args.workers = default_workers(device)

	print(f"Device          : {device}")
	print(f"Native ext      : {'yes' if _ext.AVAILABLE else 'no (pure-Python)'}")
	print(f"Self-play procs : {args.workers}")
	print(f"Checkpoint dir  : {args.checkpoint_dir}")
	print(f"Checkpoint every: {args.checkpoint_every} iteration(s)")

	def _build_optim_and_sched(model, last_iter):
		opt = torch.optim.SGD(
			weight_decay_groups(model, args.weight_decay),
			lr=args.lr,
			momentum=args.momentum,
			nesterov=True,
		)
		if last_iter > 0:
			for pg in opt.param_groups:
				pg.setdefault("initial_lr", pg["lr"])
		sched = torch.optim.lr_scheduler.MultiStepLR(
			opt,
			milestones=args.lr_milestones,
			gamma=args.lr_gamma,
			last_epoch=last_iter - 1 if last_iter > 0 else -1,
		)
		return opt, sched

	# ---- model & optimiser ----
	model = ChessNet(num_res_blocks=args.res_blocks, num_filters=args.filters)
	model.to(device)
	optimizer, scheduler = _build_optim_and_sched(model, 0)

	# ---- resume from checkpoint ----
	start_iter = 0
	generation = 0
	latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
	if os.path.exists(latest_path) and not args.from_scratch:
		ckpt = torch.load(latest_path, map_location=device, weights_only=False)
		# Use architecture from checkpoint when resuming
		saved_res = ckpt.get("num_res_blocks", args.res_blocks)
		saved_fil = ckpt.get("num_filters", args.filters)
		if saved_res != args.res_blocks or saved_fil != args.filters:
			print(f"Checkpoint arch ({saved_res} blocks, {saved_fil} filters) "
			      f"differs from args — using checkpoint arch.")
			args.res_blocks = saved_res
			args.filters = saved_fil
			model = ChessNet(num_res_blocks=saved_res, num_filters=saved_fil)
			model.to(device)
		start_iter = ckpt.get("iteration", 0)
		optimizer, scheduler = _build_optim_and_sched(model, start_iter)
		try:
			model.load_state_dict(ckpt["model_state_dict"])
		except RuntimeError as e:
			raise SystemExit(
				f"\nCheckpoint at {latest_path} is incompatible with the current "
				f"model definition (likely because the architecture changed — "
				f"for example, NUM_PLANES or the value head).\n"
				f"Re-run with --from-scratch, or move the old checkpoints aside.\n\n"
				f"Underlying error:\n  {e}"
			)
		# Optimizer/scheduler state are only reloaded when the optimizer class
		# itself matches — otherwise we silently start with fresh momentum.
		try:
			optimizer.load_state_dict(ckpt["optimizer_state_dict"])
		except (ValueError, KeyError):
			print("Optimizer state incompatible with --optimizer choice — "
			      "starting with a fresh optimizer.")
		sched_state = ckpt.get("scheduler_state_dict")
		if sched_state is not None:
			try:
				scheduler.load_state_dict(sched_state)
			except Exception:
				pass  # milestones may have changed; fall back to fresh schedule
		generation = ckpt.get("generation", 0)
		print(f"Resumed from iteration {start_iter}  |  "
		      f"lr={optimizer.param_groups[0]['lr']:.7f}  |  "
		      f"arena generation {generation}")
		if args.lr_milestones and start_iter >= max(args.lr_milestones):
			# Report the rate actually in the optimizer, not the one the
			# milestones imply: MultiStepLR decays incrementally, so the live
			# value comes from the restored optimizer state and can differ if
			# the milestone list changed between runs.
			cur_lr = optimizer.param_groups[0]["lr"]
			print(
				f"  WARNING: every --lr-milestones entry "
				f"({', '.join(str(m) for m in args.lr_milestones)}) is already "
				f"behind iteration {start_iter}, so no further decay will ever "
				f"be applied and this run — plus every future resume — trains "
				f"at {cur_lr:.7f}, {cur_lr / args.lr:.4g}x the initial rate.  "
				f"Extend --lr-milestones if you did not intend a frozen "
				f"learning rate."
			)
	elif args.from_scratch and os.path.exists(latest_path):
		print("--from-scratch set — ignoring existing checkpoint.")
	else:
		print("No checkpoint found — starting from scratch.")

	# NHWC weights for the training model too — cuDNN's tensor-core conv
	# kernels want channels_last, and load_state_dict/copy_ preserves the
	# destination layout so checkpoints stay format-agnostic.
	if device.type == "cuda":
		model.to(memory_format=torch.channels_last)

	model.eval()

	replay_buffer = deque(maxlen=args.buffer_size)

	# ---- graceful interrupt ----
	interrupted = False

	def _handle_sigint(_sig, _frame):
		nonlocal interrupted
		if interrupted:
			sys.exit(1)
		interrupted = True
		print("\nInterrupt received — finishing current step and saving…")

	signal.signal(signal.SIGINT, _handle_sigint)

	# ---- self-play worker pool ----
	# Workers hold their own fp16 inference copy of the net, so the fp32
	# training weights here stay pristine.  The pool is persistent: CUDA
	# context creation is paid once, not once per iteration.
	pool_cfg = {
		"device": str(device),
		"res_blocks": args.res_blocks,
		"filters": args.filters,
		"simulations": args.simulations,
		"mcts_batch": args.mcts_batch,
		"max_moves": args.max_moves,
		"value_discount": args.value_discount,
		"fpu_reduction": args.fpu_reduction,
		"dirichlet_alpha": args.dirichlet_alpha,
		"dirichlet_eps": args.dirichlet_eps,
		"eval_sims": args.eval_sims,
		"half": device.type in ("cuda", "mps"),
		"seed": args.seed,
		"weights": None,
	}
	weights_path = os.path.join(args.checkpoint_dir, ".selfplay_weights.pt")
	ref_weights_path = os.path.join(args.checkpoint_dir, ".arena_ref.pt")

	# ---- training loop ----
	end_iter = start_iter + args.iterations
	iteration = start_iter

	with SelfPlayPool(args.workers, pool_cfg, weights_path,
	                  ref_weights_path) as pool:
		for iteration in range(start_iter, end_iter):
			if interrupted:
				break

			print(f"\n{'=' * 60}")
			print(f"  Iteration {iteration + 1}  (total target: {end_iter})")
			print(f"{'=' * 60}")

			# -- self-play --
			pool.set_weights(model)
			print(f"Self-play: {args.games_per_iter} games, "
			      f"{args.simulations} sims/move, "
			      f"{args.workers} workers …")
			iter_examples = []
			done = 0
			moves_total = 0
			t0 = time.time()
			width = len(str(args.games_per_iter))
			for examples, result, moves, secs in pool.play(
				args.games_per_iter, stop_early=lambda: interrupted,
			):
				iter_examples.extend(examples)
				done += 1
				moves_total += moves
				print(f"  Game {done:>{width}}/{args.games_per_iter}  "
				      f"moves={moves:<4} result={result:<7} "
				      f"{secs:5.1f}s")

			elapsed = time.time() - t0
			replay_buffer.extend(iter_examples)
			mps_ = moves_total / elapsed if elapsed > 0 else 0.0
			print(f"Self-play done in {elapsed:.1f}s  "
			      f"({done} games, {moves_total} moves, {mps_:.1f} moves/s)  |  "
			      f"Buffer: {len(replay_buffer)} positions")

			pstats = policy_target_stats(iter_examples)
			vstats = value_target_stats(iter_examples)
			if pstats:
				print(f"  Target entropy: {pstats['entropy']:.3f} nats  "
				      f"(uniform-over-legal would be "
				      f"{pstats['uniform_entropy']:.3f})  "
				      f"top1={pstats['top1']:.3f}  "
				      f"legal={pstats['legal']:.1f}")
			if vstats:
				print(f"  Value targets : win {vstats['win']*100:.0f}%  "
				      f"draw {vstats['draw']*100:.0f}%  "
				      f"loss {vstats['loss']*100:.0f}%")

			if interrupted:
				break

			# -- held-out measurement --
			# These positions were produced by this iteration's self-play and have
			# not reached a gradient step yet, so they are a free held-out set.
			# Measure before training, print after, next to the training loss.
			heldout = evaluate_examples(model, device, iter_examples,
			                            amp=args.amp)

			# -- training --
			# Tie the step budget to the rate of new data, not to the buffer size.
			steps = 0
			if args.sample_reuse > 0 and iter_examples:
				steps = max(1, math.ceil(len(iter_examples) * args.sample_reuse
				                         / args.batch_size))
			if steps:
				budget = f"{steps} steps (~{args.sample_reuse:g}x reuse)"
			else:
				budget = f"{args.epochs} full epochs"
			print(f"Training: {budget}, batch {args.batch_size}, "
			      f"lr={optimizer.param_groups[0]['lr']:.5f} …")
			t0 = time.time()
			losses = train_on_data(
				model, optimizer, device, replay_buffer,
				batch_size=args.batch_size, steps=steps, epochs=args.epochs,
				value_weight=args.value_weight, amp=args.amp,
			)
			elapsed = time.time() - t0
			if losses:
				print(f"  Policy loss : {losses['policy_loss']:.4f}")
				if pstats:
					# Cross-entropy bottoms out at the target's own entropy, so
					# this gap is what is actually still learnable.  Near zero
					# means the net already matches its targets and only better
					# targets (more sims, a better value head) can help.
					gap = losses['policy_loss'] - pstats['entropy']
					print(f"                (target entropy "
					      f"{pstats['entropy']:.4f}, headroom {gap:+.4f})")
				print(f"  Value  loss : {losses['value_loss']:.4f}")
				print(f"  Total  loss : {losses['total_loss']:.4f}")
				print(f"  Trained in {elapsed:.1f}s")
			if heldout:
				# The training loss above is fitted; this one is not.  A value MSE
				# above the predict-a-draw baseline means the head is feeding MCTS
				# confident noise on positions it has not seen, which is worse for
				# search than having no value head at all.
				# Only meaningful once some games were decisive: an all-draw batch
				# has a baseline of exactly 0, which nothing can beat.
				flag = ("  <-- WORSE THAN GUESSING"
				        if heldout["value_baseline"] > 0.05
				        and heldout["value_loss"] > heldout["value_baseline"] else "")
				print(f"  Held-out    : policy CE {heldout['policy_loss']:.4f}  "
				      f"value MSE {heldout['value_loss']:.4f}{flag}")
				print(f"                (train {losses['policy_loss']:.4f} / "
				      f"{losses['value_loss']:.4f}; predict-a-draw baseline "
				      f"{heldout['value_baseline']:.4f})")

			# Step the LR scheduler once per iteration regardless of whether a
			# training update happened — this keeps the schedule aligned with the
			# iteration counter across resumes.
			scheduler.step()

			iter_num = iteration + 1

			# -- arena --
			# Training loss cannot tell you whether the net got stronger; a run
			# sitting in a fixed point posts a perfectly stable loss curve
			# forever.  A match against a frozen earlier net can, and promoting
			# the reference only on a win turns "is it improving" into a
			# monotone generation counter.
			if (args.eval_every > 0 and args.eval_games > 0
					and iter_num % args.eval_every == 0):
				if not os.path.exists(ref_weights_path):
					_atomic_save(fp16_state_dict(model), ref_weights_path)
					print(f"Arena reference initialised from iteration "
					      f"{iter_num} (generation {generation}) — "
					      f"first comparison at iteration "
					      f"{iter_num + args.eval_every}.")
				else:
					# Workers still hold the pre-training weights from this
					# iteration's self-play, so republish before measuring.
					pool.set_weights(model)
					pool.publish_ref_from_file(ref_weights_path)
					print(f"Arena: {args.eval_games} games vs generation "
					      f"{generation} reference, "
					      f"{args.eval_sims} sims/move …")
					t0 = time.time()
					scores = []
					wins = draws = losses_n = 0
					for score, _result, _mv, _sec in pool.match(args.eval_games):
						scores.append(score)
						if score > 0.75:
							wins += 1
						elif score < 0.25:
							losses_n += 1
						else:
							draws += 1
					win_rate = sum(scores) / len(scores) if scores else 0.5
					# Wald interval on the mean score; wide at 30 games, which
					# is worth seeing rather than hiding.
					if len(scores) > 1:
						sd = float(np.std(scores, ddof=1))
						ci = 1.96 * sd / math.sqrt(len(scores))
					else:
						ci = 0.0
					print(f"  W-D-L: {wins}-{draws}-{losses_n}   "
					      f"score {win_rate * 100:.1f}% "
					      f"(+/-{ci * 100:.1f})   "
					      f"Elo {elo_diff(win_rate):+.0f}   "
					      f"in {time.time() - t0:.0f}s")
					if win_rate >= args.eval_promote:
						_atomic_save(fp16_state_dict(model), ref_weights_path)
						generation += 1
						print(f"  Promoted — arena generation "
						      f"{generation}.")
					else:
						print(f"  Not promoted (needs "
						      f"{args.eval_promote * 100:.0f}%) — reference "
						      f"stays at generation {generation}.")

			# -- ground-truth suite --
			# The arena only ever compares the net against another copy of itself,
			# so a run that random-walks in place still posts ~50% forever while its
			# generation counter ratchets up on lucky matches.  This suite is fixed
			# and external: its score cannot drift with the opponent, so trending it
			# is the honest answer to whether the net is actually getting stronger.
			if args.gt_every > 0 and iter_num % args.gt_every == 0:
				t0 = time.time()
				passed, total, breakdown = score_model(model, device, args.gt_sims)
				parts = "  ".join(f"{c} {p}/{t}"
				                  for c, (p, t) in breakdown.items())
				print(f"Ground truth: {passed}/{total} "
				      f"({100 * passed / total:.0f}%)   {parts}   "
				      f"in {time.time() - t0:.0f}s")

			# -- checkpoint --
			# Always refresh latest.pt; keep a numbered snapshot only every
			# checkpoint-every iterations (plus the final iteration so nothing
			# is lost at the end of a run).
			keep_numbered = (
				args.checkpoint_every > 0 and
				(iter_num % args.checkpoint_every == 0 or iter_num == end_iter)
			)
			save_checkpoint(
				model, optimizer, scheduler, iter_num, args.checkpoint_dir,
				args.res_blocks, args.filters, numbered=keep_numbered,
				generation=generation,
			)
			if keep_numbered:
				print(f"Checkpoint saved  (iteration {iter_num}, snapshot kept)")
			else:
				print(f"Checkpoint saved  (iteration {iter_num}, latest only)")

	# Final save on interrupt (always numbered so work isn't lost).
	if interrupted:
		save_checkpoint(
			model, optimizer, scheduler, iteration + 1, args.checkpoint_dir,
			args.res_blocks, args.filters, numbered=True,
			generation=generation,
		)
		print(f"Emergency checkpoint saved  (iteration {iteration + 1})")

	print("\nTraining finished.")


if __name__ == "__main__":
	main()
