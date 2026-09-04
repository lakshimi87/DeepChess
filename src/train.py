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


def checkpoint_payload(model, optimizer, scheduler, iteration,
                      num_res_blocks, num_filters, generation=0,
                      optimizer_name=""):
	"""The dict every checkpoint in this project stores.

	``optimizer_name`` is recorded so a resume can tell whether the saved
	optimiser state belongs to the optimiser now being built.  Without it,
	switching --optimizer silently loaded one optimiser's moments into
	another's slots, or dropped them and started at full step size on a net
	that had converged at a thousandth of it.
	"""
	return {
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
		"iteration": iteration,
		"num_res_blocks": num_res_blocks,
		"num_filters": num_filters,
		"generation": generation,
		"optimizer": optimizer_name,
	}


def save_checkpoint(model, optimizer, scheduler, iteration, checkpoint_dir,
                    num_res_blocks, num_filters, numbered=True, generation=0,
                    optimizer_name=""):
	"""Write ``latest.pt`` and (optionally) ``model_iter_XXXX.pt``.

	``latest.pt`` is always refreshed so play.sh/resume always see the most
	recent weights.  ``numbered`` controls whether a permanent snapshot is
	also emitted — the training loop only does this every N iterations to
	keep disk usage bounded.

	Both writes go through :func:`_atomic_save`, so a reader is never exposed
	to a partially written checkpoint.
	"""
	os.makedirs(checkpoint_dir, exist_ok=True)
	payload = checkpoint_payload(
		model, optimizer, scheduler, iteration, num_res_blocks, num_filters,
		generation=generation, optimizer_name=optimizer_name,
	)
	if numbered:
		numbered_path = os.path.join(checkpoint_dir, f"model_iter_{iteration:04d}.pt")
		_atomic_save(payload, numbered_path)
	_atomic_save(payload, os.path.join(checkpoint_dir, "latest.pt"))


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_on_data(model, optimizer, device, replay_buffer,
                  batch_size=256, steps=0, epochs=1, value_weight=1.0,
                  amp=True, extra_examples=None):
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

	*extra_examples* are mixed into the sample and count against the same step
	budget, so adding them trades self-play data for anchor data rather than
	buying extra gradient steps.  See --anchor-data-dir for why the loop needs
	a target that is not a function of its own output.
	"""
	n = len(replay_buffer)
	if n < batch_size:
		return None

	extra = list(extra_examples) if extra_examples else []
	data = list(replay_buffer)
	if steps > 0:
		want = steps * batch_size
		# Trim the anchor rather than growing the budget, and never let it
		# crowd the replay buffer below one batch: self-play data is what the
		# loop is supposed to be learning from, the anchor only holds it in
		# place.
		if extra:
			extra = extra[:max(0, want - batch_size)]
		want -= len(extra)
		if want <= n:
			idx = random.sample(range(n), want)
		else:
			idx = [random.randrange(n) for _ in range(want)]
		data = [data[i] for i in idx]
		epochs = 1
	else:
		random.shuffle(data)
	data.extend(extra)

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
	abs_sum = 0.0
	abs_max = 0.0
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
			pred_v = pred.float().squeeze(-1)
			p_sum += float(-(p * F.log_softmax(logits.float(), dim=1)
			                 ).sum(dim=1).sum())
			v_sum += float(F.mse_loss(pred_v, v, reduction="sum"))
			abs_sum += float(pred_v.abs().sum())
			abs_max = max(abs_max, float(pred_v.abs().max()))
			n += len(chunk)
	vals = np.array([e[2] for e in examples], dtype=np.float32)
	return {
		"policy_loss": p_sum / n,
		"value_loss": v_sum / n,
		"value_baseline": float((vals ** 2).mean()),
		# The value head's output *scale*, which no other number in the loop
		# exposes.  A collapsing head keeps posting a falling value loss --
		# converging on the degenerate constant looks exactly like learning --
		# while mean|V| walks to zero.  It is also what decides whether an
		# absolute resign threshold is still reachable at all.
		"pred_abs_mean": abs_sum / n,
		"pred_abs_max": abs_max,
	}


# ---------------------------------------------------------------------------
# Supervised anchor
# ---------------------------------------------------------------------------

# Reproduce src/pretrain.py's targets exactly.  The anchor is only worth
# anything if it is the same objective the net was pre-trained on; a
# differently-shaped policy target would pull the head somewhere new rather
# than holding it where the supervised phase left it.
ANCHOR_SF_WEIGHT = 0.60
ANCHOR_HUMAN_WEIGHT = 0.30
ANCHOR_POLICY_FLOOR = 0.10
ANCHOR_RESULT_WEIGHT = 0.15


def load_anchor_dataset(data_dir, max_rows=0, seed=1234):
	"""Index the Stockfish-labelled shards for use as a training anchor.

	Every target in the self-play loop is a function of the net's own output:
	the policy target is its search's visit counts, the value target is the
	game its own moves produced, and the search value blended into that target
	is literally its own value head.  Nothing in the loop is anchored to a
	quantity from outside it, so once the net drifts there is no gradient
	anywhere that pulls back — run5 lost 91% -> 38% on the ground-truth suite
	while every loss it could see fell monotonically.

	These shards are the one quantity in the project that does not move.
	Mixing a fraction of them into every training step turns the supervised
	solution from a starting point into a constraint.
	"""
	from .pretrain import LabelledPositions

	paths = sorted(os.path.join(data_dir, f)
	               for f in os.listdir(data_dir) if f.endswith(".tsv"))
	if not paths:
		raise SystemExit(f"No label shards in {data_dir}")
	ds = LabelledPositions(paths, ANCHOR_SF_WEIGHT, ANCHOR_HUMAN_WEIGHT,
	                       ANCHOR_POLICY_FLOOR, ANCHOR_RESULT_WEIGHT)
	if max_rows and len(ds) > max_rows:
		# The row index is ~12 bytes/row, so a full 70M-row corpus costs ~1 GB
		# of resident memory next to a replay buffer that is already the
		# largest allocation in the run.  Subsampling the index keeps the
		# anchor's diversity far above anything self-play produces while
		# bounding that cost.
		rng = np.random.default_rng(seed)
		keep = np.sort(rng.choice(len(ds), size=max_rows, replace=False))
		ds = LabelledPositions(
			paths, ANCHOR_SF_WEIGHT, ANCHOR_HUMAN_WEIGHT,
			ANCHOR_POLICY_FLOOR, ANCHOR_RESULT_WEIGHT,
			index=(ds.shard_id[keep], ds.offset[keep], ds.bucket[keep]))
	return ds, len(paths)


def sample_anchor(dataset, count):
	"""Draw *count* random labelled positions in replay-buffer tuple form."""
	if dataset is None or count <= 0:
		return []
	idx = np.random.randint(0, len(dataset), size=count)
	return [dataset[int(i)] for i in idx]


def ref_matches_arch(path, num_res_blocks, num_filters):
	"""Whether the fp16 arena reference at *path* fits the current net.

	``.arena_ref.pt`` is not versioned and not architecture-tagged, so a file
	left behind by an earlier run of a different size sits there looking
	valid.  run5 carried an 8x128 reference from run4 into a 16x192 run: the
	arena would have killed every worker on a shape mismatch the first time it
	fired, which is to say the one guard rail that could have caught the
	collapse was broken before the run started.
	"""
	if not os.path.exists(path):
		return False
	try:
		state = torch.load(path, map_location="cpu", weights_only=True)
		probe = ChessNet(num_res_blocks=num_res_blocks, num_filters=num_filters)
		probe.load_state_dict({k: v.float() for k, v in state.items()})
	except Exception:
		return False
	return True


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
	parser.add_argument("--resign-threshold", type=float, default=-0.9,
	                    help="Resign when the mover's own root Q stays at or "
	                         "below this for --resign-plies of its own turns. "
	                         "0 or above disables resignation. Decided games "
	                         "that instead grind to the 50-move rule are "
	                         "scored as draws, which mislabels every position "
	                         "in them")
	parser.add_argument("--resign-plies", type=int, default=2,
	                    help="Consecutive turns by one side below "
	                         "--resign-threshold before it resigns")
	parser.add_argument("--resign-disable-frac", type=float, default=0.1,
	                    help="Fraction of self-play games that ignore both "
	                         "resignation and adjudication and play on; the "
	                         "only way to see a false positive, so keep it "
	                         "above 0")
	parser.add_argument("--adjudicate-material", type=float, default=5.0,
	                    help="Award the game to a side that has held this "
	                         "material lead, in pawns, for "
	                         "--adjudicate-plies plies.  0 disables.  This is "
	                         "the only decisive-label source in the loop that "
	                         "does not read the network: --resign-threshold is "
	                         "an absolute root-Q bound, so it silently stops "
	                         "firing the moment the value head's output scale "
	                         "shrinks below it, and the draw-labelled games "
	                         "that result are what feed the collapse.")
	parser.add_argument("--adjudicate-plies", type=int, default=20,
	                    help="Plies the material lead must persist before "
	                         "adjudication fires.  Long enough that a "
	                         "temporary sacrifice does not score the game, "
	                         "short enough to stop a decided game grinding to "
	                         "the 50-move rule and mislabelling every position "
	                         "in it as a draw.")
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
	parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd",
	                    help="Optimiser for the training step.  Use adamw to "
	                         "continue from src/pretrain.py, which is AdamW: "
	                         "handing a net that converged under one optimiser "
	                         "to another with no state and no warmup is a step "
	                         "discontinuity, not a resume.")
	parser.add_argument("--lr", type=float, default=None,
	                    help="Initial learning rate.  Defaults to 0.02 for "
	                         "sgd and 1e-4 for adamw — the sgd value is three "
	                         "orders of magnitude above where pre-training "
	                         "ends (OneCycle to 1e-5), so carrying it over "
	                         "unchanged is what turned run5's collapse into a "
	                         "single-iteration cliff.  Stepped down by "
	                         "--lr-gamma at each --lr-milestones.")
	parser.add_argument("--warmup-iters", type=int, default=0,
	                    help="Ramp the LR linearly from 1/N to full over the "
	                         "first N iterations.  Matters most on the first "
	                         "iteration after pre-training, which trains on a "
	                         "buffer holding one iteration of self-play — the "
	                         "smallest and most correlated sample the run will "
	                         "ever see, at the largest step size.")
	parser.add_argument("--lr-milestones", type=int, nargs="+",
	                    default=[1500, 3000, 4000],
	                    help="Absolute iteration numbers at which to decay the "
	                         "learning rate by --lr-gamma, counted against the "
	                         "iteration number the log prints — a milestone of "
	                         "1500 first applies on the iteration labelled "
	                         "1500.  (MultiStepLR, which this replaced, "
	                         "applied it one iteration later.)  These are "
	                         "absolute, "
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
	parser.add_argument("--search-value-weight", type=float, default=0.35,
	                    help="Fraction of each position's value target taken "
	                         "from its own MCTS root Q instead of the game "
	                         "outcome.  A pure outcome label is identical for "
	                         "every position in a game; root Q varies ply to "
	                         "ply, so two decided games stop sharing one "
	                         "label per side.  Applied to decided games only: "
	                         "the MSE fixed point of a target "
	                         "(1-w)*outcome + w*V is V = outcome whatever w "
	                         "is, so on a draw this does not escape the "
	                         "constant-0 fixed point, it only contracts V "
	                         "toward 0 by w per iteration out of the net's own "
	                         "output.  Keeping the draw rate down is "
	                         "--adjudicate-material's job, not this flag's.  "
	                         "Only meaningful once the value head carries "
	                         "signal — set 0 when training from a random "
	                         "init, non-zero after src/pretrain.py.  Games "
	                         "truncated at --max-moves always take the search "
	                         "value outright: they are unfinished, not drawn.")
	parser.add_argument("--value-discount", type=float, default=1.0,
	                    help="Per-move discount applied to value targets "
	                         "(1.0 = AlphaZero paper; <1 weakens early-game "
	                         "signal where the game outcome is noisier).")
	parser.add_argument("--buffer-size", type=int, default=1000000,
	                    help="Replay buffer capacity (positions).  This is the "
	                         "loop's largest single deficit against a working "
	                         "AlphaZero run: AlphaGo Zero sampled from its most "
	                         "recent 500,000 *games*, while a 100k-position "
	                         "window here holds ~820 -- three orders of "
	                         "magnitude less diversity, so consecutive gradient "
	                         "steps see nearly the same data and the value head "
	                         "memorises it (run4 held-out value MSE ran 2x its "
	                         "training MSE for 258 iterations).  Unlike every "
	                         "other axis of that gap, this one costs RAM rather "
	                         "than GPU time: at ~24k new positions per "
	                         "iteration, 1M holds ~40 iterations for a few GB.")
	parser.add_argument("--anchor-data-dir", type=str, default="",
	                    help="Directory of Stockfish label shards (the "
	                         "--data-dir given to src/pretrain.py) to mix into "
	                         "every training step.  Every other target in the "
	                         "loop is a function of the net's own output, so "
	                         "nothing pulls the weights back once they drift; "
	                         "these rows are the only fixed quantity "
	                         "available, and they cost RAM and CPU rather than "
	                         "GPU time.  Empty disables anchoring.")
	parser.add_argument("--anchor-frac", type=float, default=0.25,
	                    help="Fraction of each iteration's training sample "
	                         "drawn from --anchor-data-dir instead of the "
	                         "replay buffer.  Counts against the same step "
	                         "budget, so this trades self-play data for anchor "
	                         "data rather than adding steps.  Ignored without "
	                         "--anchor-data-dir.")
	parser.add_argument("--anchor-max-rows", type=int, default=8_000_000,
	                    help="Cap on indexed anchor rows (~12 bytes each).  "
	                         "0 indexes the whole corpus, which is ~1 GB of "
	                         "resident index for a 70M-row one.")
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
	parser.add_argument("--gt-abort-drop", type=float, default=0.10,
	                    help="Stop the run and restore the best checkpoint "
	                         "once the ground-truth score falls this fraction "
	                         "below the best seen (0.10 = a 10%% relative "
	                         "drop).  0 disables.  Every diagnostic in run5 "
	                         "fired correctly — held-out value MSE was flagged "
	                         "worse than guessing from iteration 1 and the "
	                         "suite read 38%% against a 91%% start — and none "
	                         "of them did anything but print, so the run kept "
	                         "training on its own collapse.  Enabling this "
	                         "measures the suite once before the first "
	                         "iteration to establish the baseline.")
	parser.add_argument("--no-amp", dest="amp", action="store_false",
	                    help="Disable bf16 autocast in the training step "
	                         "(kept as an escape hatch; bf16 needs no loss "
	                         "scaling so it should be safe to leave on).")
	parser.add_argument("--from-scratch", action="store_true",
	                    help="Ignore any existing latest.pt and start fresh. "
	                         "Use this after architecture changes so old "
	                         "incompatible checkpoints don't block resume.")
	args = parser.parse_args()

	if args.lr is None:
		# Resolved here rather than as an argparse default so that "not given"
		# stays distinguishable: an SGD rate handed to AdamW is not a slightly
		# aggressive setting, it diverges.
		args.lr = 1e-4 if args.optimizer == "adamw" else 0.02

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
	print(f"Optimizer       : {args.optimizer}  lr={args.lr:g}"
	      + (f"  warmup={args.warmup_iters} iter(s)"
	         if args.warmup_iters > 0 else "  (no warmup)"))
	print("Adjudication    : "
	      + (f"{args.adjudicate_material:g} pawns for "
	         f"{args.adjudicate_plies} plies"
	         if args.adjudicate_material > 0 and args.adjudicate_plies > 0
	         else "off"))

	def _lr_scale(it):
		"""LR multiplier at 0-based iteration index *it*.

		Warmup times the milestone decay.  Expressing the whole schedule as a
		pure function of the iteration number is what makes it resume-safe:
		MultiStepLR decays incrementally, so its live rate depended on
		restored optimiser state and quietly disagreed with the milestone list
		whenever that list changed between runs.
		"""
		warm = 1.0
		if args.warmup_iters > 0:
			warm = min(1.0, (it + 1) / args.warmup_iters)
		decay = args.lr_gamma ** sum(1 for m in args.lr_milestones
		                             if it + 1 >= m)
		return warm * decay

	def _build_optim_and_sched(model, last_iter):
		groups = weight_decay_groups(model, args.weight_decay)
		if args.optimizer == "adamw":
			opt = torch.optim.AdamW(groups, lr=args.lr)
		else:
			opt = torch.optim.SGD(
				groups,
				lr=args.lr,
				momentum=args.momentum,
				nesterov=args.momentum > 0,
			)
		for pg in opt.param_groups:
			pg.setdefault("initial_lr", pg["lr"])
		sched = torch.optim.lr_scheduler.LambdaLR(
			opt,
			lr_lambda=_lr_scale,
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
		# Optimizer state is only reloaded when the optimiser that wrote it is
		# the one being built now.  Checkpoints from src/pretrain.py carry no
		# optimiser state at all, so this branch is also the pre-training
		# hand-off: there is nothing to restore and the first iteration starts
		# at full step size unless --warmup-iters says otherwise.
		saved_optimizer = ckpt.get("optimizer", "")
		if "optimizer_state_dict" not in ckpt:
			print(f"Checkpoint carries no optimizer state (pre-trained net?) "
			      f"— starting {args.optimizer} fresh."
			      + ("" if args.warmup_iters > 0 else
			         "  Consider --warmup-iters."))
		elif saved_optimizer and saved_optimizer != args.optimizer:
			print(f"Checkpoint optimizer ({saved_optimizer}) differs from "
			      f"--optimizer {args.optimizer} — starting fresh.")
		else:
			try:
				optimizer.load_state_dict(ckpt["optimizer_state_dict"])
			except (ValueError, KeyError, TypeError) as exc:
				print(f"Optimizer state could not be restored ({exc}) — "
				      f"starting {args.optimizer} fresh.")
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
		"search_value_weight": args.search_value_weight,
		"resign_threshold": args.resign_threshold,
		"resign_plies": args.resign_plies,
		"resign_disable_frac": args.resign_disable_frac,
		"adjudicate_material": args.adjudicate_material,
		"adjudicate_plies": args.adjudicate_plies,
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
	best_gt_path = os.path.join(args.checkpoint_dir, "best_gt.pt")

	# ---- supervised anchor ----
	anchor_ds = None
	if args.anchor_data_dir and args.anchor_frac > 0.0:
		t0 = time.time()
		anchor_ds, n_shards = load_anchor_dataset(
			args.anchor_data_dir, args.anchor_max_rows, args.seed)
		print(f"Anchor          : {len(anchor_ds):,} rows from {n_shards} "
		      f"shard(s), {args.anchor_frac:.0%} of each step "
		      f"(indexed in {time.time() - t0:.1f}s)")
	elif args.anchor_frac > 0.0:
		print("Anchor          : off (no --anchor-data-dir).  Every training "
		      "target is now a function of the net's own output.")

	# ---- arena reference ----
	# Established before the first iteration, not on the arena's first firing.
	# Initialising it there cost a full --eval-every of extra delay before any
	# comparison could happen (25 iterations became 50), which is long past
	# the point where a collapse has already consumed the run.
	if args.eval_every > 0 and args.eval_games > 0:
		# The arena fires on iter_num % eval_every, so the first comparison is
		# the next multiple of eval_every past start_iter, not start_iter plus
		# eval_every.
		next_arena = (start_iter // args.eval_every + 1) * args.eval_every
		if ref_matches_arch(ref_weights_path, args.res_blocks, args.filters):
			print(f"Arena reference : {ref_weights_path} (generation "
			      f"{generation}), first comparison at iteration "
			      f"{next_arena}")
		else:
			stale = os.path.exists(ref_weights_path)
			_atomic_save(fp16_state_dict(model), ref_weights_path)
			print("Arena reference : "
			      + ("replaced — the existing file does not match "
			         f"{args.res_blocks}x{args.filters} "
			         if stale else "initialised ")
			      + f"from iteration {start_iter} (generation {generation}), "
			      f"first comparison at iteration {next_arena}")

	# ---- ground-truth baseline ----
	# The abort rule needs a score from *before* any self-play update, so the
	# thing it protects (the pre-trained net) is what it compares against.
	best_gt = None
	if args.gt_every > 0 and args.gt_abort_drop > 0.0:
		t0 = time.time()
		passed, total, _breakdown = score_model(model, device, args.gt_sims)
		best_gt = passed / total
		print(f"Ground truth    : {passed}/{total} ({best_gt * 100:.0f}%) at "
		      f"iteration {start_iter} — abort below "
		      f"{best_gt * (1 - args.gt_abort_drop) * 100:.0f}%  "
		      f"in {time.time() - t0:.0f}s")
		_atomic_save(
			checkpoint_payload(model, optimizer, scheduler, start_iter,
			                   args.res_blocks, args.filters,
			                   generation=generation,
			                   optimizer_name=args.optimizer),
			best_gt_path)

	# ---- training loop ----
	end_iter = start_iter + args.iterations
	iteration = start_iter
	aborted = False

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
			resigned = 0
			adjudicated = 0
			truncated = 0
			t0 = time.time()
			width = len(str(args.games_per_iter))
			for examples, result, moves, secs in pool.play(
				args.games_per_iter, stop_early=lambda: interrupted,
			):
				iter_examples.extend(examples)
				done += 1
				moves_total += moves
				if result.endswith(" R"):
					resigned += 1
				elif result.endswith(" A"):
					adjudicated += 1
				elif result == "*":
					truncated += 1
				print(f"  Game {done:>{width}}/{args.games_per_iter}  "
				      f"moves={moves:<4} result={result:<7} "
				      f"{secs:5.1f}s")

			elapsed = time.time() - t0
			replay_buffer.extend(iter_examples)
			mps_ = moves_total / elapsed if elapsed > 0 else 0.0
			print(f"Self-play done in {elapsed:.1f}s  "
			      f"({done} games, {moves_total} moves, {mps_:.1f} moves/s)  |  "
			      f"Buffer: {len(replay_buffer)} positions")
			if done:
				# Truncated games are the ones still labelled a draw despite
				# being decided — the noise resignation is there to remove.
				# Watch it fall as the value head becomes usable; if it stays
				# high the threshold is never being reached and the value head
				# is still the bottleneck.
				print(f"  Game endings  : resigned {resigned}/{done} "
				      f"({resigned / done * 100:.0f}%)  "
				      f"adjudicated {adjudicated}/{done} "
				      f"({adjudicated / done * 100:.0f}%)  "
				      f"hit move limit {truncated}/{done} "
				      f"({truncated / done * 100:.0f}%)")

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
			n_anchor = 0
			if anchor_ds is not None and steps:
				n_anchor = int(round(steps * args.batch_size * args.anchor_frac))
			anchor_examples = []
			if n_anchor:
				t0 = time.time()
				anchor_examples = sample_anchor(anchor_ds, n_anchor)
				budget += (f", {n_anchor} anchor rows "
				           f"({time.time() - t0:.1f}s)")
			print(f"Training: {budget}, batch {args.batch_size}, "
			      f"lr={optimizer.param_groups[0]['lr']:.5f} …")
			t0 = time.time()
			losses = train_on_data(
				model, optimizer, device, replay_buffer,
				batch_size=args.batch_size, steps=steps, epochs=args.epochs,
				value_weight=args.value_weight, amp=args.amp,
				extra_examples=anchor_examples,
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
				# train_on_data returns None when the buffer is still smaller
				# than one batch, which is the normal state of the first
				# iteration or two of a fresh run.
				trained = (f"train {losses['policy_loss']:.4f} / "
				           f"{losses['value_loss']:.4f}; " if losses
				           else "no training step yet; ")
				print(f"                ({trained}predict-a-draw baseline "
				      f"{heldout['value_baseline']:.4f})")
				# The number that makes a collapsing value head visible while
				# every loss is still falling.  Measured on this iteration's
				# own positions, before the update that follows.
				print(f"  Value scale : mean|V| "
				      f"{heldout['pred_abs_mean']:.4f}  max|V| "
				      f"{heldout['pred_abs_max']:.4f}")
				if (args.resign_threshold < 0.0
						and heldout["pred_abs_max"] < abs(args.resign_threshold)):
					print(f"                WARNING: max|V| is below "
					      f"|--resign-threshold| ({abs(args.resign_threshold):.2f}), "
					      f"so resignation can no longer fire at all"
					      + ("; adjudication is carrying the decisive labels."
					         if args.adjudicate_material > 0 else
					         " and nothing else supplies decisive labels."))

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
				if not ref_matches_arch(ref_weights_path, args.res_blocks,
				                        args.filters):
					_atomic_save(fp16_state_dict(model), ref_weights_path)
					print(f"Arena reference re-initialised from iteration "
					      f"{iter_num} (generation {generation}) — "
					      f"next comparison at iteration "
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
				score = passed / total
				print(f"Ground truth: {passed}/{total} "
				      f"({100 * score:.0f}%)   {parts}   "
				      f"in {time.time() - t0:.0f}s")
				if args.gt_abort_drop > 0.0:
					if best_gt is None or score > best_gt:
						best_gt = score
						_atomic_save(
							checkpoint_payload(
								model, optimizer, scheduler, iter_num,
								args.res_blocks, args.filters,
								generation=generation,
								optimizer_name=args.optimizer),
							best_gt_path)
						print(f"  New best — kept at {best_gt_path}")
					elif score < best_gt * (1.0 - args.gt_abort_drop):
						# Stop before the checkpoint write below, so latest.pt
						# is not carrying the collapsed weights when the
						# rollback lands on it.
						print(f"  ABORT: {100 * score:.0f}% is "
						      f"{100 * (1 - score / best_gt):.0f}% below the "
						      f"run best {100 * best_gt:.0f}% — the loop is "
						      f"training on its own degradation.")
						if os.path.exists(best_gt_path):
							best = torch.load(best_gt_path, map_location="cpu",
							                  weights_only=False)
							_atomic_save(best, latest_path)
							print(f"  Restored the best checkpoint (iteration "
							      f"{best.get('iteration', '?')}) to "
							      f"{latest_path}.")
						aborted = True
						break

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
				generation=generation, optimizer_name=args.optimizer,
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
			generation=generation, optimizer_name=args.optimizer,
		)
		print(f"Emergency checkpoint saved  (iteration {iteration + 1})")

	if aborted:
		raise SystemExit(
			"\nAborted on a ground-truth regression.  latest.pt holds the "
			"best checkpoint of the run; diagnose before resuming, because a "
			"plain resume will walk straight back into it."
		)

	print("\nTraining finished.")


if __name__ == "__main__":
	main()
