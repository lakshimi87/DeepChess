#!/usr/bin/env python3
"""DeepChess self-play training pipeline.

Run repeatedly to accumulate training iterations — each invocation
loads the latest checkpoint and continues from where it left off.

    ./train.sh                       # default settings
    ./train.sh --iterations 50       # override iteration count
    ./train.sh --simulations 400     # stronger self-play
"""

import argparse
import os
import random
import signal
import sys
import time
from collections import deque

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import _ext
from .board_utils import encode_board, NUM_MOVES
from .model import ChessNet
from .mcts import MCTS
from .paths import CHECKPOINTS_DIR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def save_checkpoint(model, optimizer, scheduler, iteration, checkpoint_dir,
                    num_res_blocks, num_filters, numbered=True):
	"""Write ``latest.pt`` and (optionally) ``model_iter_XXXX.pt``.

	``latest.pt`` is always refreshed so play.sh/resume always see the most
	recent weights.  ``numbered`` controls whether a permanent snapshot is
	also emitted — the training loop only does this every N iterations to
	keep disk usage bounded.
	"""
	os.makedirs(checkpoint_dir, exist_ok=True)
	payload = {
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
		"iteration": iteration,
		"num_res_blocks": num_res_blocks,
		"num_filters": num_filters,
	}
	latest = os.path.join(checkpoint_dir, "latest.pt")
	torch.save(payload, latest)
	if numbered:
		numbered_path = os.path.join(checkpoint_dir, f"model_iter_{iteration:04d}.pt")
		torch.save(payload, numbered_path)


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def play_game(model, device, num_simulations=400, max_moves=512, mcts_batch=16,
              value_discount=1.0):
	"""Play one self-play game and return training examples.

	Returns a list of (state, policy_target, value_target) tuples.

	*value_discount* in (0, 1] applies ``discount ** (moves_until_end)`` to
	each position's value target — so positions close to the decisive outcome
	carry a stronger signal than those 200 moves earlier where the result is
	far noisier.  Set to 1.0 to reproduce the AlphaZero paper exactly.
	"""
	board = chess.Board()
	mcts = MCTS(model, device, num_simulations=num_simulations,
	            batch_size=mcts_batch)
	history = []  # (encoded_state, policy, turn)

	move_count = 0
	while not board.is_game_over() and move_count < max_moves:
		temperature = 1.0 if move_count < 30 else 0.1
		state = encode_board(board)
		move, policy = mcts.search(
			board, temperature=temperature, add_noise=True,
		)
		if move is None:
			break
		history.append((state, policy, board.turn))
		board.push(move)
		move_count += 1

	# Determine result
	if board.is_checkmate():
		winner = not board.turn  # side to move is mated
	else:
		winner = None  # draw

	examples = []
	total = len(history)
	for i, (state, policy, player) in enumerate(history):
		if winner is None:
			value = 0.0
		elif winner == player:
			value = 1.0
		else:
			value = -1.0
		if value_discount < 1.0:
			moves_remaining = total - i
			value *= value_discount ** moves_remaining
		examples.append((state, policy, value))

	return examples, board.result()


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_on_data(model, optimizer, device, replay_buffer,
                  batch_size=256, epochs=5, value_weight=1.0):
	"""Train model on the replay buffer for *epochs* full passes.

	*value_weight* scales the MSE value loss relative to the cross-entropy
	policy loss.  Policy logits span 4672 slots and typically produce losses
	~2–5, while the value MSE is <= 1 — without up-weighting, the value head
	barely sees any gradient.
	"""
	if len(replay_buffer) < batch_size:
		return None

	data = list(replay_buffer)
	random.shuffle(data)

	states = torch.FloatTensor(np.array([d[0] for d in data]))
	policies = torch.FloatTensor(np.array([d[1] for d in data]))
	values = torch.FloatTensor(np.array([d[2] for d in data]))

	dataset = TensorDataset(states, policies, values)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	model.train()
	total_p_loss = 0.0
	total_v_loss = 0.0
	n_batches = 0

	for _epoch in range(epochs):
		for b_states, b_policies, b_values in loader:
			b_states = b_states.to(device)
			b_policies = b_policies.to(device)
			b_values = b_values.to(device)

			policy_logits, pred_values = model(b_states)

			p_loss = -(b_policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
			v_loss = F.mse_loss(pred_values.squeeze(-1), b_values)
			loss = p_loss + value_weight * v_loss

			optimizer.zero_grad()
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
	parser.add_argument("--games-per-iter", type=int, default=50,
	                    help="Self-play games per iteration")
	parser.add_argument("--simulations", type=int, default=400,
	                    help="MCTS simulations per move during self-play")
	parser.add_argument("--mcts-batch", type=int, default=16,
	                    help="MCTS leaf batch size (virtual-loss parallelism). "
	                         "Higher = fewer NN forward passes but less tree "
	                         "diversity.  8–32 is a good range.")
	parser.add_argument("--max-moves", type=int, default=512,
	                    help="Maximum moves per self-play game")
	parser.add_argument("--batch-size", type=int, default=256,
	                    help="Training batch size")
	parser.add_argument("--epochs", type=int, default=5,
	                    help="Training epochs per iteration")
	parser.add_argument("--lr", type=float, default=0.02,
	                    help="Initial learning rate (SGD+momentum).  Stepped "
	                         "down by --lr-gamma at each --lr-milestones.")
	parser.add_argument("--lr-milestones", type=int, nargs="+",
	                    default=[100, 300, 600],
	                    help="Absolute iteration numbers at which to decay the "
	                         "learning rate by --lr-gamma.")
	parser.add_argument("--lr-gamma", type=float, default=0.1,
	                    help="Multiplicative LR decay at each milestone.")
	parser.add_argument("--momentum", type=float, default=0.9,
	                    help="SGD momentum")
	parser.add_argument("--weight-decay", type=float, default=1e-4,
	                    help="Weight decay (L2 regularisation)")
	parser.add_argument("--value-weight", type=float, default=1.0,
	                    help="Weight applied to the MSE value loss when summed "
	                         "with the policy cross-entropy.  Bump above 1.0 if "
	                         "the value head still underfits.")
	parser.add_argument("--value-discount", type=float, default=1.0,
	                    help="Per-move discount applied to value targets "
	                         "(1.0 = AlphaZero paper; <1 weakens early-game "
	                         "signal where the game outcome is noisier).")
	parser.add_argument("--buffer-size", type=int, default=50000,
	                    help="Replay buffer capacity (positions)")
	parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINTS_DIR,
	                    help="Directory for model checkpoints")
	parser.add_argument("--checkpoint-every", type=int, default=10,
	                    help="Write a numbered checkpoint every N iterations "
	                         "(latest.pt is refreshed every iteration)")
	parser.add_argument("--res-blocks", type=int, default=10,
	                    help="Residual blocks in the network")
	parser.add_argument("--filters", type=int, default=128,
	                    help="Convolutional filters per layer")
	parser.add_argument("--from-scratch", action="store_true",
	                    help="Ignore any existing latest.pt and start fresh. "
	                         "Use this after architecture changes so old "
	                         "incompatible checkpoints don't block resume.")
	args = parser.parse_args()

	# Always make sure the checkpoint directory exists.
	os.makedirs(args.checkpoint_dir, exist_ok=True)

	device = get_device()
	print(f"Device          : {device}")
	print(f"Native ext      : {'yes' if _ext.AVAILABLE else 'no (pure-Python)'}")
	print(f"Checkpoint dir  : {args.checkpoint_dir}")
	print(f"Checkpoint every: {args.checkpoint_every} iteration(s)")

	def _build_optim_and_sched(model, last_iter):
		opt = torch.optim.SGD(
			model.parameters(),
			lr=args.lr,
			momentum=args.momentum,
			nesterov=True,
			weight_decay=args.weight_decay,
		)
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
		print(f"Resumed from iteration {start_iter}  |  "
		      f"lr={optimizer.param_groups[0]['lr']:.5f}")
	elif args.from_scratch and os.path.exists(latest_path):
		print("--from-scratch set — ignoring existing checkpoint.")
	else:
		print("No checkpoint found — starting from scratch.")

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

	# ---- training loop ----
	end_iter = start_iter + args.iterations
	iteration = start_iter
	for iteration in range(start_iter, end_iter):
		if interrupted:
			break

		print(f"\n{'=' * 60}")
		print(f"  Iteration {iteration + 1}  (total target: {end_iter})")
		print(f"{'=' * 60}")

		# -- self-play --
		print(f"Self-play: {args.games_per_iter} games, "
		      f"{args.simulations} sims/move …")
		iter_examples = []
		t0 = time.time()
		for g in range(args.games_per_iter):
			if interrupted:
				break
			examples, result = play_game(
				model, device,
				num_simulations=args.simulations,
				max_moves=args.max_moves,
				mcts_batch=args.mcts_batch,
				value_discount=args.value_discount,
			)
			iter_examples.extend(examples)
			print(f"  Game {g + 1:>{len(str(args.games_per_iter))}}"
			      f"/{args.games_per_iter}  "
			      f"moves={len(examples):<4} result={result}")

		elapsed = time.time() - t0
		replay_buffer.extend(iter_examples)
		print(f"Self-play done in {elapsed:.1f}s  |  "
		      f"Buffer: {len(replay_buffer)} positions")

		if interrupted:
			break

		# -- training --
		print(f"Training: {args.epochs} epochs, batch {args.batch_size}, "
		      f"lr={optimizer.param_groups[0]['lr']:.5f} …")
		t0 = time.time()
		losses = train_on_data(
			model, optimizer, device, replay_buffer,
			batch_size=args.batch_size, epochs=args.epochs,
			value_weight=args.value_weight,
		)
		elapsed = time.time() - t0
		if losses:
			print(f"  Policy loss : {losses['policy_loss']:.4f}")
			print(f"  Value  loss : {losses['value_loss']:.4f}")
			print(f"  Total  loss : {losses['total_loss']:.4f}")
			print(f"  Trained in {elapsed:.1f}s")

		# Step the LR scheduler once per iteration regardless of whether a
		# training update happened — this keeps the schedule aligned with the
		# iteration counter across resumes.
		scheduler.step()

		# -- checkpoint --
		# Always refresh latest.pt; keep a numbered snapshot only every
		# checkpoint-every iterations (plus the final iteration so nothing
		# is lost at the end of a run).
		iter_num = iteration + 1
		keep_numbered = (
			args.checkpoint_every > 0 and
			(iter_num % args.checkpoint_every == 0 or iter_num == end_iter)
		)
		save_checkpoint(
			model, optimizer, scheduler, iter_num, args.checkpoint_dir,
			args.res_blocks, args.filters, numbered=keep_numbered,
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
		)
		print(f"Emergency checkpoint saved  (iteration {iteration + 1})")

	print("\nTraining finished.")


if __name__ == "__main__":
	main()
