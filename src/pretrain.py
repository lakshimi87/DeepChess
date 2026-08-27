"""Supervised pre-training on Stockfish-labelled positions.

    python -m src.pretrain --data-dir data/bootstrap/labels --epochs 3

Self-play from scratch on one GPU does not reach the regime where AlphaZero's
loop works.  The run this replaces spent 20 hours to produce 6.1M positions
and held a 100k-position replay window -- roughly 1/1000th of AlphaGo Zero's
500k-*game* window -- and its value head converged to the constant a
draw-dominated corpus makes optimal.  Target entropy stopped moving at
iteration 75 and the ground-truth suite random-walked for the remaining 175.

This module breaks that fixed point from outside the loop: it fits the net to
positions labelled by an engine that is already strong, so self-play restarts
from a value head that carries real information instead of one that has to
discover it from its own noise.

Targets
-------
**Value** comes from Stockfish's WDL at the labelling depth, as
``2*expectation - 1``, from the side-to-move's point of view -- matching
``encode_board``, which renders every position with the mover as white.
Because it is per-position, two drawn games no longer collapse to the same
label, which is precisely what the self-play loop could not arrange for
itself.  ``--result-weight`` blends in the game's actual outcome: a little of
it keeps some practical-play signal that a pure engine eval discards, and the
self-play phase afterwards reintroduces the rest.

**Policy** mixes Stockfish's best move, the move the human actually played,
and a uniform floor over the legal moves.  The floor is not cosmetic: MCTS
explores in proportion to the prior, so a move trained to exactly zero is one
search will never look at again, and a one-hot target teaches exactly that.
Keeping the human move alongside the engine's is what stops the policy from
collapsing onto a single line -- across the corpus the two disagree often
enough to leave the net a distribution rather than a lookup table.
"""
import argparse
import os
import time
import zlib

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .board_utils import NUM_MOVES, encode_board, get_legal_move_indices
from .model import ChessNet
from .paths import CHECKPOINTS_DIR


def _wdl_value(w, d, ll):
	"""WDL permille -> expected score in [-1, 1], side-to-move POV."""
	total = w + d + ll
	if total <= 0:
		return 0.0
	return (w - ll) / total


class LabelledPositions(Dataset):
	"""Random access over label shards, encoding boards on demand.

	An encoded position is a 20x8x8 float32 tensor -- 5 KB -- so materialising
	20M of them would need 100 GB.  The shards keep FENs instead (~60 bytes)
	and this re-encodes through the C++ extension in the DataLoader workers,
	which turns a storage problem into spare CPU time.
	"""

	def __init__(self, paths, sf_weight, human_weight, floor,
	             result_weight, index=None):
		self.paths = paths
		self.sf_weight = sf_weight
		self.human_weight = human_weight
		self.floor = floor
		self.result_weight = result_weight
		self._fh = {}
		if index is not None:
			self.shard_id, self.offset, self.bucket = index
			return

		# One sequential pass builds the row index and the split bucket
		# together.  Computing the buckets afterwards would mean a seek per
		# row, which on a corpus this size costs far more than the CRC.
		shard_ids, offsets, buckets = [], [], []
		for i, path in enumerate(paths):
			with open(path, "rb") as fh:
				pos = 0
				for line in fh:
					offsets.append(pos)
					shard_ids.append(i)
					# Piece placement, turn, castling, ep — but not the move
					# counters, so the same position at a different move
					# number lands in the same bucket.
					key = line.split(b"\t", 1)[0].rsplit(b" ", 2)[0]
					buckets.append(zlib.crc32(key) % 1000)
					pos += len(line)
		self.shard_id = np.array(shard_ids, dtype=np.int16)
		self.offset = np.array(offsets, dtype=np.int64)
		self.bucket = np.array(buckets, dtype=np.int16)

	def __len__(self):
		return len(self.offset)

	def _handle(self, sid):
		# One handle per (worker, shard): file objects do not survive the fork
		# into DataLoader workers, so they are opened lazily on first use.
		fh = self._fh.get(sid)
		if fh is None:
			fh = self._fh[sid] = open(self.paths[sid], "rb")
		return fh

	def __getitem__(self, i):
		fh = self._handle(int(self.shard_id[i]))
		fh.seek(int(self.offset[i]))
		parts = fh.readline().decode().rstrip("\n").split("\t")
		fen, wdl, sf_best, _cp, _moves, played, result = parts

		board = chess.Board(fen)
		state = encode_board(board)
		legal_moves, indices = get_legal_move_indices(board)

		policy = np.zeros(NUM_MOVES, dtype=np.float32)
		if indices:
			policy[indices] = self.floor / len(indices)
		by_uci = {m.uci(): idx for m, idx in zip(legal_moves, indices)}
		# An unmatched move means the shard and the board disagree, which only
		# happens on a malformed row; dropping the weight is safer than
		# guessing an index, and renormalising below keeps the target valid.
		if (idx := by_uci.get(sf_best)) is not None:
			policy[idx] += self.sf_weight
		if (idx := by_uci.get(played)) is not None:
			policy[idx] += self.human_weight
		total = policy.sum()
		if total > 0:
			policy /= total

		w, d, ll = (int(x) for x in wdl.split(","))
		value = _wdl_value(w, d, ll)
		if self.result_weight > 0.0:
			# Stored result is white-relative; encode_board is mover-relative.
			outcome = float(result)
			if board.turn == chess.BLACK:
				outcome = -outcome
			value = ((1.0 - self.result_weight) * value
			         + self.result_weight * outcome)
		return state, policy, np.float32(value)


def _collate(batch):
	states = torch.from_numpy(np.stack([b[0] for b in batch]))
	policies = torch.from_numpy(np.stack([b[1] for b in batch]))
	values = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32))
	return states, policies, values


@torch.no_grad()
def evaluate(model, loader, device, limit_batches=0):
	"""Policy CE, value MSE and the predict-a-draw baseline on held-out rows.

	The baseline is the same one the self-play loop prints.  It is the number
	that matters: a value head scoring worse than a constant is feeding MCTS
	confident noise, and that is the failure this whole pipeline exists to
	fix.
	"""
	model.eval()
	p_sum = v_sum = base_sum = 0.0
	n = 0
	for i, (s, p, v) in enumerate(loader):
		if limit_batches and i >= limit_batches:
			break
		s, p, v = s.to(device), p.to(device), v.to(device)
		with torch.autocast("cuda", dtype=torch.bfloat16,
		                    enabled=device.type == "cuda"):
			logits, pred = model(s)
		p_sum += float(-(p * F.log_softmax(logits.float(), 1)).sum(1).sum())
		v_sum += float(F.mse_loss(pred.float().squeeze(-1), v, reduction="sum"))
		base_sum += float((v ** 2).sum())
		n += s.size(0)
	model.train()
	if n == 0:
		return None
	return {"policy": p_sum / n, "value": v_sum / n, "baseline": base_sum / n,
	        "entropy": None, "n": n}


@torch.no_grad()
def target_entropy(loader, limit_batches=20):
	"""Entropy of the policy targets — the floor policy CE cannot go below."""
	tot, n = 0.0, 0
	for i, (_s, p, _v) in enumerate(loader):
		if i >= limit_batches:
			break
		q = p.clamp_min(1e-12)
		tot += float(-(p * q.log()).sum(1).sum())
		n += p.size(0)
	return tot / max(n, 1)


def main():
	ap = argparse.ArgumentParser(description=__doc__,
	                             formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--data-dir", required=True)
	ap.add_argument("--out", default=os.path.join(CHECKPOINTS_DIR, "pretrained.pt"))
	# 16x192 (11.54M) rather than run4's 8x128 (3.07M).  Measured on this
	# GPU, that is 3.8x the parameters for a 29% drop in MCTS-batch inference
	# throughput (45.1k -> 31.9k positions/s), because at batch 128 the card
	# is latency-bound rather than compute-bound and the extra width is
	# nearly free.  AlphaZero's 20x256 is 24.8M and costs 52%.
	ap.add_argument("--res-blocks", type=int, default=16)
	ap.add_argument("--filters", type=int, default=192)
	ap.add_argument("--epochs", type=int, default=3)
	ap.add_argument("--batch-size", type=int, default=1024)
	ap.add_argument("--lr", type=float, default=2e-3)
	ap.add_argument("--min-lr", type=float, default=1e-5)
	ap.add_argument("--weight-decay", type=float, default=1e-4)
	ap.add_argument("--value-weight", type=float, default=1.0)
	ap.add_argument("--workers", type=int, default=12)
	ap.add_argument("--val-frac", type=float, default=0.01)
	ap.add_argument("--sf-weight", type=float, default=0.60,
	                help="Target mass on Stockfish's best move.")
	ap.add_argument("--human-weight", type=float, default=0.30,
	                help="Target mass on the move actually played.")
	ap.add_argument("--policy-floor", type=float, default=0.10,
	                help="Mass spread uniformly over legal moves.  Zero here "
	                     "trains unplayed moves to a prior of 0, which MCTS "
	                     "then never explores.")
	ap.add_argument("--result-weight", type=float, default=0.15,
	                help="Blend of game outcome into the Stockfish value.")
	ap.add_argument("--log-every", type=int, default=200)
	ap.add_argument("--resume", default="")
	args = ap.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	paths = sorted(os.path.join(args.data_dir, f)
	               for f in os.listdir(args.data_dir) if f.endswith(".tsv"))
	if not paths:
		raise SystemExit(f"No label shards in {args.data_dir}")

	print(f"Device      : {device}")
	print(f"Label shards: {len(paths)}")
	t0 = time.time()
	full = LabelledPositions(paths, args.sf_weight, args.human_weight,
	                         args.policy_floor, args.result_weight)
	print(f"Positions   : {len(full):,}  (indexed in {time.time()-t0:.1f}s)")

	# Split on a hash of the position, not at random.  Shards come from
	# different Lichess months and are deduped only within a month, so the
	# same FEN can appear in two shards; a random split would put one copy in
	# train and the other in validation and quietly inflate every held-out
	# number reported below.  Hashing sends every copy to the same side.
	buckets = full.bucket
	cut = max(1, int(round(args.val_frac * 1000)))
	is_val = buckets < cut
	val_idx = np.flatnonzero(is_val)
	train_idx = np.flatnonzero(~is_val)
	mk = lambda idx: LabelledPositions(
		paths, args.sf_weight, args.human_weight, args.policy_floor,
		args.result_weight,
		index=(full.shard_id[idx], full.offset[idx], full.bucket[idx]))
	train_ds, val_ds = mk(train_idx), mk(val_idx)
	print(f"Train/val   : {len(train_ds):,} / {len(val_ds):,}")

	dl = lambda ds, sh: DataLoader(
		ds, batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
		collate_fn=_collate, pin_memory=True, drop_last=sh,
		persistent_workers=args.workers > 0,
		prefetch_factor=4 if args.workers > 0 else None)
	train_dl, val_dl = dl(train_ds, True), dl(val_ds, False)

	model = ChessNet(num_res_blocks=args.res_blocks, num_filters=args.filters)
	if args.resume:
		ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
		model = ChessNet(num_res_blocks=ckpt.get("num_res_blocks", args.res_blocks),
		                 num_filters=ckpt.get("num_filters", args.filters))
		model.load_state_dict(ckpt["model_state_dict"])
		print(f"Resumed weights from {args.resume}")
	model.to(device)
	if device.type == "cuda":
		model = model.to(memory_format=torch.channels_last)
	n_params = sum(p.numel() for p in model.parameters())
	print(f"Model       : {args.res_blocks}x{args.filters}, {n_params/1e6:.2f}M params")

	opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
	                        weight_decay=args.weight_decay)
	steps_total = args.epochs * (len(train_ds) // args.batch_size)
	sched = torch.optim.lr_scheduler.OneCycleLR(
		opt, max_lr=args.lr, total_steps=max(steps_total, 1),
		pct_start=0.05, final_div_factor=args.lr / args.min_lr)
	print(f"Schedule    : {steps_total:,} steps, OneCycle to {args.lr}")

	ent = target_entropy(val_dl)
	print(f"Target entropy: {ent:.4f} nats  (policy CE floor)\n", flush=True)

	step = 0
	for epoch in range(args.epochs):
		t_ep = time.time()
		run_p = run_v = 0.0
		run_n = ep_seen = 0
		for states, policies, values in train_dl:
			states = states.to(device, non_blocking=True)
			if device.type == "cuda":
				states = states.contiguous(memory_format=torch.channels_last)
			policies = policies.to(device, non_blocking=True)
			values = values.to(device, non_blocking=True)

			with torch.autocast("cuda", dtype=torch.bfloat16,
			                    enabled=device.type == "cuda"):
				logits, pred = model(states)
			p_loss = -(policies * F.log_softmax(logits.float(), 1)).sum(1).mean()
			v_loss = F.mse_loss(pred.float().squeeze(-1), values)
			loss = p_loss + args.value_weight * v_loss

			opt.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()
			if step < steps_total:
				sched.step()

			run_p += p_loss.item(); run_v += v_loss.item(); run_n += 1
			step += 1
			ep_seen += states.size(0)
			if run_n % args.log_every == 0:
				print(f"  ep{epoch+1} step {step:,}/{steps_total:,}  "
				      f"policy {run_p/run_n:.4f} (headroom "
				      f"{run_p/run_n - ent:+.4f})  value {run_v/run_n:.4f}  "
				      f"lr {sched.get_last_lr()[0]:.2e}  "
				      f"{ep_seen/(time.time()-t_ep+1e-9):.0f} pos/s", flush=True)
				run_p = run_v = 0.0; run_n = 0

		val = evaluate(model, val_dl, device)
		flag = ("  <-- WORSE THAN GUESSING"
		        if val["value"] > val["baseline"] else "")
		print(f"[epoch {epoch+1}] val policy CE {val['policy']:.4f}  "
		      f"value MSE {val['value']:.4f}{flag}  "
		      f"(baseline {val['baseline']:.4f}, n={val['n']:,})  "
		      f"in {(time.time()-t_ep)/60:.1f}m", flush=True)

		# No optimizer/scheduler state: this run uses AdamW+OneCycle while the
		# self-play loop uses SGD+MultiStepLR, so there is nothing meaningful
		# to hand over.  train.py catches the resulting KeyError and starts
		# with a fresh optimizer -- passing an explicit None would raise a
		# TypeError it does not catch.
		payload = {
			"model_state_dict": model.state_dict(),
			"scheduler_state_dict": None,
			"iteration": 0,
			"num_res_blocks": args.res_blocks,
			"num_filters": args.filters,
			"generation": 0,
			"pretrain_epochs": epoch + 1,
		}
		tmp = args.out + ".tmp"
		torch.save(payload, tmp)
		os.replace(tmp, args.out)
		print(f"           saved {args.out}", flush=True)

	print("Pre-training done.")


if __name__ == "__main__":
	main()
