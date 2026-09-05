"""Generate self-play *positions* for Stockfish to label.

    python -m src.gen_positions --checkpoint checkpoints/pretrained_70M.pt \
                                --target-positions 1500000 \
                                --out-dir data/expert/positions

Self-play produces two things and they have opposite value here.

Its **targets** -- MCTS visit counts, and game outcomes -- are what every run
from run5 to run8 trained on, and every one of them lost ground on the
ground-truth suite at a rate proportional to how much self-play was in the mix.

Its **distribution** is the one thing supervised data cannot supply.  The
pre-training corpus is 87M positions from Lichess games between 1800+ humans.
Doubling it from 8M to 70M bought 21 points, about 1 sigma per doubling: the
marginal human position is redundant because it is not a position this net gets
wrong.  Positions from the net's *own* play are exactly where its errors live,
and none of them have ever been labelled.

So this keeps the distribution and throws the targets away.  Games are played
by the net; the labels come from Stockfish afterwards via ``tools/label_sf.py``,
which is strictly stronger than a 400-simulation tree over this net and cannot
be contaminated by it.  Output is the same three-column format
``tools/label_sf.py`` already reads, so the rest of the pipeline is unchanged.

**Resignation and adjudication are off, deliberately.**  In the training loop
they exist to stop a decided game from being mislabelled a draw -- a targets
problem, and there are no targets here.  What they also do is end games before
the endgame, and the endgame is this net's largest deficit (57-72 of 91 on the
suite, and the category run8 damaged most, -15 points).  Played out, a game
reaches the positions the corpus is missing.

**The played-move column is the net's own move, so train with
``--human-weight 0``.**  On the Lichess corpus that column is a 1800+ player's
move and 0.30 of the policy target rides on it.  Here it is the net's own
choice: leaving the weight on would put 30% of the target on the net imitating
itself, which is the self-play pathology this whole detour exists to escape.
The column is written anyway to keep the file format identical.
"""
import argparse
import os
import sys
import time
from collections import Counter

import chess
import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import perf
from src.mcts import MCTS
from src.model import ChessNet
from src.selfplay import fp16_state_dict


def _worker(rank, task_q, result_q, cfg):
	"""Play games on demand, replying with the raw positions of each.

	Mirrors ``selfplay._worker``'s setup -- one inference copy per process, one
	torch thread, a distinct RNG stream per rank -- but nothing else: no
	reference net, no arena, no examples.  It returns FENs.
	"""
	try:
		perf.configure(num_threads=1)
		seed = (cfg["seed"] + rank * 7919) % (2 ** 31)
		np.random.seed(seed)
		import random as _random
		_random.seed(seed)
		torch.manual_seed(seed)

		device = torch.device(cfg["device"])
		model = ChessNet(num_res_blocks=cfg["res_blocks"],
		                 num_filters=cfg["filters"])
		model.load_state_dict(
			torch.load(cfg["weights"], map_location="cpu", weights_only=True))
		model = perf.to_inference(model, device, half=cfg["half"])
		mcts = MCTS(model, device, num_simulations=cfg["simulations"],
		            batch_size=cfg["mcts_batch"],
		            fpu_reduction=cfg["fpu_reduction"],
		            dirichlet_alpha=cfg["dirichlet_alpha"],
		            dirichlet_eps=cfg["dirichlet_eps"])
		result_q.put(("ready", rank, None, None))
	except Exception:
		import traceback
		result_q.put(("error", rank, traceback.format_exc(), None))
		return

	while True:
		try:
			kind, payload = task_q.get()
		except (EOFError, OSError):
			return
		if kind == "stop":
			return
		t0 = time.perf_counter()
		try:
			rows, ending = play_positions(
				mcts,
				max_moves=cfg["max_moves"],
				temp_moves=cfg["temp_moves"],
			)
			result_q.put(("game", rows, ending, time.perf_counter() - t0))
		except Exception:
			import traceback
			result_q.put(("error", payload, traceback.format_exc(), None))
			return


def play_positions(mcts, max_moves=512, temp_moves=30):
	"""One game; returns ``([(fen, uci), ...], ending)``.

	The game runs to a real finish -- mate, stalemate, the 50-move rule,
	threefold repetition -- or to *max_moves*, with no resignation and no
	adjudication.  Root Dirichlet noise stays on for the whole game and the
	first *temp_moves* plies are sampled at temperature 1.0, both for the same
	reason: without them thousands of games from the fixed start position
	collapse onto a handful of distinct lines, and the corpus would be one game
	repeated.
	"""
	board = chess.Board()
	rows = []
	while not board.is_game_over() and len(rows) < max_moves:
		temperature = 1.0 if len(rows) < temp_moves else 0.01
		move, _policy = mcts.search(board, temperature=temperature,
		                            add_noise=True)
		if move is None:
			break
		rows.append((board.fen(), move.uci()))
		board.push(move)

	if board.is_game_over():
		outcome = board.result()          # "1-0", "0-1", "1/2-1/2"
		ending = board.outcome().termination.name.lower()
	else:
		# Unfinished, not drawn.  The result column is written as 0 because the
		# format has nowhere to say "unknown", which is the second reason to
		# pre-train this corpus with --result-weight 0: on these rows the
		# column is not a draw, it is a missing value.
		outcome = "1/2-1/2"
		ending = "move_limit"
	score = {"1-0": "1", "0-1": "-1"}.get(outcome, "0")
	return [(fen, uci, score) for fen, uci in rows], ending


class ShardWriter:
	"""Append rows, rolling to a new file every *shard_size* lines.

	Shards are written to a ``.tmp`` name and renamed when full, because
	``tools/label_sf.py`` picks up input shards while it is already running: a
	half-written file that appears in the directory would be labelled short and
	then skipped forever, since the labeller decides what is pending by which
	output names exist.
	"""

	def __init__(self, out_dir, shard_size, first_index=0):
		self.out_dir = out_dir
		self.shard_size = shard_size
		self.index = first_index
		self.count = 0
		self.total = 0
		self._fh = None
		os.makedirs(out_dir, exist_ok=True)

	def _path(self):
		return os.path.join(self.out_dir, f"pos_{self.index:05d}.tsv")

	def write(self, rows):
		for fen, uci, score in rows:
			if self._fh is None:
				self._fh = open(self._path() + ".tmp", "w")
				self.count = 0
			self._fh.write(f"{fen}\t{uci}\t{score}\n")
			self.count += 1
			self.total += 1
			if self.count >= self.shard_size:
				self.close()

	def close(self):
		if self._fh is None:
			return
		self._fh.close()
		os.replace(self._path() + ".tmp", self._path())
		print(f"  shard complete: {self._path()} ({self.count} positions)",
		      flush=True)
		self._fh = None
		self.index += 1


def main():
	ap = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--checkpoint", default="checkpoints/pretrained_70M.pt")
	ap.add_argument("--out-dir", default="data/expert/positions")
	ap.add_argument("--target-positions", type=int, default=1_500_000)
	ap.add_argument("--shard-size", type=int, default=500_000,
	                help="Smaller than the 1M Lichess shards on purpose: the "
	                     "labeller only starts a shard once it is renamed into "
	                     "place, so a smaller shard means labelling begins "
	                     "sooner and the two stages overlap further")
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--simulations", type=int, default=400,
	                help="Matches the training loop.  These positions are "
	                     "meant to be the ones the net actually reaches, so "
	                     "the search that reaches them should be the one the "
	                     "loop uses rather than a cheaper stand-in")
	ap.add_argument("--mcts-batch", type=int, default=16)
	ap.add_argument("--max-moves", type=int, default=512)
	ap.add_argument("--temp-moves", type=int, default=30)
	ap.add_argument("--fpu-reduction", type=float, default=0.25)
	ap.add_argument("--dirichlet-alpha", type=float, default=0.3)
	ap.add_argument("--dirichlet-eps", type=float, default=0.25)
	ap.add_argument("--half", type=int, default=1)
	ap.add_argument("--seed", type=int, default=1234)
	ap.add_argument("--device", default="cuda")
	ap.add_argument("--dedup", type=int, default=1,
	                help="Drop repeat FENs.  Thousands of games from one start "
	                     "position share their opening plies, and a duplicate "
	                     "costs a full Stockfish search to learn nothing")
	args = ap.parse_args()

	device = args.device if torch.cuda.is_available() else "cpu"
	ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
	model = ChessNet(num_res_blocks=ckpt.get("num_res_blocks", 16),
	                 num_filters=ckpt.get("num_filters", 192))
	model.load_state_dict(ckpt["model_state_dict"])

	os.makedirs(args.out_dir, exist_ok=True)
	weights_path = os.path.join(args.out_dir, ".gen_weights.pt")
	tmp = weights_path + ".tmp"
	torch.save(fp16_state_dict(model), tmp)
	os.replace(tmp, weights_path)

	cfg = {
		"device": device,
		"res_blocks": ckpt.get("num_res_blocks", 16),
		"filters": ckpt.get("num_filters", 192),
		"weights": weights_path,
		"half": bool(args.half),
		"simulations": args.simulations,
		"mcts_batch": args.mcts_batch,
		"max_moves": args.max_moves,
		"temp_moves": args.temp_moves,
		"fpu_reduction": args.fpu_reduction,
		"dirichlet_alpha": args.dirichlet_alpha,
		"dirichlet_eps": args.dirichlet_eps,
		"seed": args.seed,
	}

	existing = sorted(f for f in os.listdir(args.out_dir)
	                  if f.startswith("pos_") and f.endswith(".tsv"))
	first_index = (int(existing[-1][4:9]) + 1) if existing else 0

	print(f"Checkpoint  : {args.checkpoint} "
	      f"({cfg['res_blocks']}x{cfg['filters']})")
	print(f"Device      : {device}   workers {args.workers}   "
	      f"{args.simulations} sims/move")
	print(f"Output      : {args.out_dir}  "
	      f"(shards of {args.shard_size}, starting at pos_{first_index:05d})")
	print(f"Target      : {args.target_positions} positions")
	print(f"Endings     : resignation off, adjudication off — games play out")
	if existing:
		print(f"Resuming    : {len(existing)} shard(s) already present")
	print(flush=True)

	ctx = mp.get_context("spawn")
	task_q, result_q = ctx.Queue(), ctx.Queue()
	procs = []
	for rank in range(args.workers):
		p = ctx.Process(target=_worker, args=(rank, task_q, result_q, cfg),
		                daemon=True)
		p.start()
		procs.append(p)

	ready = 0
	while ready < args.workers:
		msg = result_q.get()
		if msg[0] != "ready":
			raise RuntimeError(f"worker failed to start:\n{msg[2]}")
		ready += 1
	print(f"{args.workers} workers ready\n", flush=True)

	writer = ShardWriter(args.out_dir, args.shard_size, first_index)
	seen = set() if args.dedup else None
	endings = Counter()
	games = dups = 0
	t_start = time.time()
	in_flight = 0
	next_id = 0

	def dispatch(n):
		nonlocal in_flight, next_id
		for _ in range(n):
			task_q.put(("play", next_id))
			next_id += 1
			in_flight += 1

	dispatch(args.workers * 2)
	try:
		while writer.total < args.target_positions:
			msg = result_q.get()
			in_flight -= 1
			if msg[0] == "error":
				raise RuntimeError(f"worker error:\n{msg[2]}")
			_kind, rows, ending, secs = msg
			games += 1
			endings[ending] += 1
			if seen is not None:
				fresh = []
				for row in rows:
					if row[0] in seen:
						dups += 1
						continue
					seen.add(row[0])
					fresh.append(row)
				rows = fresh
			writer.write(rows)
			dispatch(1)

			if games % 25 == 0:
				elapsed = time.time() - t_start
				rate = writer.total / max(elapsed, 1e-9)
				remain = (args.target_positions - writer.total) / max(rate, 1e-9)
				print(f"  {games:>6} games  {writer.total:>9} positions  "
				      f"{rate:6.1f} pos/s  "
				      f"{writer.total / games:5.1f} kept/game  "
				      f"dup {dups / max(dups + writer.total, 1) * 100:4.1f}%  "
				      f"ETA {remain / 3600:5.2f}h", flush=True)
	finally:
		writer.close()
		for _ in procs:
			try:
				task_q.put(("stop", None))
			except Exception:
				pass
		for p in procs:
			p.join(timeout=10)
			if p.is_alive():
				p.terminate()

	elapsed = time.time() - t_start
	print(f"\nDone: {writer.total} positions from {games} games "
	      f"in {elapsed / 3600:.2f}h  ({writer.total / elapsed:.1f} pos/s)")
	print(f"Duplicates dropped: {dups} "
	      f"({dups / max(dups + writer.total, 1) * 100:.1f}%)")
	print("Game endings: " + "  ".join(
		f"{k} {v} ({v / max(games, 1) * 100:.0f}%)"
		for k, v in endings.most_common()))


if __name__ == "__main__":
	main()
