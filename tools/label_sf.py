"""Label extracted positions with Stockfish evaluations.

    python tools/label_sf.py --in-dir data/bootstrap/positions \
                             --out-dir data/bootstrap/labels --depth 14

This is the piece that breaks the self-play loop's fixed point.  The loop's
value head sits at the predict-a-draw baseline because 57% of its games are
drawn and every position in a drawn game gets the label 0.0 — the constant
predictor is genuinely optimal for that data.  A Stockfish WDL evaluation is
per-*position*, so a drawish middlegame and a won endgame get different
labels even when both games ended in a draw, and the constant predictor stops
being optimal.

**Why WDL and not centipawns.**  Stockfish's ``UCI_ShowWDL`` reports win/draw/
loss permille at the current search depth, so ``2*expectation - 1`` lands in
[-1, 1] already calibrated against the value head's own target semantics.
The usual ``tanh(cp/400)`` needs a hand-tuned scale that silently decides how
much of the eval range collapses to +/-1, and gets it wrong at both ends.

**Why multipv=1.**  Measured on this machine, depth 14 costs 52 ms/pos at
multipv=1 and 427 ms/pos at multipv=8 — multipv defeats alpha-beta pruning,
so a soft policy target costs 8x the search budget.  The PGN already carries
a policy target for free (the move a 1800+ player actually chose), and
Stockfish's own best move comes back from multipv=1 anyway.  Spending the 8x
on volume instead buys far more, given that the replay window is the loop's
largest deficit.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import chess
import chess.engine

_engine = None
_depth = 14
_multipv = 1


def _init(sf_path, depth, multipv, hash_mb):
	global _engine, _depth, _multipv
	_depth, _multipv = depth, multipv
	_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
	_engine.configure({"Threads": 1, "Hash": hash_mb, "UCI_ShowWDL": True})


def _fmt_score(score):
	"""Centipawns as an int, or ``M<n>`` for a forced mate in n."""
	if score.is_mate():
		return f"M{score.mate()}"
	return str(score.score())


def _label(line):
	"""Return one output row, or None if the position is unusable."""
	parts = line.rstrip("\n").split("\t")
	if len(parts) != 3:
		return None
	fen, played, result = parts
	try:
		board = chess.Board(fen)
	except ValueError:
		return None
	if board.is_game_over() or not any(board.legal_moves):
		return None
	try:
		info = _engine.analyse(board, chess.engine.Limit(depth=_depth),
		                       multipv=_multipv)
	except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
		return None
	if isinstance(info, dict):
		info = [info]
	top = info[0]
	# .relative is from the side to move's point of view, which is the same
	# perspective encode_board() renders the position in.  Using .white()
	# here would sign-flip every black-to-move label.
	rel = top["score"].relative
	wdl = rel.wdl()
	pv = top.get("pv")
	if not pv:
		return None
	best = pv[0].uci()
	moves = ",".join(f"{i['pv'][0].uci()}:{_fmt_score(i['score'].relative)}"
	                 for i in info if i.get("pv"))
	return (f"{fen}\t{wdl.wins},{wdl.draws},{wdl.losses}\t{best}\t"
	        f"{_fmt_score(rel)}\t{moves}\t{played}\t{result}\n")


def main():
	ap = argparse.ArgumentParser(description=__doc__,
	                             formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--in-dir", required=True)
	ap.add_argument("--out-dir", required=True)
	ap.add_argument("--stockfish", default="third_party/stockfish/stockfish-ubuntu-x86-64-bmi2")
	ap.add_argument("--depth", type=int, default=14)
	ap.add_argument("--multipv", type=int, default=1,
	                help="Raising this is the single most expensive knob: "
	                     "multipv=8 costs ~8x multipv=1 at the same depth, for "
	                     "a policy signal the PGN already supplies for free.")
	ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
	ap.add_argument("--hash-mb", type=int, default=128)
	ap.add_argument("--max-shards", type=int, default=0)
	ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
	                help="After draining the input directory, re-scan it every "
	                     "SECONDS instead of exiting.  Extraction turns one "
	                     "month into ~8M positions in minutes while labelling "
	                     "them takes ~14h, so the two run as a pipeline rather "
	                     "than in sequence.  0 exits when the queue is empty.")
	args = ap.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	if not os.path.exists(args.stockfish):
		sys.exit(f"Stockfish not found at {args.stockfish}")

	def pending():
		"""Input shards that have no corresponding output shard yet.

		Re-read on every pass rather than enumerated once, so a labelling run
		picks up shards that extraction writes while it is already going, and
		so an interrupted run resumes by simply skipping what finished.
		"""
		names = sorted(f for f in os.listdir(args.in_dir) if f.endswith(".tsv"))
		out = [n for n in names
		       if not os.path.exists(os.path.join(args.out_dir,
		                                          n.replace("pos_", "lab_")))]
		return out[:args.max_shards] if args.max_shards else out

	print(f"{len(pending())} shards to label (depth {args.depth}, multipv "
	      f"{args.multipv}, {args.workers} workers"
	      f"{f', watching every {args.watch}s' if args.watch else ''})",
	      flush=True)

	grand_total, grand_start = 0, time.time()
	with Pool(args.workers, initializer=_init,
	          initargs=(args.stockfish, args.depth, args.multipv,
	                    args.hash_mb)) as pool:
		while True:
			todo = pending()
			if not todo:
				if not args.watch:
					break
				print(f"  idle — nothing unlabelled, re-scanning in "
				      f"{args.watch}s", flush=True)
				time.sleep(args.watch)
				continue

			for name in todo:
				src = os.path.join(args.in_dir, name)
				dst = os.path.join(args.out_dir, name.replace("pos_", "lab_"))
				with open(src) as fh:
					lines = fh.readlines()
				t0, done, kept = time.time(), 0, 0
				# Write to .tmp and rename: a shard is either fully labelled or
				# absent, so an interrupted run resumes on shard boundaries and
				# never reads a half-written file back as complete.
				with open(dst + ".tmp", "w") as out:
					for row in pool.imap(_label, lines, chunksize=16):
						done += 1
						if row:
							out.write(row)
							kept += 1
						if done % 50000 == 0:
							rate = done / (time.time() - t0)
							eta = (len(lines) - done) / rate / 60
							print(f"  {name}: {done:,}/{len(lines):,}  "
							      f"{rate:.0f} pos/s  ETA {eta:.0f}m", flush=True)
				os.replace(dst + ".tmp", dst)
				grand_total += kept
				el = time.time() - t0
				print(f"[done] {name} -> {kept:,}/{len(lines):,} rows in "
				      f"{el/60:.1f}m ({done/el:.0f} pos/s) | total "
				      f"{grand_total:,} | "
				      f"{(time.time()-grand_start)/3600:.2f}h elapsed", flush=True)

	print(f"Labelled {grand_total:,} positions in "
	      f"{(time.time()-grand_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
	main()
