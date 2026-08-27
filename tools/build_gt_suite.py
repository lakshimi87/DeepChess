"""Build a large, Stockfish-verified ground-truth suite.

    python tools/build_gt_suite.py --in-dir data/gt_src/positions \
                                   --out resources/gt_suite.json --depth 20

The hand-written suite in validate_gt.py has 27 positions, which puts one
standard deviation at roughly +/-9 percentage points.  Run4 scored between
41% and 63% on it across 258 iterations and none of that spread was
distinguishable from noise -- the only absolute yardstick in the project
could not resolve the effect sizes it was being asked about.  Several hundred
positions bring that under +/-2 points.

Two rules keep the generated tests honest:

**Held-out source.**  Build from a month that is *not* in the training
corpus.  A suite drawn from the same games the net trains on stops measuring
generalisation and starts measuring recall.

**A margin, not just a best move.**  A position where three moves are within
10cp of each other has no single right answer, and scoring it as though it
did adds variance without adding information.  Every generated test requires
the best move to beat the second best by a category-specific margin, so
"wrong" means the net missed something real.

Mates are found by enumeration rather than by search: pushing each legal move
and asking python-chess whether it is checkmate is exact, free, and finds
*every* mating move, so positions with more than one mate stay usable
instead of being scored against an arbitrary choice among them.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from multiprocessing import Pool

import chess
import chess.engine

_engine = None
_depth = 20


def _init(sf_path, depth, hash_mb):
	global _engine, _depth
	_depth = depth
	_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
	_engine.configure({"Threads": 1, "Hash": hash_mb, "UCI_ShowWDL": True})


def _mates_in_1(board):
	"""Every legal move that delivers immediate checkmate."""
	out = []
	for m in board.legal_moves:
		board.push(m)
		if board.is_checkmate():
			out.append(m.uci())
		board.pop()
	return out


def _piece_count(board):
	return sum(1 for sq in chess.SQUARES
	           if (p := board.piece_at(sq)) and p.piece_type != chess.KING)


def _cp(score, mate_cp=10000):
	"""Score as a comparable integer, mates far outside the cp range."""
	if score.is_mate():
		m = score.mate()
		return mate_cp - abs(m) if m > 0 else -mate_cp + abs(m)
	return score.score()


def _analyse(fen):
	"""Classify one position into a move test, an eval test, or neither."""
	try:
		board = chess.Board(fen)
	except ValueError:
		return None
	if board.is_game_over():
		return None
	n_legal = board.legal_moves.count()
	if n_legal < 2:
		return None  # forced move: nothing to get wrong

	# Mate in 1 needs no search, and enumeration catches every mating move.
	mates = _mates_in_1(board)
	if mates:
		return {"kind": "move", "category": "Mate in 1", "fen": fen,
		        "moves": mates, "margin": None,
		        "desc": f"Mate in 1 ({len(mates)} mating move"
		                f"{'s' if len(mates) > 1 else ''}, {n_legal} legal)"}

	try:
		info = _engine.analyse(board, chess.engine.Limit(depth=_depth),
		                       multipv=2)
	except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
		return None
	if len(info) < 2 or not info[0].get("pv") or not info[1].get("pv"):
		return None

	best, second = info[0], info[1]
	best_move = best["pv"][0]
	rel = best["score"].relative
	gap = _cp(rel) - _cp(second["score"].relative)
	if gap < 0:
		return None  # multipv ordering violated; skip rather than trust it
	wdl = rel.wdl()
	pieces = _piece_count(board)

	is_capture = board.is_capture(best_move) or best_move.promotion is not None

	# Categories are ordered most-specific first: a mate-in-2 that also wins
	# material should be filed as the mate.
	if rel.is_mate() and 0 < rel.mate() <= 3 and gap >= 500:
		cat, need = f"Mate in {rel.mate()}", 500
	elif pieces <= 10 and gap >= 150:
		cat, need = "Endgame", 150
	elif is_capture and gap >= 200:
		cat, need = "Win Material", 200
	elif not is_capture and gap >= 300:
		cat, need = "Tactics", 300
	elif board.fullmove_number <= 12 and gap >= 80:
		cat, need = "Opening", 80
	elif gap >= 120:
		cat, need = "Middlegame", 120
	else:
		cat = None

	out = None
	if cat:
		out = {"kind": "move", "category": cat, "fen": fen,
		       "moves": [best_move.uci()], "margin": gap,
		       "desc": f"{cat}: best beats 2nd by {gap}cp "
		               f"({n_legal} legal, {pieces} pieces)"}

	# Eval tests come from the same search, so they cost nothing extra.  Only
	# unambiguous verdicts qualify: a value head cannot be marked wrong on a
	# position Stockfish itself calls unclear.
	exp = wdl.expectation()
	turn = "white" if board.turn == chess.WHITE else "black"
	other = "black" if turn == "white" else "white"
	ev = None
	if exp >= 0.95:
		ev = turn
	elif exp <= 0.05:
		ev = other
	elif wdl.draws >= 900 and abs(exp - 0.5) <= 0.02:
		ev = "draw"
	if ev:
		out = out or {}
		out.setdefault("kind", "eval-only")
		out["eval"] = {"fen": fen, "expected": ev,
		               "desc": f"{ev} (WDL {wdl.wins}/{wdl.draws}/{wdl.losses}, "
		                       f"{pieces} pieces)"}
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__,
	                             formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--in-dir", required=True,
	                help="Position shards from a month NOT in the training set.")
	ap.add_argument("--out", default="resources/gt_suite.json")
	ap.add_argument("--stockfish",
	                default="third_party/stockfish/stockfish-ubuntu-x86-64-bmi2")
	ap.add_argument("--depth", type=int, default=20)
	ap.add_argument("--workers", type=int, default=6)
	ap.add_argument("--hash-mb", type=int, default=256)
	ap.add_argument("--candidates", type=int, default=60000,
	                help="Positions to sift.  Most fail the margin test.")
	ap.add_argument("--per-category", type=int, default=90)
	ap.add_argument("--eval-per-class", type=int, default=90)
	ap.add_argument("--seed", type=int, default=20260827)
	args = ap.parse_args()

	shards = sorted(os.path.join(args.in_dir, f)
	                for f in os.listdir(args.in_dir) if f.endswith(".tsv"))
	if not shards:
		sys.exit(f"No position shards in {args.in_dir}")

	fens = []
	for path in shards:
		with open(path) as fh:
			for line in fh:
				fens.append(line.split("\t", 1)[0])
		if len(fens) >= args.candidates * 4:
			break
	random.Random(args.seed).shuffle(fens)
	fens = fens[:args.candidates]
	print(f"Sifting {len(fens):,} candidates at depth {args.depth} "
	      f"({args.workers} workers)", flush=True)

	moves, evals = {}, {"white": [], "black": [], "draw": []}
	seen = 0
	with Pool(args.workers, initializer=_init,
	          initargs=(args.stockfish, args.depth, args.hash_mb)) as pool:
		for res in pool.imap_unordered(_analyse, fens, chunksize=8):
			seen += 1
			if seen % 2000 == 0:
				got = {k: len(v) for k, v in moves.items()}
				print(f"  {seen:,}/{len(fens):,}  move={got}  "
				      f"eval={ {k: len(v) for k, v in evals.items()} }",
				      flush=True)
			if not res:
				continue
			if res.get("kind") == "move" and "category" in res:
				bucket = moves.setdefault(res["category"], [])
				if len(bucket) < args.per_category:
					bucket.append(res)
			ev = res.get("eval")
			if ev and len(evals[ev["expected"]]) < args.eval_per_class:
				evals[ev["expected"]].append(ev)
			# Stop early once every bucket is full.
			if (moves and all(len(v) >= args.per_category for v in moves.values())
			        and len(moves) >= 6
			        and all(len(v) >= args.eval_per_class for v in evals.values())):
				break

	move_tests = [m for cat in sorted(moves) for m in moves[cat]]
	eval_tests = [e for k in ("white", "black", "draw") for e in evals[k]]
	suite = {
		"meta": {
			"source": os.path.abspath(args.in_dir),
			"stockfish_depth": args.depth,
			"candidates_sifted": seen,
			"note": "Generated by tools/build_gt_suite.py from a month held "
			        "out of the training corpus.  Every move test requires "
			        "the best move to beat the second best by the margin "
			        "recorded in its entry.",
		},
		"move_tests": [{k: t[k] for k in ("category", "fen", "moves", "desc")}
		               for t in move_tests],
		"eval_tests": eval_tests,
	}
	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	tmp = args.out + ".tmp"
	with open(tmp, "w") as fh:
		json.dump(suite, fh, indent=1)
	os.replace(tmp, args.out)

	print(f"\nWrote {args.out}")
	print(f"  move tests: {len(move_tests):,}  "
	      f"{dict(Counter(t['category'] for t in move_tests))}")
	print(f"  eval tests: {len(eval_tests):,}  "
	      f"{dict(Counter(t['expected'] for t in eval_tests))}")
	n = len(move_tests) + len(eval_tests)
	if n:
		print(f"  1 sigma on a 50% scorer: +/-{50 / n ** 0.5:.1f} points "
		      f"(was +/-9.6 on the 27-position suite)")


if __name__ == "__main__":
	main()
