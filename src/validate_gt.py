#!/usr/bin/env python3
"""Ground truth validation for DeepChess.

Tests the neural engine (and classical baseline) against curated positions
with known best moves and known evaluations.  Use after training to measure
how well the model has learned.

    ./validate_gt.sh                    # test latest checkpoint
    ./validate_gt.sh --history          # show progress across all checkpoints
    ./validate_gt.sh --simulations 400  # more MCTS sims (slower but fairer)
"""

import argparse
import glob
import math
import os
import sys
import time

import chess
import torch

from .board_utils import encode_board
from .model import ChessNet
from .mcts import MCTS
from .engine import _ClassicalEngine, get_device
from .paths import CHECKPOINTS_DIR


# ═══════════════════════════════════════════════════════════════════════
# Ground-truth test positions
# ═══════════════════════════════════════════════════════════════════════

# Each move test: (category, FEN, [acceptable_uci_moves], description)
MOVE_TESTS = [
	# ── Mate in 1 ─────────────────────────────────────────────────
	("Mate in 1",
	 "1k6/8/1K6/8/8/8/8/7R w - - 0 1",
	 ["h1h8"],
	 "Rook back rank mate"),

	("Mate in 1",
	 "3k4/8/3K4/8/8/8/8/R7 w - - 0 1",
	 ["a1a8"],
	 "Rook back rank mate"),

	("Mate in 1",
	 "5k2/8/5K2/8/8/8/8/7R w - - 0 1",
	 ["h1h8"],
	 "Rook back rank mate"),

	("Mate in 1",
	 "6k1/5ppp/6N1/8/8/8/8/4R1K1 w - - 0 1",
	 ["e1e8"],
	 "Back rank mate, knight guards"),

	("Mate in 1",
	 "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
	 ["h5f7"],
	 "Scholar's mate Qxf7#"),

	# ── Win material (capture undefended piece) ───────────────────
	("Win Material",
	 "rnb1kbnr/pppppppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
	 ["e4d5"],
	 "Capture hanging queen"),

	("Win Material",
	 "rnbqkbnr/pppppppp/8/3r4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
	 ["e4d5"],
	 "Capture hanging rook"),

	("Win Material",
	 "rnbqk1nr/pppppppp/8/5b2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
	 ["e4f5"],
	 "Capture hanging bishop"),

	("Win Material",
	 "rnbqkb1r/pppppppp/8/4n3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
	 ["d4e5"],
	 "Capture hanging knight"),

	("Win Material",
	 "rnbqkbnr/ppp1pppp/8/3p4/2B5/8/PPPPPPPP/RNBQK1NR b KQkq - 0 1",
	 ["d5c4"],
	 "Capture hanging bishop (black)"),

	# ── Opening quality (any reasonable book move passes) ─────────
	("Opening",
	 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
	 ["e2e4", "d2d4", "c2c4", "g1f3", "b1c3", "e2e3", "d2d3",
	  "g2g3", "b2b3", "a2a3", "b2b4"],
	 "Reasonable first move"),

	("Opening",
	 "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
	 ["e7e5", "c7c5", "e7e6", "c7c6", "d7d5", "g8f6", "d7d6",
	  "g7g6", "b7b6", "a7a6", "b8c6"],
	 "Response to 1.e4"),

	("Opening",
	 "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
	 ["d7d5", "g8f6", "e7e6", "c7c5", "f7f5", "g7g6", "c7c6",
	  "d7d6", "b8c6"],
	 "Response to 1.d4"),

	("Opening",
	 "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
	 ["g1f3", "f1c4", "d2d4", "b1c3", "f2f4", "f1b5", "f1e2",
	  "d2d3"],
	 "2nd move in 1.e4 e5"),

	("Opening",
	 "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
	 ["c2c4", "g1f3", "c1f4", "e2e3", "b1c3", "c1g5", "b1d2"],
	 "2nd move in 1.d4 d5"),
]

# Each eval test: (FEN, expected_winner, description)
# expected_winner = "white" | "black" | "draw"
#
# Positions are mid-game configurations reachable from normal play — an
# AlphaZero-style value head only sees the distribution of positions that
# arise in self-play, so "starting position minus one backrank piece" is
# out-of-distribution and tells us nothing about whether the net has
# learned to count material.  These FENs keep realistic pawn structures,
# developed minor pieces, and intact castling rights.
EVAL_TESTS = [
	("r1b1kb1r/ppp2ppp/2n2n2/3pp3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 6",
	 "white",
	 "White up a queen (mid-game)"),

	("r1bqkb1r/ppp2ppp/2n2n2/3pp3/4P3/2N2N2/PPPP1PPP/R1BQKB2 w Qkq - 0 6",
	 "black",
	 "Black up a rook (mid-game)"),

	("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4",
	 "draw",
	 "Four Knights Game (equal)"),

	("r1bqk2r/ppp2ppp/5n2/3pp3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 6",
	 "white",
	 "White up bishop + knight (mid-game)"),

	("r1bqkb1r/ppp2ppp/2n2n2/3pp3/4P3/2N2N2/PPPP1PPP/R1B1KB1R w KQkq - 0 6",
	 "black",
	 "Black up a queen (mid-game)"),
]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _uci_to_san(fen, uci_list, limit=3):
	"""Convert the first *limit* UCI strings to SAN for display."""
	board = chess.Board(fen)
	out = []
	for u in uci_list[:limit]:
		try:
			out.append(board.san(chess.Move.from_uci(u)))
		except Exception:
			out.append(u)
	return "/".join(out)


def _eval_ok(value, expected, turn, threshold=0.15, draw_tol=0.30):
	"""Does *value* (from current player's POV) match *expected* winner?"""
	current = "white" if turn == chess.WHITE else "black"
	if expected == "draw":
		return abs(value) < draw_tol
	if expected == current:
		return value > threshold
	return value < -threshold


# ═══════════════════════════════════════════════════════════════════════
# Engine runners
# ═══════════════════════════════════════════════════════════════════════

def neural_move(model, device, fen, sims):
	board = chess.Board(fen)
	mcts = MCTS(model, device, num_simulations=sims)
	move, _ = mcts.search(board, temperature=0.01)
	if move is None:
		return None, "-"
	return move.uci(), board.san(move)


def classical_move(fen, depth=3):
	board = chess.Board(fen)
	move = _ClassicalEngine(depth=depth).get_move(board)
	if move is None:
		return None, "-"
	return move.uci(), board.san(move)


def neural_eval(model, device, fen):
	"""Return value from current player's perspective."""
	board = chess.Board(fen)
	state = encode_board(board)
	with torch.no_grad():
		t = torch.from_numpy(state).unsqueeze(0).to(device)
		_, v = model(t)
	return v.item()


def classical_eval(fen):
	"""Return normalised eval from current player's perspective."""
	board = chess.Board(fen)
	raw = _ClassicalEngine.evaluate(board)
	if board.turn == chess.BLACK:
		raw = -raw
	return math.tanh(raw / 500.0)


# ═══════════════════════════════════════════════════════════════════════
# Full test-suite runner
# ═══════════════════════════════════════════════════════════════════════

def _ordered_categories():
	seen, cats = set(), []
	for cat, *_ in MOVE_TESTS:
		if cat not in seen:
			cats.append(cat)
			seen.add(cat)
	return cats


def run_suite(model, device, sims, depth=3):
	"""Run every test.  Returns ``(results_dict, categories)``."""
	cats = _ordered_categories()
	res = dict(nm={}, cm={}, ne=[], ce=[])  # neural/classical × move/eval

	for cat, fen, acceptable, desc in MOVE_TESTS:
		exp_san = _uci_to_san(fen, acceptable)

		# Neural
		if model is not None:
			uci, san = neural_move(model, device, fen, sims)
			ok = uci in acceptable
			res["nm"].setdefault(cat, []).append((ok, san, exp_san, desc))

		# Classical
		uci, san = classical_move(fen, depth)
		ok = (uci in acceptable) if uci else False
		res["cm"].setdefault(cat, []).append((ok, san, exp_san, desc))

	for fen, expected, desc in EVAL_TESTS:
		board = chess.Board(fen)
		if model is not None:
			v = neural_eval(model, device, fen)
			ok = _eval_ok(v, expected, board.turn)
			res["ne"].append((ok, v, expected, desc))
		v = classical_eval(fen)
		ok = _eval_ok(v, expected, board.turn)
		res["ce"].append((ok, v, expected, desc))

	return res, cats


# ═══════════════════════════════════════════════════════════════════════
# Pretty printing
# ═══════════════════════════════════════════════════════════════════════

P = "PASS"
F = "FAIL"


def _score(tests):
	return sum(t[0] for t in tests), len(tests)


def print_detail(res, cats, has_neural):
	"""Detailed per-test results for the primary engine."""
	key = "nm" if has_neural else "cm"
	label = "Neural" if has_neural else "Classical"

	for cat in cats:
		tests = res[key].get(cat, [])
		p, t = _score(tests)
		print(f"\n{'─' * 56}")
		print(f"  {cat}  ({label})  [{p}/{t}]")
		print(f"{'─' * 56}")
		for i, (ok, san, exp, desc) in enumerate(tests, 1):
			tag = P if ok else F
			print(f"  [{i}] {tag:4s}  {desc:<34s} {san:<10s} (expected {exp})")

	# Eval
	key_e = "ne" if has_neural else "ce"
	tests = res[key_e]
	if tests:
		p, t = _score(tests)
		print(f"\n{'─' * 56}")
		print(f"  Evaluation  ({label})  [{p}/{t}]")
		print(f"{'─' * 56}")
		for i, (ok, v, exp, desc) in enumerate(tests, 1):
			tag = P if ok else F
			print(f"  [{i}] {tag:4s}  {desc:<34s} val={v:+.3f}  (expected {exp})")


def print_summary(res, cats, has_neural):
	"""Side-by-side summary table.  Returns (n_pass, n_total) for neural."""
	print(f"\n{'=' * 56}")
	print(f"  SUMMARY")
	print(f"{'=' * 56}")

	hdr = f"  {'Category':<20s}"
	if has_neural:
		hdr += f"{'Neural':>10s}"
	hdr += f"{'Classical':>12s}"
	print(hdr)
	print(f"  {'─' * 50}")

	n_p, n_t = 0, 0
	c_p, c_t = 0, 0

	for cat in cats:
		line = f"  {cat:<20s}"
		if has_neural:
			p, t = _score(res["nm"].get(cat, []))
			n_p += p; n_t += t
			line += f"{p:>4d}/{t:<4d}    "
		p, t = _score(res["cm"].get(cat, []))
		c_p += p; c_t += t
		line += f"{p:>4d}/{t:<4d}"
		print(line)

	# Eval row
	line = f"  {'Evaluation':<20s}"
	if has_neural:
		p, t = _score(res["ne"])
		n_p += p; n_t += t
		line += f"{p:>4d}/{t:<4d}    "
	p, t = _score(res["ce"])
	c_p += p; c_t += t
	line += f"{p:>4d}/{t:<4d}"
	print(line)

	print(f"  {'─' * 50}")
	line = f"  {'TOTAL':<20s}"
	if has_neural:
		pct = 100 * n_p / n_t if n_t else 0
		line += f"{n_p:>4d}/{n_t:<4d} ({pct:4.0f}%)"
	pct = 100 * c_p / c_t if c_t else 0
	line += f"  {c_p:>4d}/{c_t:<4d} ({pct:4.0f}%)"
	print(line)

	return n_p, n_t


# ═══════════════════════════════════════════════════════════════════════
# Training-history view
# ═══════════════════════════════════════════════════════════════════════

def run_history(checkpoint_dir, device, sims, depth=3):
	pattern = os.path.join(checkpoint_dir, "model_iter_*.pt")
	files = sorted(glob.glob(pattern))
	if not files:
		print("\n  No numbered checkpoints found in", checkpoint_dir)
		return

	cats = _ordered_categories()

	print(f"\n{'=' * 70}")
	print(f"  TRAINING PROGRESS  ({len(files)} checkpoints, {sims} sims/move)")
	print(f"{'=' * 70}")

	hdr = f"  {'Iter':>5s}"
	for cat in cats:
		hdr += f"  {cat[:8]:>8s}"
	hdr += f"  {'Eval':>6s}  {'Total':>12s}"
	print(hdr)
	print(f"  {'─' * 64}")

	for fpath in files:
		ckpt = torch.load(fpath, map_location=device, weights_only=False)
		it = ckpt.get("iteration", "?")
		m = ChessNet(
			num_res_blocks=ckpt.get("num_res_blocks", 10),
			num_filters=ckpt.get("num_filters", 128),
		)
		m.load_state_dict(ckpt["model_state_dict"])
		m.to(device)
		m.eval()

		r, _ = run_suite(m, device, sims, depth)

		gp, gt = 0, 0
		line = f"  {str(it):>5s}"
		for cat in cats:
			p, t = _score(r["nm"].get(cat, []))
			gp += p; gt += t
			line += f"    {p:>2d}/{t:<2d}  "
		p, t = _score(r["ne"])
		gp += p; gt += t
		line += f"  {p:>2d}/{t:<2d}"
		pct = 100 * gp / gt if gt else 0
		line += f"   {gp:>2d}/{gt:<2d} ({pct:4.0f}%)"
		print(line)

	# Classical baseline row
	r, _ = run_suite(None, device, sims, depth)
	gp, gt = 0, 0
	line = f"  {'base':>5s}"
	for cat in cats:
		p, t = _score(r["cm"].get(cat, []))
		gp += p; gt += t
		line += f"    {p:>2d}/{t:<2d}  "
	p, t = _score(r["ce"])
	gp += p; gt += t
	line += f"  {p:>2d}/{t:<2d}"
	pct = 100 * gp / gt if gt else 0
	line += f"   {gp:>2d}/{gt:<2d} ({pct:4.0f}%)"
	print(f"  {'─' * 64}")
	print(line + "  (classical baseline)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
	ap = argparse.ArgumentParser(
		description="DeepChess — ground-truth validation",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	ap.add_argument("--checkpoint",
	                default=os.path.join(CHECKPOINTS_DIR, "latest.pt"),
	                help="Path to model checkpoint")
	ap.add_argument("--checkpoint-dir", default=CHECKPOINTS_DIR,
	                help="Directory with numbered checkpoints (for --history)")
	ap.add_argument("--simulations", type=int, default=200,
	                help="MCTS simulations per test position")
	ap.add_argument("--depth", type=int, default=3,
	                help="Classical engine search depth")
	ap.add_argument("--history", action="store_true",
	                help="Evaluate every numbered checkpoint and show progress")
	args = ap.parse_args()

	device = get_device()

	print(f"{'=' * 56}")
	print(f"  DeepChess Ground Truth Validation")
	print(f"{'=' * 56}")
	print(f"  Device           : {device}")
	print(f"  MCTS simulations : {args.simulations}")
	print(f"  Classical depth  : {args.depth}")

	# Load model
	model = None
	if os.path.exists(args.checkpoint):
		ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
		nr = ckpt.get("num_res_blocks", 10)
		nf = ckpt.get("num_filters", 128)
		model = ChessNet(num_res_blocks=nr, num_filters=nf)
		model.load_state_dict(ckpt["model_state_dict"])
		model.to(device)
		model.eval()
		it = ckpt.get("iteration", "?")
		params = sum(p.numel() for p in model.parameters())
		print(f"  Checkpoint       : {args.checkpoint} (iter {it})")
		print(f"  Architecture     : {nr} res blocks, {nf} filters ({params:,} params)")
	else:
		print(f"  Checkpoint       : not found ({args.checkpoint})")
		print(f"                     Running classical engine only.")

	# Run
	t0 = time.time()
	results, cats = run_suite(model, device, args.simulations, args.depth)
	elapsed = time.time() - t0

	has_neural = model is not None
	print_detail(results, cats, has_neural)
	n_pass, n_total = print_summary(results, cats, has_neural)
	print(f"\n  Completed in {elapsed:.1f}s")

	# History
	if args.history:
		run_history(args.checkpoint_dir, device, args.simulations, args.depth)

	# Exit code: 0 if >=60% pass, 1 otherwise (useful in CI)
	if has_neural and n_total > 0:
		sys.exit(0 if n_pass / n_total >= 0.6 else 1)


if __name__ == "__main__":
	main()
