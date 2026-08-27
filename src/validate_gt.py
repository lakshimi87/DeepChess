#!/usr/bin/env python3
"""Ground truth validation for DeepChess.

Tests the neural engine (and classical baseline) against curated positions
with known best moves and known evaluations.  Use after training to measure
how well the model has learned.

    ./validate_gt.sh                    # test latest checkpoint + 20-game match
    ./validate_gt.sh --history          # show progress across all checkpoints
    ./validate_gt.sh --simulations 400  # more MCTS sims (slower but fairer)
    ./validate_gt.sh --games 0          # skip the head-to-head match
    ./validate_gt.sh --games 50         # longer match for tighter win-rate CI
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import chess
import torch

from .board_utils import encode_board, get_legal_move_indices
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

	("Mate in 1",
	 "k7/8/1K6/2Q5/8/8/8/8 w - - 0 1",
	 ["c5c8"],
	 "Queen mates K in corner"),

	("Mate in 1",
	 "6rk/6pp/7N/8/8/8/8/4K3 w - - 0 1",
	 ["h6f7"],
	 "Smothered mate Nf7#"),

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

	("Win Material",
	 "rnbqkbnr/ppp1pppp/8/8/3P4/2N1n3/PPP1PPPP/R1BQKBNR w KQkq - 0 1",
	 ["f2e3"],
	 "Capture hanging knight (pawn)"),

	# ── Endgame: technique that decides a won game ───────────────
	("Endgame",
	 "k7/4P3/4K3/8/8/8/8/8 w - - 0 1",
	 ["e7e8q", "e7e8r", "e7e8b", "e7e8n"],
	 "Promote pawn (winning)"),

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

	# Endgame evals (clear material picture; tests the value head on
	# endgames it rarely sees during mid-game self-play).
	("4k3/8/4K3/8/8/8/8/3Q4 w - - 0 1",
	 "white",
	 "K+Q vs K (winning)"),

	("4k3/8/4K3/8/8/8/8/3R4 w - - 0 1",
	 "white",
	 "K+R vs K (winning)"),

	("8/8/4k3/8/4B3/4K3/8/8 w - - 0 1",
	 "draw",
	 "K+B vs K (insufficient material)"),
]


# The 27 tests above are hand-written and stay as the core.  A generated
# suite (tools/build_gt_suite.py) extends them when present: 27 positions put
# one sigma at +/-9.6 points, which is wider than any effect a training run
# produces, so run4's 41%-63% swing over 258 iterations was indistinguishable
# from noise.  Loading happens at import so the training loop's periodic
# score_model() call measures against the same suite validate_gt.sh does.

CORE_MOVE_TESTS = len(MOVE_TESTS)
CORE_EVAL_TESTS = len(EVAL_TESTS)
SUITE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "resources", "gt_suite.json")
SUITE_META = None


def load_generated_suite(path=None):
	"""Append the generated suite to the built-in tests.  Idempotent."""
	global SUITE_META
	path = path or SUITE_PATH
	del MOVE_TESTS[CORE_MOVE_TESTS:]
	del EVAL_TESTS[CORE_EVAL_TESTS:]
	SUITE_META = None
	if not os.path.exists(path):
		return 0
	try:
		with open(path) as fh:
			suite = json.load(fh)
	except (OSError, ValueError) as e:
		print(f"warning: could not read {path}: {e}", file=sys.stderr)
		return 0
	for t in suite.get("move_tests", []):
		MOVE_TESTS.append((t["category"], t["fen"], t["moves"], t["desc"]))
	for t in suite.get("eval_tests", []):
		EVAL_TESTS.append((t["fen"], t["expected"], t["desc"]))
	SUITE_META = suite.get("meta")
	return len(suite.get("move_tests", [])) + len(suite.get("eval_tests", []))


load_generated_suite()


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

def neural_move(model, device, fen, sims, top_k=3):
	"""Run MCTS and return chosen move plus diagnostic info.

	Returns dict with: uci, san, top (list of (san, prob) tuples sorted by
	visit count), val (raw value-head output from current player's POV).
	"""
	board = chess.Board(fen)
	mcts = MCTS(model, device, num_simulations=sims)
	move, policy = mcts.search(board, temperature=0.01)

	# Raw value head (independent of MCTS).
	state = encode_board(board)
	with torch.no_grad():
		t = torch.from_numpy(state).unsqueeze(0).to(device)
		_, v = model(t)
	val = float(v.item())

	if move is None:
		return dict(uci=None, san="-", top=[], val=val)

	legal_moves, indices = get_legal_move_indices(board)
	scored = sorted(
		((float(policy[idx]), m) for m, idx in zip(legal_moves, indices)),
		key=lambda x: x[0],
		reverse=True,
	)
	top = []
	for prob, m in scored[:top_k]:
		if prob <= 0:
			break
		try:
			top.append((board.san(m), prob))
		except Exception:
			top.append((m.uci(), prob))
	return dict(uci=move.uci(), san=board.san(move), top=top, val=val)


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


def run_suite(model, device, sims, depth=3, with_classical=True):
	"""Run every test.  Returns ``(results_dict, categories)``.

	Each entry in ``res['nm'][cat]`` / ``res['cm'][cat]`` is a dict with
	keys: ok, san, uci, exp, desc, dt, n_legal, and (neural only) top, val.

	*with_classical* runs the pure-Python minimax baseline alongside the net.
	It costs roughly half a second per position, which is nothing across the
	27 hand-written tests and several minutes across a generated suite, so
	the caller turns it off once the suite is large.
	"""
	cats = _ordered_categories()
	res = dict(nm={}, cm={}, ne=[], ce=[])

	for cat, fen, acceptable, desc in MOVE_TESTS:
		exp_san = _uci_to_san(fen, acceptable)
		n_legal = chess.Board(fen).legal_moves.count()

		if model is not None:
			t0 = time.time()
			nm = neural_move(model, device, fen, sims)
			dt = time.time() - t0
			ok = nm["uci"] in acceptable
			res["nm"].setdefault(cat, []).append(dict(
				ok=ok, san=nm["san"], uci=nm["uci"], exp=exp_san,
				desc=desc, dt=dt, n_legal=n_legal,
				top=nm["top"], val=nm["val"],
			))

		if with_classical:
			t0 = time.time()
			uci, san = classical_move(fen, depth)
			dt = time.time() - t0
			ok = (uci in acceptable) if uci else False
			res["cm"].setdefault(cat, []).append(dict(
				ok=ok, san=san, uci=uci, exp=exp_san,
				desc=desc, dt=dt, n_legal=n_legal,
			))

	for fen, expected, desc in EVAL_TESTS:
		board = chess.Board(fen)
		if model is not None:
			v = neural_eval(model, device, fen)
			ok = _eval_ok(v, expected, board.turn)
			res["ne"].append(dict(ok=ok, val=v, exp=expected, desc=desc))
		if with_classical:
			v = classical_eval(fen)
			ok = _eval_ok(v, expected, board.turn)
			res["ce"].append(dict(ok=ok, val=v, exp=expected, desc=desc))

	return res, cats


def score_model(model, device, sims):
	"""Neural-only score on the ground-truth suite: ``(passed, total, breakdown)``.

	:func:`run_suite` also runs the classical engine on every position, which
	is pure waste when all that is wanted is a progress number for the net.

	This is the metric to trend across a training run.  Unlike an arena match
	it is an *absolute* yardstick: it cannot drift with the opponent, and it
	has no promotion ratchet to launder noise into apparent progress.
	"""
	passed = total = 0
	breakdown = {}
	for cat, fen, acceptable, _desc in MOVE_TESTS:
		ok = int(neural_move(model, device, fen, sims)["uci"] in acceptable)
		p, t = breakdown.get(cat, (0, 0))
		breakdown[cat] = (p + ok, t + 1)
		passed += ok
		total += 1
	ep = et = 0
	for fen, expected, _desc in EVAL_TESTS:
		ep += int(_eval_ok(neural_eval(model, device, fen), expected,
		                   chess.Board(fen).turn))
		et += 1
	breakdown["Eval"] = (ep, et)
	return passed + ep, total + et, breakdown


# ═══════════════════════════════════════════════════════════════════════
# Pretty printing
# ═══════════════════════════════════════════════════════════════════════

P = "PASS"
F = "FAIL"


def _score(tests):
	return sum(t["ok"] for t in tests), len(tests)


def _fmt_top(top):
	"""Format top-K candidate moves as 'Nf3 62% e4 18% d4 12%'."""
	if not top:
		return "-"
	return "  ".join(f"{s} {p*100:.0f}%" for s, p in top)


def print_detail(res, cats, has_neural):
	"""Detailed per-test results.

	When the neural engine is available, prints neural and classical
	side-by-side per test plus top-3 candidates and the value-head output.
	"""
	width = 78

	for cat in cats:
		nm = res["nm"].get(cat, []) if has_neural else []
		cm = res["cm"].get(cat, [])
		tests_for_score = nm if has_neural else cm
		p, t = _score(tests_for_score)

		# Header summarises pass-rate and avg neural time.
		header = f"  {cat}  ({'Neural' if has_neural else 'Classical'}) [{p}/{t}]"
		if has_neural and nm:
			avg_ms = 1000 * sum(x["dt"] for x in nm) / len(nm)
			header += f"   avg {avg_ms:.0f} ms/move"
		print(f"\n{'─' * width}")
		print(header)
		print(f"{'─' * width}")

		count = max(len(nm), len(cm))
		for i in range(count):
			n = nm[i] if i < len(nm) else None
			c = cm[i] if i < len(cm) else None
			desc = (n or c)["desc"]
			exp = (n or c)["exp"]
			n_legal = (n or c)["n_legal"]
			print(f"  [{i+1:2d}] {desc}   ({n_legal} legal, expected {exp})")
			if n is not None:
				tag = P if n["ok"] else F
				print(f"        N {tag:4s} {n['san']:<8s} val={n['val']:+.2f}  "
				      f"top: {_fmt_top(n['top'])}")
			if c is not None:
				tag = P if c["ok"] else F
				print(f"        C {tag:4s} {c['san']:<8s}")

	# Eval tests
	ne = res["ne"] if has_neural else []
	ce = res["ce"]
	if ne or ce:
		tests_for_score = ne if has_neural else ce
		p, t = _score(tests_for_score)
		print(f"\n{'─' * width}")
		print(f"  Evaluation  ({'Neural' if has_neural else 'Classical'}) [{p}/{t}]")
		print(f"{'─' * width}")
		count = max(len(ne), len(ce))
		for i in range(count):
			n = ne[i] if i < len(ne) else None
			c = ce[i] if i < len(ce) else None
			desc = (n or c)["desc"]
			exp = (n or c)["exp"]
			print(f"  [{i+1:2d}] {desc}   (expected {exp})")
			if n is not None:
				tag = P if n["ok"] else F
				print(f"        N {tag:4s} val={n['val']:+.3f}")
			if c is not None:
				tag = P if c["ok"] else F
				print(f"        C {tag:4s} val={c['val']:+.3f}")


def print_summary(res, cats, has_neural):
	"""Side-by-side summary table.  Returns (n_pass, n_total) for neural."""
	width = 62
	print(f"\n{'=' * width}")
	print(f"  SUMMARY")
	print(f"{'=' * width}")

	hdr = f"  {'Category':<20s}"
	if has_neural:
		hdr += f"{'Neural':>12s}"
	hdr += f"{'Classical':>14s}"
	print(hdr)
	print(f"  {'─' * (width - 2)}")

	n_p, n_t = 0, 0
	c_p, c_t = 0, 0

	for cat in cats:
		line = f"  {cat:<20s}"
		if has_neural:
			p, t = _score(res["nm"].get(cat, []))
			n_p += p; n_t += t
			line += f"{p:>5d}/{t:<5d} "
		p, t = _score(res["cm"].get(cat, []))
		c_p += p; c_t += t
		line += f"{p:>6d}/{t:<5d}"
		print(line)

	line = f"  {'Evaluation':<20s}"
	if has_neural:
		p, t = _score(res["ne"])
		n_p += p; n_t += t
		line += f"{p:>5d}/{t:<5d} "
	p, t = _score(res["ce"])
	c_p += p; c_t += t
	line += f"{p:>6d}/{t:<5d}"
	print(line)

	print(f"  {'─' * (width - 2)}")
	line = f"  {'TOTAL':<20s}"
	if has_neural:
		pct = 100 * n_p / n_t if n_t else 0
		line += f"{n_p:>5d}/{n_t:<3d} ({pct:3.0f}%) "
	pct = 100 * c_p / c_t if c_t else 0
	line += f"{c_p:>4d}/{c_t:<3d} ({pct:3.0f}%)"
	print(line)

	return n_p, n_t


def print_comparison(res, cats):
	"""Neural vs Classical: agreement, both-pass, confidence, timing.

	Only printed when both engines ran (i.e. a neural checkpoint loaded).
	"""
	width = 78
	print(f"\n{'=' * width}")
	print("  NEURAL vs CLASSICAL")
	print(f"{'=' * width}")
	print(f"  {'Category':<16s} {'Agree':>9s} {'BothOK':>9s} "
	      f"{'Conf':>8s} {'N ms':>8s} {'C ms':>8s}")
	print(f"  {'─' * (width - 4)}")

	tot_agree = tot_both = tot_n = 0
	confs = []
	n_times = []
	c_times = []
	for cat in cats:
		nm = res["nm"].get(cat, [])
		cm = res["cm"].get(cat, [])
		if not nm or not cm:
			continue
		agree = sum(1 for n, c in zip(nm, cm) if n["uci"] == c["uci"])
		both = sum(1 for n, c in zip(nm, cm) if n["ok"] and c["ok"])
		cat_conf = [n["top"][0][1] for n in nm if n["top"]]
		avg_conf = sum(cat_conf) / len(cat_conf) if cat_conf else 0.0
		avg_n = 1000 * sum(n["dt"] for n in nm) / len(nm)
		avg_c = 1000 * sum(c["dt"] for c in cm) / len(cm)
		confs.extend(cat_conf)
		n_times.extend(n["dt"] for n in nm)
		c_times.extend(c["dt"] for c in cm)
		tot_agree += agree
		tot_both += both
		tot_n += len(nm)
		print(f"  {cat:<16s} {agree:>4d}/{len(nm):<4d} "
		      f"{both:>4d}/{len(nm):<4d} "
		      f"{avg_conf*100:>6.0f}%  {avg_n:>6.0f}  {avg_c:>6.0f}")

	if tot_n:
		print(f"  {'─' * (width - 4)}")
		avg_conf = sum(confs) / len(confs) if confs else 0.0
		avg_n = 1000 * sum(n_times) / len(n_times) if n_times else 0.0
		avg_c = 1000 * sum(c_times) / len(c_times) if c_times else 0.0
		print(f"  {'TOTAL':<16s} {tot_agree:>4d}/{tot_n:<4d} "
		      f"{tot_both:>4d}/{tot_n:<4d} "
		      f"{avg_conf*100:>6.0f}%  {avg_n:>6.0f}  {avg_c:>6.0f}")


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
			num_res_blocks=ckpt.get("num_res_blocks", 16),
			num_filters=ckpt.get("num_filters", 192),
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
# Head-to-head match: Neural vs Classical (the "ground truth" opponent)
# ═══════════════════════════════════════════════════════════════════════

def play_match(model, device, num_games, sims, depth,
               opening_temp_plies=8, max_plies=300):
	"""Play *num_games* between the neural engine and the classical engine.

	Colors alternate every game so neither side gets a permanent first-move
	advantage.  A small temperature is applied to the neural engine's
	opening moves so the 20 games don't collapse into one deterministic
	line repeated 10 times.

	Returns ``(wins, losses, draws)`` from the neural engine's perspective.
	"""
	classical = _ClassicalEngine(depth=depth)
	mcts = MCTS(model, device, num_simulations=sims, batch_size=16)

	wins = losses = draws = 0
	width = 78
	print(f"\n{'=' * width}")
	print(f"  MATCH  Neural vs Classical (GT)   {num_games} games, "
	      f"{sims} sims, depth {depth}")
	print(f"{'=' * width}")
	print(f"  {'#':>3s}  {'Neural':>7s}  {'Result':>7s}  {'Outcome':>8s}  "
	      f"{'Plies':>5s}  {'Time':>7s}")
	print(f"  {'─' * (width - 4)}")

	match_t0 = time.time()
	for g in range(num_games):
		neural_is_white = (g % 2 == 0)
		board = chess.Board()
		ply = 0
		g_t0 = time.time()

		while not board.is_game_over(claim_draw=True) and ply < max_plies:
			neural_to_move = (board.turn == chess.WHITE) == neural_is_white
			if neural_to_move:
				temp = 1.0 if ply < opening_temp_plies else 0.01
				move, _ = mcts.search(board, temperature=temp)
			else:
				move = classical.get_move(board)
			if move is None:
				break
			board.push(move)
			ply += 1

		result = board.result(claim_draw=True)
		if result == "1-0":
			neural_won = neural_is_white
		elif result == "0-1":
			neural_won = not neural_is_white
		else:
			neural_won = None

		if neural_won is True:
			wins += 1
			outcome = "WIN"
		elif neural_won is False:
			losses += 1
			outcome = "LOSS"
		else:
			draws += 1
			outcome = "DRAW"

		dt = time.time() - g_t0
		color = "white" if neural_is_white else "black"
		print(f"  {g+1:>3d}  {color:>7s}  {result:>7s}  {outcome:>8s}  "
		      f"{ply:>5d}  {dt:>6.1f}s")

	total = wins + losses + draws
	score = wins + 0.5 * draws
	win_rate = 100 * score / total if total else 0.0
	elapsed = time.time() - match_t0

	print(f"  {'─' * (width - 4)}")
	print(f"  RESULT  W={wins}  L={losses}  D={draws}   "
	      f"score {score:.1f}/{total}   win rate {win_rate:.1f}%   "
	      f"({elapsed:.1f}s)")

	return wins, losses, draws


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
	ap.add_argument("--simulations", type=int, default=400,
	                help="MCTS simulations per test position")
	ap.add_argument("--depth", type=int, default=3,
	                help="Classical engine search depth")
	ap.add_argument("--history", action="store_true",
	                help="Evaluate every numbered checkpoint and show progress")
	ap.add_argument("--games", type=int, default=20,
	                help="Play this many games against the classical engine "
	                     "(0 to skip)")
	ap.add_argument("--core-only", action="store_true",
	                help="Use only the 27 hand-written tests, ignoring any "
	                     "generated suite in resources/.  Fast, but one sigma "
	                     "is +/-9.6 points — too coarse to read a training "
	                     "run's progress from.")
	ap.add_argument("--suite", default=None,
	                help="Path to a generated suite JSON "
	                     "(default: resources/gt_suite.json)")
	ap.add_argument("--classical", dest="classical", default=None,
	                action="store_true",
	                help="Force the classical baseline on.  It is on by "
	                     "default for the 27 hand-written tests and off for a "
	                     "generated suite, where its ~0.5s/position would add "
	                     "several minutes without changing what the net scores.")
	ap.add_argument("--no-classical", dest="classical", action="store_false")
	ap.add_argument("--detail", action="store_true",
	                help="Print every test individually.  Off by default: a "
	                     "generated suite runs to hundreds of positions.")
	args = ap.parse_args()

	if args.core_only:
		load_generated_suite("")
	elif args.suite:
		load_generated_suite(args.suite)

	device = get_device()

	print(f"{'=' * 56}")
	print(f"  DeepChess Ground Truth Validation")
	print(f"{'=' * 56}")
	print(f"  Device           : {device}")
	print(f"  MCTS simulations : {args.simulations}")
	print(f"  Classical depth  : {args.depth}")
	n_tests = len(MOVE_TESTS) + len(EVAL_TESTS)
	sigma = 50 / n_tests ** 0.5
	origin = ("hand-written only" if SUITE_META is None
	          else f"{CORE_MOVE_TESTS + CORE_EVAL_TESTS} hand-written + "
	               f"{n_tests - CORE_MOVE_TESTS - CORE_EVAL_TESTS} generated "
	               f"@ depth {SUITE_META.get('stockfish_depth', '?')}")
	# Printed with the suite because a score is unreadable without it: the
	# whole reason this suite was enlarged is that 27 tests could not tell a
	# real gain from a coin flip.
	print(f"  Test positions   : {n_tests} ({origin})")
	print(f"                     1 sigma on a 50% scorer: +/-{sigma:.1f} points")

	# Load model
	model = None
	if os.path.exists(args.checkpoint):
		ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
		nr = ckpt.get("num_res_blocks", 16)
		nf = ckpt.get("num_filters", 192)
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
	with_classical = args.classical
	if with_classical is None:
		with_classical = n_tests <= 100
	if not with_classical:
		print(f"  Classical engine : skipped ({n_tests} tests; "
		      f"pass --classical to include it)")

	t0 = time.time()
	results, cats = run_suite(model, device, args.simulations, args.depth,
	                          with_classical=with_classical)
	elapsed = time.time() - t0

	has_neural = model is not None
	if args.detail:
		print_detail(results, cats, has_neural)
	n_pass, n_total = print_summary(results, cats, has_neural)
	if has_neural and with_classical:
		print_comparison(results, cats)
	print(f"\n  Completed in {elapsed:.1f}s")

	# Head-to-head match
	if has_neural and args.games > 0:
		play_match(model, device, args.games, args.simulations, args.depth)

	# History
	if args.history:
		run_history(args.checkpoint_dir, device, args.simulations, args.depth)

	# Exit code: 0 if >=60% pass, 1 otherwise (useful in CI)
	if has_neural and n_total > 0:
		sys.exit(0 if n_pass / n_total >= 0.6 else 1)


if __name__ == "__main__":
	main()
