"""Is the search stronger than the net it is built from?

    python -m src.sims_ladder --checkpoint checkpoints/pretrained_70M.pt

Self-play RL rests on one assumption, and the whole loop is downhill without
it: the search must be a *policy improvement operator*.  Training on MCTS visit
counts is only learning if those visit counts pick better moves than the raw
policy head they were seeded from.  Every run from run5 to run8 lost ground on
the ground-truth suite, at a rate proportional to the self-play fraction
(anchor 0.25 -2.40 pts/iter, anchor 0.50 -0.32, anchor 1.00 zero by
definition), which is the signature of a target that is *worse* than the net
rather than a loop that is merely mistuned.  This measures the assumption
directly instead of inferring it from the damage.

Two readings, both on the same 837-point suite the runs trend:

**The ladder.**  The suite scored at a range of simulation counts, plus a
zero-search row where the move is the argmax of the policy head with no tree at
all.  If the score climbs with simulations, search adds something and the loop
could in principle be uphill at a high enough budget -- and the crossover point
says what budget.  If it is flat or falls, the tree is re-ranking a distilled
depth-14 prior using a value head of its own and getting it wrong, and no
anchor fraction can fix that.

**Sharpness.**  The visit distribution against the prior it came from, on the
same positions.  A genuine improvement should be *more* confident than the
prior: the search has looked ahead and knows something the prior does not.

The run logs invite a comparison that does not hold up -- self-play targets at
1.91-2.30 nats against the pre-training targets' 1.24 -- but those 1.24 nats
are not a measurement of anything.  The pre-training target is *constructed*:
0.60 on Stockfish's move, 0.30 on the human's, 0.10 spread over the rest, and
its entropy is whatever that recipe implies.  Setting it beside a search output
compares a label-smoothing constant to a distribution.  The honest comparison
is prior against visits on identical positions, which is what this measures.

Two knobs, because two different quantities are wanted.  *Noise on* reproduces
the training target exactly, Dirichlet root noise included, since that is what
self-play writes into the replay buffer.  *Noise off* asks the narrower
question of whether the tree sharpens the prior at all.  A gap between them
localises the damage in the exploration noise rather than in the search.

Scoring matches :func:`src.validate_gt.score_model` exactly -- same suite, same
``batch_size=1`` MCTS, same acceptance sets -- so a row here is directly
comparable to the "Ground truth" lines in any run log.
"""
import argparse
import json
import math
import os
import sys
import time

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.board_utils import encode_board, get_legal_move_indices
from src.engine import get_device
from src.mcts import MCTS
from src.model import ChessNet
from src.validate_gt import (EVAL_TESTS, MOVE_TESTS, _eval_ok, neural_eval)


def load_model(path, device):
	ckpt = torch.load(path, map_location=device, weights_only=False)
	model = ChessNet(num_res_blocks=ckpt.get("num_res_blocks", 16),
	                 num_filters=ckpt.get("num_filters", 192))
	model.load_state_dict(ckpt["model_state_dict"])
	model.to(device).eval()
	return model, ckpt


@torch.no_grad()
def prior_over_legal(model, device, board):
	"""``(indices, probabilities)`` for the policy head, masked to legal moves.

	The mask-then-renormalise is what the search itself does when it expands a
	node, so this is the prior the tree starts from rather than the raw 4672-way
	softmax, most of whose mass sits on moves that are not available.
	"""
	state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
	logits, _value = model(state)
	logits = logits[0].float().cpu().numpy()
	legal_moves, indices = get_legal_move_indices(board)
	if not indices:
		return legal_moves, indices, np.zeros(0, dtype=np.float64)
	sub = logits[indices].astype(np.float64)
	sub -= sub.max()
	probs = np.exp(sub)
	probs /= probs.sum()
	return legal_moves, indices, probs


def entropy_top1(probs):
	"""``(entropy in nats, largest probability)``; zero-safe."""
	probs = np.asarray(probs, dtype=np.float64)
	probs = probs[probs > 0.0]
	if probs.size == 0:
		return 0.0, 0.0
	return float(-(probs * np.log(probs)).sum()), float(probs.max())


@torch.no_grad()
def score_raw_policy(model, device):
	"""Move-test score with no tree at all: argmax of the masked policy head.

	This is the zero-search baseline every other row is compared against.  It is
	not ``sims=1``: a one-simulation search still expands the root and returns a
	visit distribution, so it would measure a degenerate tree rather than the
	net's own opinion.
	"""
	passed = 0
	breakdown = {}
	for cat, fen, acceptable, _desc in MOVE_TESTS:
		board = chess.Board(fen)
		legal_moves, _indices, probs = prior_over_legal(model, device, board)
		ok = 0
		if len(probs):
			ok = int(legal_moves[int(np.argmax(probs))].uci() in acceptable)
		p, t = breakdown.get(cat, (0, 0))
		breakdown[cat] = (p + ok, t + 1)
		passed += ok
	return passed, len(MOVE_TESTS), breakdown


@torch.no_grad()
def score_with_search(model, device, sims):
	"""Move-test score at *sims* simulations.

	``batch_size=1`` deliberately: it is what :func:`score_model` uses, so these
	numbers sit on the same scale as every "Ground truth" line in the run logs.
	Batching would be faster and would also change the search, since virtual
	loss pushes a batched descent onto paths a serial one would not take.
	"""
	mcts = MCTS(model, device, num_simulations=sims)
	passed = 0
	breakdown = {}
	for cat, fen, acceptable, _desc in MOVE_TESTS:
		board = chess.Board(fen)
		move, _policy = mcts.search(board, temperature=0.01)
		ok = int(move is not None and move.uci() in acceptable)
		p, t = breakdown.get(cat, (0, 0))
		breakdown[cat] = (p + ok, t + 1)
		passed += ok
	return passed, len(MOVE_TESTS), breakdown


@torch.no_grad()
def score_eval_tests(model, device):
	"""The suite's evaluation half, which no amount of search can move.

	Reported once and added to every ladder row so the totals stay comparable to
	the 837-point figure, but kept separate in the arithmetic: mixing a
	search-invariant 278 points into the ladder would damp exactly the slope the
	ladder exists to measure.
	"""
	ok = 0
	for fen, expected, _desc in EVAL_TESTS:
		ok += int(_eval_ok(neural_eval(model, device, fen), expected,
		                   chess.Board(fen).turn))
	return ok, len(EVAL_TESTS)


@torch.no_grad()
def sharpness(model, device, sims, add_noise, limit=0, fens=None):
	"""Prior vs visit distribution on the same positions.

	*add_noise* selects which quantity is being measured.  True reproduces the
	training target exactly -- Dirichlet noise at the root is part of what
	self-play writes into the replay buffer, so a comparison without it would
	flatter the loop.  False asks the narrower question of whether the search
	sharpens the prior at all once the exploration noise is set aside.

	*fens* replaces the move-test positions, and which set is used decides what
	the answer means.  The suite is tactics -- mates, hanging pieces, positions
	built so that one move is right -- exactly where a tree pays off, so
	measuring only there overstates what the search does for the loop.  The
	loop's targets are written on self-play game positions, most of them quiet,
	and that is the distribution to pass in here.
	"""
	mcts = MCTS(model, device, num_simulations=sims)
	if fens is not None:
		rows = [(None, f, None, None) for f in (fens[:limit] if limit
		                                        else fens)]
	else:
		rows = MOVE_TESTS[:limit] if limit else MOVE_TESTS
	pri_h, pri_t1, vis_h, vis_t1, agree, legal_n = [], [], [], [], [], []
	for _cat, fen, _acceptable, _desc in rows:
		board = chess.Board(fen)
		legal_moves, indices, probs = prior_over_legal(model, device, board)
		if len(probs) < 2:
			continue
		_move, visit_policy = mcts.search(board, temperature=1.0,
		                                  add_noise=add_noise)
		visits = np.asarray(visit_policy, dtype=np.float64)[indices]
		if visits.sum() <= 0.0:
			continue
		visits /= visits.sum()
		h_p, t1_p = entropy_top1(probs)
		h_v, t1_v = entropy_top1(visits)
		pri_h.append(h_p)
		pri_t1.append(t1_p)
		vis_h.append(h_v)
		vis_t1.append(t1_v)
		agree.append(int(np.argmax(probs) == np.argmax(visits)))
		legal_n.append(len(indices))
	if not pri_h:
		return None
	return {
		"n": len(pri_h),
		"prior_entropy": float(np.mean(pri_h)),
		"prior_top1": float(np.mean(pri_t1)),
		"visit_entropy": float(np.mean(vis_h)),
		"visit_top1": float(np.mean(vis_t1)),
		"argmax_agreement": float(np.mean(agree)),
		"mean_legal": float(np.mean(legal_n)),
	}


def main():
	ap = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--checkpoint", default="checkpoints/pretrained_70M.pt")
	ap.add_argument("--sims", default="25,50,100,200,400,800,1600,3200",
	                help="Comma-separated ladder, scored in ascending order so "
	                     "an interrupted run still leaves the cheap rows behind")
	ap.add_argument("--sharpness-sims", default="400",
	                help="Simulation counts to measure prior-vs-visits at.  "
	                     "400 is what the self-play loop ran, so it is the one "
	                     "that describes the targets run8 actually trained on")
	ap.add_argument("--sharpness-limit", type=int, default=150,
	                help="Positions to sample for sharpness.  The quantity is "
	                     "an average over positions, not a pass count, so it "
	                     "converges long before the full 559")
	ap.add_argument("--sharpness-fens", default="",
	                help="Measure sharpness on the FENs in this file (first "
	                     "tab-separated field of each line) instead of on the "
	                     "test suite.  Point it at self-play positions: the "
	                     "suite is tactics, where a tree always pays, while "
	                     "the loop writes its targets on quiet game positions")
	ap.add_argument("--out", default="", help="Write results as JSON here")
	args = ap.parse_args()

	device = get_device()
	model, ckpt = load_model(args.checkpoint, device)
	sims_list = [int(s) for s in args.sims.split(",") if s.strip()]
	sharp_list = [int(s) for s in args.sharpness_sims.split(",") if s.strip()]

	print(f"Checkpoint : {args.checkpoint}")
	print(f"             {ckpt.get('num_res_blocks')}x{ckpt.get('num_filters')}"
	      f"  iteration {ckpt.get('iteration')}"
	      f"  pretrain_epochs {ckpt.get('pretrain_epochs')}")
	print(f"Device     : {device}")
	print(f"Suite      : {len(MOVE_TESTS)} move tests + {len(EVAL_TESTS)} eval "
	      f"tests = {len(MOVE_TESTS) + len(EVAL_TESTS)} points\n", flush=True)

	t0 = time.time()
	eval_ok, eval_n = score_eval_tests(model, device)
	print(f"Eval tests (search-invariant): {eval_ok}/{eval_n}  "
	      f"in {time.time() - t0:.0f}s\n", flush=True)

	results = {"checkpoint": args.checkpoint, "eval": [eval_ok, eval_n],
	           "ladder": [], "sharpness": []}

	print("  sims        moves            total        vs raw    time")
	print("  " + "-" * 60, flush=True)

	t0 = time.time()
	raw_ok, raw_n, raw_bd = score_raw_policy(model, device)
	secs = time.time() - t0
	print(f"  {'0 (raw)':>8}  {raw_ok:>4}/{raw_n} ({raw_ok / raw_n * 100:4.1f}%)"
	      f"   {raw_ok + eval_ok:>4}/{raw_n + eval_n}"
	      f"    {'baseline':>8}  {secs:5.0f}s", flush=True)
	results["ladder"].append({"sims": 0, "moves": [raw_ok, raw_n],
	                          "total": raw_ok + eval_ok, "secs": secs,
	                          "breakdown": raw_bd})

	for sims in sims_list:
		t0 = time.time()
		ok, n, bd = score_with_search(model, device, sims)
		secs = time.time() - t0
		print(f"  {sims:>8}  {ok:>4}/{n} ({ok / n * 100:4.1f}%)"
		      f"   {ok + eval_ok:>4}/{n + eval_n}"
		      f"    {ok - raw_ok:>+8}  {secs:5.0f}s", flush=True)
		results["ladder"].append({"sims": sims, "moves": [ok, n],
		                          "total": ok + eval_ok, "secs": secs,
		                          "breakdown": bd})
		if args.out:
			with open(args.out, "w") as fh:
				json.dump(results, fh, indent=1)

	sharp_fens = None
	if args.sharpness_fens:
		with open(args.sharpness_fens) as fh:
			sharp_fens = [ln.split("\t")[0].strip() for ln in fh
			              if ln.strip()]
		# Read from the head of a shard the games were appended to in play
		# order, so a plain prefix would be all openings.  An even stride over
		# the whole file keeps the mix of game phases the generator produced.
		if args.sharpness_limit and len(sharp_fens) > args.sharpness_limit:
			step = len(sharp_fens) // args.sharpness_limit
			sharp_fens = sharp_fens[::step][:args.sharpness_limit]
		results["sharpness_fens"] = args.sharpness_fens

	label = (f"{len(sharp_fens)} positions from {args.sharpness_fens}"
	         if sharp_fens else "test-suite positions (tactics)")
	print(f"\nSharpness — prior vs visit distribution, {label}")
	print("  sims  noise      prior H  top1     visits H  top1    argmax agree")
	print("  " + "-" * 68, flush=True)
	for sims in sharp_list:
		for add_noise in (True, False):
			s = sharpness(model, device, sims, add_noise,
			              limit=args.sharpness_limit, fens=sharp_fens)
			if s is None:
				continue
			s.update(sims=sims, add_noise=add_noise)
			results["sharpness"].append(s)
			print(f"  {sims:>4}  {'on ' if add_noise else 'off':<9}"
			      f"  {s['prior_entropy']:6.3f}  {s['prior_top1']:.3f}"
			      f"    {s['visit_entropy']:7.3f}  {s['visit_top1']:.3f}"
			      f"        {s['argmax_agreement'] * 100:5.1f}%", flush=True)

	print(f"\n(n={results['sharpness'][0]['n'] if results['sharpness'] else 0} "
	      f"positions, mean "
	      f"{results['sharpness'][0]['mean_legal']:.1f} legal moves; "
	      f"noise 'on' is the actual training target)")

	if args.out:
		with open(args.out, "w") as fh:
			json.dump(results, fh, indent=1)
		print(f"\nWrote {args.out}")


if __name__ == "__main__":
	main()
