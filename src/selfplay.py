"""Parallel self-play.

Self-play is CPU-bound, not GPU-bound: at 600 sims/move roughly 45 ms per
move goes into python-chess move generation, board copies and PUCT
bookkeeping while only ~15 ms goes into GPU forward passes.  A single
self-play process therefore leaves both the GPU (~6% utilisation) and 27 of
28 CPU cores idle.

This module runs self-play in a pool of persistent worker processes.  Each
worker keeps its own fp16 inference copy of the network on the GPU and plays
whole games independently; the parent only ships weights down and collects
finished games.  Workers are *persistent* across training iterations so the
CUDA/spawn start-up cost is paid once per run rather than once per iteration.

Protocol (parent -> worker on ``task_q``):
    ("weights",     path)              reload fp16 weights from *path*
    ("ref_weights", path)              reload the arena opponent's fp16 weights
    ("play",        (game_id, resign)) play one self-play game; *resign* is
                                       the threshold this game runs at
    ("match",       (game_id, white))  current vs reference; *white* is True
                                       when the current net has white
    ("stop",        None)              exit

Replies (worker -> parent on ``result_q``):
    ("ready", rank,    None,     None,   None,  None)
    ("game",  game_id, examples, result, moves, seconds, audit)
    ("match", game_id, score,    result, moves, seconds)
    ("error", game_id, traceback, None,  None,  None)
"""

import os
import queue
import signal
import time
import traceback

import chess
import chess.syzygy
import numpy as np
import torch
import torch.multiprocessing as mp

from . import perf
from .board_utils import encode_board
from .mcts import MCTS
from .model import ChessNet


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

# Material values in pawns, for adjudication only.  Deliberately crude: the
# point of this scale is that it does not come from the network, so it cannot
# drift when the value head does.
_ADJ_VALUES = (
	(chess.QUEEN, 9.0),
	(chess.ROOK, 5.0),
	(chess.BISHOP, 3.0),
	(chess.KNIGHT, 3.0),
	(chess.PAWN, 1.0),
)


def material_balance(board):
	"""Material in pawns, positive when White is ahead."""
	total = 0.0
	for piece_type, value in _ADJ_VALUES:
		total += value * (len(board.pieces(piece_type, chess.WHITE))
		                  - len(board.pieces(piece_type, chess.BLACK)))
	return total


def probe_tablebase(tablebase, board):
	"""Exact game result for *board* from the side to move, or None.

	Returns 1.0 / 0.0 / -1.0 -- win, draw, loss -- and None when the position
	is not in the tables or cannot be probed.

	Why this and not the material rule.  Adjudicating on material asserts an
	outcome the game never reached, so it is a heuristic label: a fortress or a
	wrong-coloured bishop scores as a win, and a piece sacrificed into a mating
	net scores as a loss.  Both are exactly the endgame judgements the net most
	needs to get right, and run8 lost 15 of its 91 Endgame points over 100
	iterations with that rule supplying half the decisive labels.  A tablebase
	verdict is not an assertion, it is the result, so it carries no bias to
	learn.  It is also not the network's own opinion, which is what breaks the
	self-confirming loop resignation otherwise runs in.

	Cursed wins and blessed losses (WDL +/-1) are wins and losses that the
	50-move rule turns into draws, so they are reported as draws: the game
	really would be drawn if it were played on.
	"""
	if board.castling_rights:
		# Syzygy is indexed without castling rights.  Impossible to reach at
		# five men in practice, but a probe would raise rather than say so.
		return None
	try:
		wdl = tablebase.probe_wdl(board)
	except (KeyError, ValueError, IndexError, OSError):
		# A missing table is not an error here -- the loop simply plays on.
		return None
	if wdl > 1:
		return 1.0
	if wdl < -1:
		return -1.0
	return 0.0


def play_game(mcts, max_moves=512, value_discount=1.0, temp_moves=30,
              temp_high=1.0, temp_low=0.1, resign_threshold=0.0,
              resign_plies=2, resign_disable_frac=0.1,
              search_value_weight=0.0, adjudicate_material=0.0,
              adjudicate_plies=0, adjudicate_label=True,
              tablebase=None, tablebase_pieces=5):
	"""Play one self-play game with *mcts* and return (examples, result).

	*mcts* is reused across games — :meth:`MCTS.search` builds a fresh root
	every call, so there is no state to reset, and reusing it keeps the pinned
	staging buffer alive instead of reallocating it once per game.

	**Resignation.**  A side resigns once its own root Q has stayed at or below
	*resign_threshold* for *resign_plies* consecutive turns of its own.  This is
	not primarily a speed optimisation: a decided game that plays on to the
	50-move rule is scored a *draw*, so every one of its several hundred
	positions gets a 0.0 value label that contradicts the result the position
	actually deserves.  That is the single largest source of value-label noise
	in a weak-net loop, and it is what drives held-out value MSE above the
	predict-a-draw baseline.  Resignation is off when *resign_threshold* is
	>= 0.

	A *resign_disable_frac* share of games ignore resignation and play to the
	end.  Those games are the only way to see a false positive — a position
	resigned at -0.9 that was in fact holdable — so the fraction is what keeps
	the threshold auditable rather than self-confirming.

	**Adjudication.**  A side also wins once it has held a material lead of
	*adjudicate_material* pawns for *adjudicate_plies* consecutive plies.
	Unlike resignation this test never touches the network, which is the whole
	reason it exists: resignation is measured against an absolute root-Q
	threshold, so the moment the value head's output scale shrinks the
	threshold becomes unreachable and the *only* supply of decisive labels
	stops — run5 went from 74% resignations to 1% in one iteration and never
	recovered, because max|V| had fallen to 0.36 against a -0.9 threshold.  A
	material margin is measured in pawns and cannot drift with the net, so it
	keeps producing decisive labels through exactly the collapse that silences
	resignation.

	**Value targets.**  A pure game-outcome label is the same number for every
	position in a game, so in a draw-heavy corpus the constant 0.0 minimises
	the loss and the value head converges to it.  *search_value_weight* blends
	in each position's own root Q so that two decisive games stop sharing one
	label per side.

	The blend is applied to *decided* games only, and that restriction is load
	bearing.  MSE training drives V toward its target, so for a target
	``(1-w)*z + w*V`` the fixed point is V = z whatever w is: blending does not
	move the fixed point, it only decides how fast V gets there.  On a drawn
	game z is 0, so blending buys nothing — it just makes the target *look*
	varied while pulling V toward 0 by a factor of w per iteration out of the
	net's own output, with no external quantity anywhere in the loop.  That is
	the contraction that emptied run5's value head (mean|V| 0.669 -> 0.043 over
	five iterations, ~0.58x each).  A finished draw now takes a plain 0.0,
	which is simply the correct label for it; keeping the draw rate low enough
	that 0.0 is not the whole corpus is adjudication's job, not the blend's.

	Games cut off at *max_moves* still take the search value outright.  Those
	are not draws — they are unfinished — so 0.0 would be a wrong label rather
	than a degenerate one.  Adjudication should keep them rare; watch the
	reported rate and treat a rising one as the signal to tighten
	*adjudicate_material*.
	"""
	board = chess.Board()
	history = []  # (encoded_state, policy, turn, root_q)

	# One roll decides whether this game plays to a natural finish.  Both
	# early-stop mechanisms share it, so the same audit fraction that catches a
	# false resignation also catches a material lead that was in fact holdable.
	audit_game = np.random.random() < resign_disable_frac
	resign_enabled = resign_threshold < 0.0 and not audit_game
	# The material rule has two jobs and *adjudicate_label* separates them.
	# As an arbiter it decides, after the fact, whether a side that resigned
	# was in fact crushed -- that is the ground truth the resignation
	# calibration scores against, and it is recorded in `would_adjudicate`
	# below whatever else happens.  As a label source it ends the game and
	# writes a win, which is a heuristic target rather than a played result.
	# Switching the margin to 0 used to switch off both at once, which leaves
	# the calibration scoring against the scoreline alone -- and audit games
	# run with both early stops off end ~95% drawn, so every resignation then
	# counts as a false positive and no threshold can qualify.
	adjudicate_arbiter = adjudicate_material > 0.0 and adjudicate_plies > 0
	adjudicate_enabled = (adjudicate_arbiter and adjudicate_label
	                      and not audit_game)
	# Counted per colour: the root Q alternates POV every ply, so a single
	# counter would trip on two *different* sides each thinking they are lost.
	bad_turns = {chess.WHITE: 0, chess.BLACK: 0}
	resigned_by = None
	adjudicated_win = None
	# The same criterion, recorded rather than acted on.  In an audit game
	# adjudication is switched off, so this is the only surviving evidence that
	# a side was materially lost — and it is what the resignation calibration
	# scores against, because the game's own *result* cannot do that job: with
	# both mechanisms off these games end 95% drawn, since a net too weak to
	# convert grinds a won position to the 50-move rule.  Scoring resignations
	# against that result measures the conversion failure, not the resignation,
	# and reports 98% false positives for a net whose evaluations are right.
	# Material is not the network's opinion, so it breaks that circle.
	would_adjudicate = None
	adj_leader = None
	adj_plies = 0
	# A tablebase verdict is the game's real result reached early, not an
	# early stop, so unlike resignation and adjudication it also fires in
	# audit games.  That makes the calibration sample *better*: an audit game
	# that walks into a drawn five-man ending now ends a draw instead of
	# grinding to the 50-move rule, which is the outcome the resigning side's
	# false positive should be scored against.
	tb_winner = None
	tb_decided = False

	move_count = 0
	# board.is_game_over() defaults to claim_draw=False, so it ends a game on
	# the seventy-five-move rule and fivefold repetition but not on the fifty
	# and threefold a player would actually claim.  With repetition banned at
	# the root fivefold can no longer happen, so that left the game running to
	# --max-moves and labelling every position in it with root_q -- the net's
	# own opinion, which is the self-confirming loop this is all trying to get
	# out of.  Truncation went 13% -> 30% -> 63% over four iterations that way.
	#
	# The fifty-move test is added directly rather than via claim_draw=True,
	# which would also call can_claim_threefold_repetition() every move -- that
	# is the stack scan profiled at ~30% of per-move CPU.  This is the same
	# condition _is_terminal_fast already uses, so the search and the game now
	# agree about when a game is over instead of differing by fifty moves.
	while (not board.is_game_over() and board.halfmove_clock < 100
	       and move_count < max_moves):
		temperature = temp_high if move_count < temp_moves else temp_low
		state = encode_board(board)
		move, policy = mcts.search(board, temperature=temperature, add_noise=True)
		if move is None:
			break
		# Read root_q before the push: it is the search's evaluation of *this*
		# position from the mover's point of view, the same perspective the
		# outcome label uses.
		history.append((state, policy, board.turn, mcts.root_q))
		mover = board.turn
		board.push(move)
		move_count += 1

		if (tablebase is not None
				and chess.popcount(board.occupied) <= tablebase_pieces):
			wdl = probe_tablebase(tablebase, board)
			if wdl is not None:
				# wdl is from the POV of the side to move *now*, which is the
				# side that did not just move.
				if wdl > 0:
					tb_winner = board.turn
				elif wdl < 0:
					tb_winner = not board.turn
				else:
					tb_winner = None
				tb_decided = True
				break

		if resign_enabled:
			# root_q is from the POV of the side that was about to move, i.e.
			# `mover` — read it after the push, but attribute it to `mover`.
			if mcts.root_q <= resign_threshold:
				bad_turns[mover] += 1
				if bad_turns[mover] >= resign_plies:
					resigned_by = mover
					break
			else:
				bad_turns[mover] = 0

		if adjudicate_arbiter:
			balance = material_balance(board)
			leader = None
			if abs(balance) >= adjudicate_material:
				leader = chess.WHITE if balance > 0 else chess.BLACK
			if leader is None:
				adj_leader, adj_plies = None, 0
			else:
				adj_plies = adj_plies + 1 if leader == adj_leader else 1
				adj_leader = leader
				if adj_plies >= adjudicate_plies:
					if would_adjudicate is None:
						would_adjudicate = leader
					if adjudicate_enabled:
						adjudicated_win = leader
						break

	if tb_decided:
		winner = tb_winner
	elif resigned_by is not None:
		winner = not resigned_by
	elif adjudicated_win is not None:
		winner = adjudicated_win
	elif board.is_checkmate():
		winner = not board.turn  # side to move is mated
	else:
		winner = None  # draw (or truncated at max_moves)

	# A game that ran out of moves is unfinished, not drawn.  Its outcome
	# label carries no information at all, so the search value replaces it
	# rather than being blended with it.
	truncated = (not tb_decided and resigned_by is None
	             and adjudicated_win is None
	             and not board.is_game_over() and board.halfmove_clock < 100
	             and move_count >= max_moves)

	examples = []
	total = len(history)
	for i, (state, policy, player, root_q) in enumerate(history):
		if winner is None:
			value = 0.0
		elif winner == player:
			value = 1.0
		else:
			value = -1.0
		if value_discount < 1.0:
			value *= value_discount ** (total - i)
		if truncated:
			# Unfinished, not drawn: 0.0 would be an actively wrong label, and
			# there is no outcome to anchor to, so the search value stands in.
			value = root_q
		elif winner is not None and search_value_weight > 0.0:
			# Decided games only.  On a draw the fixed point of this blend is
			# 0 anyway, so blending there would contract the value head toward
			# zero out of its own output and nothing else — see the docstring.
			value = ((1.0 - search_value_weight) * value
			         + search_value_weight * root_q)
		examples.append((state, policy, value))

	if tb_decided:
		# T, counted separately from R and A: this one is ground truth, and a
		# run where it supplies the decisive labels is in a different state
		# from one leaning on the material rule even at the same rate.
		result = ("1/2-1/2 T" if winner is None
		          else "1-0 T" if winner == chess.WHITE else "0-1 T")
	elif resigned_by is not None:
		# board.result() is "*" here — the game is decided but not over.
		# The trailing R keeps the resignation rate greppable in the log.
		result = "1-0 R" if winner == chess.WHITE else "0-1 R"
	elif adjudicated_win is not None:
		# Likewise for A: the two mechanisms are counted separately because a
		# run where only adjudication fires has a dead value head even though
		# its decisive-label rate looks healthy.
		result = "1-0 A" if winner == chess.WHITE else "0-1 A"
	elif board.halfmove_clock >= 100 and not board.is_game_over():
		# F, because board.result() still answers "*" here: claim_draw is off,
		# so the strict rules do not call a fifty-move game over and the
		# caller would count a real draw as a truncation.  It is a draw, and
		# 0.0 is the right label -- but a run drowning in fifty-move draws is
		# in a different state from one running out of moves, and one counter
		# for both hides which.
		result = "1/2-1/2 F"
	else:
		result = board.result()

	# Audit games are the calibration sample: they played to a real finish with
	# both early-stop mechanisms off, so for each side we know both the root Q
	# it would have resigned on and whether it actually went on to lose.  A
	# truncated game has no outcome, so it teaches nothing about false
	# positives and is dropped rather than counted as "not lost".
	audit = None
	if audit_game and not truncated:
		audit = []
		for side in (chess.WHITE, chess.BLACK):
			trigger = resign_trigger_q(
				[q for _s, _p, turn, q in history if turn == side],
				resign_plies)
			if trigger is None:
				continue
			by_result = winner is not None and winner != side
			# Mated, or materially crushed for long enough that the loop would
			# have scored it a loss had adjudication been on.  Both verdicts
			# travel so the calibration's choice of ground truth stays visible
			# in the log rather than being buried in this function.
			by_material = (would_adjudicate is not None
			               and would_adjudicate != side)
			# The scale slot is filled by the caller, which sees the whole
			# iteration's audit rows; one game cannot measure it.
			audit.append((trigger, by_result or by_material, by_result, None))
	return examples, result, audit


def resign_trigger_q(own_turn_qs, resign_plies):
	"""The most demanding threshold at which *own_turn_qs* still resigns.

	Resignation fires when the mover's own root Q stays at or below the
	threshold for *resign_plies* consecutive turns of its own.  A window of
	that many turns therefore fires at every threshold at or above the window's
	*largest* value, and the side needs only its best window:

	    trigger = min over windows of (max within window)

	The side would have resigned at threshold ``t`` exactly when
	``trigger <= t``, so one number per side summarises it at every candidate
	threshold at once — which is what makes calibration a scan rather than a
	replay.  Returns None when the side never had *resign_plies* turns: that is
	not a threshold of "never", it is no evidence either way, so those sides
	stay out of the calibration sample entirely.
	"""
	if resign_plies <= 0 or len(own_turn_qs) < resign_plies:
		return None
	return min(max(own_turn_qs[i:i + resign_plies])
	           for i in range(len(own_turn_qs) - resign_plies + 1))


def sample_scale(audit_rows):
	"""Median |trigger| over one iteration's audit sides, or None.

	The scale root Q is being measured against, taken from root Q itself
	rather than from the value head's output on held-out positions: it is the
	same quantity the threshold is applied to, and it is available exactly
	where the samples are, which ``mean|V|`` is not — that is computed after
	the calibration runs.  See :func:`calibrate_resign_threshold` for what it
	is for.
	"""
	qs = sorted(abs(row[0]) for row in audit_rows)
	if not qs:
		return None
	return qs[len(qs) // 2] or None


def scaled_triggers(samples, scale_now):
	"""Trigger values of *samples* restated on the current scale.

	A sample carries the scale that produced it, so ``trigger / scale`` is
	scale-free and ``* scale_now`` puts it back in the units the live
	threshold is expressed in.  Samples with no recorded scale (an older
	checkpoint's, or an iteration whose audit sides were all at zero) pass
	through unscaled, which is the previous behaviour.
	"""
	out = []
	for row in samples:
		scale = row[3] if len(row) > 3 else None
		out.append(row[0] * scale_now / scale
		           if scale and scale_now else row[0])
	return out


def false_positives_at(samples, threshold, by_result=False, scale_now=None):
	"""``(false_positives, would_resign)`` for *threshold* over *samples*.

	*samples* are ``(trigger_q, lost, lost_by_result, scale)`` rows from audit
	games.  A false positive is a side that would have resigned and then was
	not in fact lost — the only error resignation can make, and the reason the
	audit fraction exists.

	*lost* is the operative verdict: mated, outplayed, or materially crushed
	past the adjudication margin.  *by_result* switches to the strict reading,
	where only the game's own scoreline counts.  That reading is reported but
	never calibrated against: audit games run with adjudication off, so a net
	that cannot convert a won position before the 50-move rule draws almost all
	of them, and every correct resignation in those games reads as an error.

	*scale_now* restates every sample on the current scale first — see
	:func:`calibrate_resign_threshold`.  Without it a rate computed over a
	multi-iteration window is a rate over mixed units.
	"""
	idx = 2 if by_result else 1
	keys = scaled_triggers(samples, scale_now)
	fired = [row[idx] for row, key in zip(samples, keys) if key <= threshold]
	return sum(1 for lost in fired if not lost), len(fired)


def calibrate_resign_threshold(samples, current, fp_target=0.05,
                               min_resigns=10, floor=-0.99, ceiling=-0.10,
                               scale_now=None):
	"""Loosest resignation threshold whose false-positive rate holds.

	``--resign-threshold`` is an absolute bound on root Q, but root Q is a
	visit-weighted mean over the root's edges, so its scale rides on the value
	head's.  When that scale halves the fixed threshold becomes unreachable
	and resignation stops firing entirely — run8 went from 74% resignations to
	0% by iteration 15 and stayed there for 85 iterations, with the same
	dead-lost positions evaluating -0.95 before and -0.44 after.  A threshold
	re-derived each iteration from the audit games tracks the scale instead of
	being outlived by it.

	**The sample has to be restated on the current scale for that to work.**
	Re-deriving it every iteration is not enough on its own, because the
	window spans ``--resign-calib-window`` iterations and the scale moves
	within it: a -0.5 trigger recorded when the median |trigger| was 0.98
	means "not very lost", and the same -0.5 recorded at 0.50 means "dead
	lost".  Scored against one absolute candidate those two rows contradict
	each other, and the stale ones win because there are more of them.
	Measured over six iterations from pretrained_70M with the material label
	off, that pinned the threshold at -0.84 for the last three while the sides
	actually being produced sat at -0.66 to -0.47: the log reported an
	unchanged 3/16 (19%) false positives each time — the same sixteen stale
	sides — while real resignations fell 85% -> 15% -> 2%.  The loosening was
	available and safe the whole time (5 of 6 fresh sides at -0.47, one false
	positive) and the rate could not see it, because a row that does not fire
	enters neither the numerator nor the denominator.

	So each sample carries the scale that produced it and *scale_now* puts the
	whole window back into today's units before any rate is computed.

	Candidates are the observed trigger values, because those are the only
	points where the firing set changes.  The loosest candidate that keeps
	false positives at or under *fp_target* wins: stricter thresholds are
	always available and always cost decisive labels, so the binding
	constraint is the error rate, not the yield.

	*samples* are the ``(trigger_q, lost, lost_by_result, scale)`` rows audit
	games return.  Calibration reads *lost*, the operative verdict — see
	:func:`false_positives_at` for why the scoreline alone will not do.

	Returns ``(threshold, false_positives, would_resign)``, or None when no
	candidate clears both bars — too thin a sample to justify moving, or a
	value head whose confident losses are not confirmed often enough to resign
	on at any threshold.  Either way the caller keeps the threshold it has,
	and adjudication goes on supplying the decisive labels.
	"""
	if not samples:
		return None
	best = None
	for cand in sorted(set(scaled_triggers(samples, scale_now))):
		if not floor <= cand <= ceiling:
			continue
		fp, fired = false_positives_at(samples, cand, scale_now=scale_now)
		if fired < min_resigns:
			continue
		if fp / fired <= fp_target:
			# Ascending scan, so the last candidate that passes is the loosest.
			best = (cand, fp, fired)
	return best


def play_match(mcts_cur, mcts_ref, cur_is_white, max_moves=512,
               temp_moves=12, adjudicate_material=0.0, adjudicate_plies=0):
	"""Play one arena game between two nets and score it for *mcts_cur*.

	Returns ``(score, result_string, moves)`` where score is 1.0 / 0.5 / 0.0
	from the current net's point of view.

	No Dirichlet noise here — this is meant to measure playing strength, not to
	generate training data.  Instead the first *temp_moves* plies are sampled at
	temperature 1.0 so the pair doesn't replay one identical game every time;
	after that both sides play their argmax move.

	**Adjudication.**  Same rule as :func:`play_game`, and it matters more here
	than there.  Two nets a few iterations apart play near-symmetrical games:
	scoring everything short of checkmate as a draw put run8's arena at 195-198
	draws in 200 games, which caps the achievable score near 50% no matter how
	the match actually went.  Promotion needs 55% = 110 points, so it could not
	fire even in principle, and the generation counter sat at 0 for every run
	after run2.  A material margin decides the games the board never finishes,
	which is what turns the arena back into a measurement.

	No resignation here.  It reads the mover's root Q, and the two sides run
	*different* nets whose value heads are on different scales, so one absolute
	threshold would resign asymmetrically and score the scale gap rather than
	the playing strength.  Material is measured on the board and is common to
	both sides, so it cannot favour either net.
	"""
	board = chess.Board()
	moves = 0
	adjudicate_enabled = adjudicate_material > 0.0 and adjudicate_plies > 0
	adjudicated_win = None
	adj_leader = None
	adj_plies = 0

	# Same fifty-move agreement as play_game: a match game that runs to the
	# move limit scores as a draw either way, but it costs several hundred
	# plies of search first.
	while (not board.is_game_over() and board.halfmove_clock < 100
	       and moves < max_moves):
		cur_to_move = (board.turn == chess.WHITE) == cur_is_white
		searcher = mcts_cur if cur_to_move else mcts_ref
		temperature = 1.0 if moves < temp_moves else 0.0
		move, _ = searcher.search(board, temperature=temperature,
		                          add_noise=False)
		if move is None:
			break
		board.push(move)
		moves += 1

		if adjudicate_enabled:
			balance = material_balance(board)
			leader = None
			if abs(balance) >= adjudicate_material:
				leader = chess.WHITE if balance > 0 else chess.BLACK
			if leader is None:
				adj_leader, adj_plies = None, 0
			else:
				adj_plies = adj_plies + 1 if leader == adj_leader else 1
				adj_leader = leader
				if adj_plies >= adjudicate_plies:
					adjudicated_win = leader
					break

	if adjudicated_win is not None:
		white_won = adjudicated_win == chess.WHITE
		score = 1.0 if white_won == cur_is_white else 0.0
		# board.result() is "*" here — decided, but not over.  The trailing A
		# keeps the adjudication rate greppable and separates these from the
		# games that were actually mated on the board.
		result = "1-0 A" if white_won else "0-1 A"
	elif board.is_checkmate():
		# The side to move has been mated, so the other side won.
		white_won = board.turn == chess.BLACK
		score = 1.0 if white_won == cur_is_white else 0.0
		result = board.result()
	else:
		# Draw, or truncated at max_moves — scored as a draw either way.
		score = 0.5
		result = board.result()
	return score, result, moves


def fp16_state_dict(model):
	"""``model``'s state_dict on the host, floats narrowed to fp16.

	Halves the bytes written per iteration (23 MB vs 46 MB) and matches the
	dtype the workers run inference in, so no cast happens on load.
	"""
	out = {}
	for k, v in model.state_dict().items():
		v = v.detach().cpu()
		out[k] = v.half() if v.is_floating_point() else v
	return out


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(rank, task_q, result_q, cfg):
	"""Persistent self-play worker.  Runs until it receives ("stop", None)."""
	# Ctrl-C is the parent's business: it flips its interrupt flag, finishes
	# collecting in-flight games, checkpoints, and only then tells us to stop.
	# Without this the whole pool would die on the terminal's SIGINT and the
	# parent would see "all workers died" instead of a clean shutdown.
	signal.signal(signal.SIGINT, signal.SIG_IGN)

	try:
		perf.configure(num_threads=1)
		# Distinct RNG streams per worker — otherwise every worker draws the
		# same Dirichlet noise and temperature samples and plays near-identical
		# games, which would quietly destroy self-play diversity.
		seed = (cfg["seed"] + rank * 7919) % (2 ** 31)
		np.random.seed(seed)
		import random as _random
		_random.seed(seed)
		torch.manual_seed(seed)

		device = torch.device(cfg["device"])
		model = ChessNet(num_res_blocks=cfg["res_blocks"],
		                 num_filters=cfg["filters"])
		if cfg.get("weights"):
			model.load_state_dict(
				torch.load(cfg["weights"], map_location="cpu", weights_only=True),
			)
		model = perf.to_inference(model, device, half=cfg["half"])
		mcts = MCTS(model, device, num_simulations=cfg["simulations"],
		            batch_size=cfg["mcts_batch"],
		            fpu_reduction=cfg["fpu_reduction"],
		            dirichlet_alpha=cfg["dirichlet_alpha"],
		            dirichlet_eps=cfg["dirichlet_eps"])

		# Arena opponent.  Built up front rather than on first use so the cost
		# (a second CUDA-resident copy of the net) is paid during pool start-up
		# and never shows up as a stall in the middle of an iteration.
		ref_model = ChessNet(num_res_blocks=cfg["res_blocks"],
		                     num_filters=cfg["filters"])
		ref_model = perf.to_inference(ref_model, device, half=cfg["half"])
		eval_sims = cfg["eval_sims"]
		mcts_eval = MCTS(model, device, num_simulations=eval_sims,
		                 batch_size=cfg["mcts_batch"],
		                 fpu_reduction=cfg["fpu_reduction"])
		mcts_ref = MCTS(ref_model, device, num_simulations=eval_sims,
		                batch_size=cfg["mcts_batch"],
		                fpu_reduction=cfg["fpu_reduction"])
		# One handle per worker: the tables are memory-mapped, so eight workers
		# on one machine share the page cache rather than eight copies of it.
		# A path that cannot be opened disables the probe instead of killing
		# the pool -- the run is then the same run without it, which is a
		# reading, where a dead pool is nothing.
		tablebase = None
		if cfg.get("syzygy_path"):
			try:
				tablebase = chess.syzygy.open_tablebase(cfg["syzygy_path"])
			except Exception:
				tablebase = None
		result_q.put(("ready", rank, None, None, None, None))
	except Exception:
		result_q.put(("error", -1, traceback.format_exc(), None, None, None))
		return

	while True:
		try:
			kind, payload = task_q.get()
		except (EOFError, OSError):
			return

		if kind == "stop":
			return

		if kind == "weights":
			try:
				state = torch.load(payload, map_location=device, weights_only=True)
				model.load_state_dict(state)
				model.eval()
			except Exception:
				result_q.put(("error", -1, traceback.format_exc(),
				              None, None, None))
			continue

		if kind == "ref_weights":
			try:
				state = torch.load(payload, map_location=device,
				                   weights_only=True)
				ref_model.load_state_dict(state)
				ref_model.eval()
			except Exception:
				result_q.put(("error", -1, traceback.format_exc(),
				              None, None, None))
			continue

		if kind == "match":
			game_id, cur_is_white = payload
			t0 = time.perf_counter()
			try:
				# The arena's adjudication is deliberately a *separate* key.
				# In play_game the rule is a label source and a heuristic
				# label is a biased target; here it is a scoring rule, and
				# without it two near-identical nets return 195-198 draws in
				# 200 games and promotion cannot fire however the match went.
				score, result, moves = play_match(
					mcts_eval, mcts_ref, cur_is_white,
					max_moves=cfg["max_moves"],
					adjudicate_material=cfg.get(
						"arena_adjudicate_material",
						cfg.get("adjudicate_material", 0.0)),
					adjudicate_plies=cfg.get(
						"arena_adjudicate_plies",
						cfg.get("adjudicate_plies", 0)),
				)
				result_q.put(("match", game_id, score, result, moves,
				              time.perf_counter() - t0))
			except Exception:
				result_q.put(("error", game_id, traceback.format_exc(),
				              None, None, None))
			continue

		if kind == "play":
			# The threshold rides on the task rather than being broadcast.
			# Broadcasting means putting one message per worker on a queue they
			# all pull from, which guarantees the *count* and nothing else: a
			# worker that finishes early can take two and leave another with
			# none, and that one then plays a whole iteration at a stale
			# threshold.  Per-task delivery has no such gap, and it costs a
			# float per game.
			game_id, resign_threshold = payload
			t0 = time.perf_counter()
			try:
				examples, result, audit = play_game(
					mcts,
					max_moves=cfg["max_moves"],
					value_discount=cfg["value_discount"],
					resign_threshold=resign_threshold,
					resign_plies=cfg.get("resign_plies", 2),
					resign_disable_frac=cfg.get("resign_disable_frac", 0.1),
					search_value_weight=cfg.get("search_value_weight", 0.0),
					adjudicate_material=cfg.get("adjudicate_material", 0.0),
					adjudicate_plies=cfg.get("adjudicate_plies", 0),
					adjudicate_label=cfg.get("adjudicate_label", True),
					tablebase=tablebase,
					tablebase_pieces=cfg.get("tablebase_pieces", 5),
				)
				result_q.put(("game", game_id, examples, result,
				              len(examples), time.perf_counter() - t0,
				              audit))
			except Exception:
				result_q.put(("error", game_id, traceback.format_exc(),
				              None, None, None))


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

class SelfPlayPool:
	"""Persistent pool of self-play worker processes.

	    with SelfPlayPool(4, cfg, weights_path) as pool:
	        for iteration in ...:
	            pool.set_weights(model)
	            for examples, result, moves, secs in pool.play(50):
	                ...
	"""

	def __init__(self, num_workers, cfg, weights_path, ref_weights_path=None):
		self.num_workers = max(1, int(num_workers))
		self.cfg = dict(cfg)
		self.weights_path = weights_path
		self.ref_weights_path = (
			ref_weights_path or weights_path + ".ref"
		)
		self._ctx = mp.get_context("spawn")
		self._task_q = None
		self._result_q = None
		self._procs = []

	# -- lifecycle ----------------------------------------------------------

	def start(self):
		self._task_q = self._ctx.Queue()
		self._result_q = self._ctx.Queue()
		for rank in range(self.num_workers):
			p = self._ctx.Process(
				target=_worker,
				args=(rank, self._task_q, self._result_q, self.cfg),
				daemon=True,
			)
			p.start()
			self._procs.append(p)

		# Block until every worker has its model on the GPU, so the first
		# iteration's timing isn't polluted by CUDA context creation.
		ready = 0
		while ready < self.num_workers:
			msg = self._result_q.get()
			if msg[0] == "ready":
				ready += 1
			else:
				self.close()
				raise RuntimeError(f"self-play worker failed to start:\n{msg[2]}")

	def close(self):
		if self._task_q is not None:
			for _ in self._procs:
				try:
					self._task_q.put(("stop", None))
				except Exception:
					pass
		for p in self._procs:
			p.join(timeout=10)
			if p.is_alive():
				p.terminate()
		self._procs = []

	def __enter__(self):
		self.start()
		return self

	def __exit__(self, *_exc):
		self.close()

	# -- work ---------------------------------------------------------------

	def set_weights(self, model):
		"""Publish *model*'s weights to every worker as fp16.

		Written once to a temp file and renamed, so a worker can never read a
		half-written checkpoint.
		"""
		tmp = self.weights_path + ".tmp"
		torch.save(fp16_state_dict(model), tmp)
		os.replace(tmp, self.weights_path)
		for _ in self._procs:
			self._task_q.put(("weights", self.weights_path))

	def set_ref_weights(self, model):
		"""Publish *model*'s weights as the arena opponent for every worker."""
		tmp = self.ref_weights_path + ".tmp"
		torch.save(fp16_state_dict(model), tmp)
		os.replace(tmp, self.ref_weights_path)
		for _ in self._procs:
			self._task_q.put(("ref_weights", self.ref_weights_path))

	def publish_ref_from_file(self, path):
		"""Point every worker at an existing fp16 reference file."""
		for _ in self._procs:
			self._task_q.put(("ref_weights", path))

	def match(self, num_games):
		"""Play *num_games* current-vs-reference games.

		Colours alternate so a net that is only good with white can't inflate
		its score.  Yields ``(score, result, moves, secs)`` in completion order,
		score being 1/0.5/0 from the current net's point of view.
		"""
		pending = [(i, i % 2 == 0) for i in range(num_games)]
		in_flight = 0

		def _dispatch(n):
			nonlocal in_flight
			for _ in range(n):
				if not pending:
					return
				self._task_q.put(("match", pending.pop(0)))
				in_flight += 1

		_dispatch(2 * self.num_workers)

		while in_flight > 0:
			try:
				msg = self._result_q.get(timeout=1.0)
			except queue.Empty:
				if not any(p.is_alive() for p in self._procs):
					raise RuntimeError("all self-play workers died")
				continue

			in_flight -= 1
			if msg[0] == "error":
				raise RuntimeError(f"arena worker error:\n{msg[2]}")
			_kind, _gid, score, result, moves, secs = msg
			yield score, result, moves, secs
			_dispatch(1)

	def play(self, num_games, stop_early=None, resign_threshold=None):
		"""Dispatch *num_games* and yield ``(examples, result, moves, secs)``.

		Results arrive in completion order, not submission order.  When
		*stop_early* is given and returns True, remaining undispatched games are
		dropped; games already in flight are still collected so no work is
		wasted and the queues are left clean for the next iteration.

		Yields ``(examples, result, moves, secs, audit)``.  *audit* is None for
		ordinary games and a list of ``(trigger_q, lost, lost_by_result)``
		triples for the ``--resign-disable-frac`` games that played to a
		finish — the sample :func:`calibrate_resign_threshold` needs.

		*resign_threshold* overrides the configured one for this batch of
		games.  It travels with each task instead of being broadcast, so a
		threshold re-derived between iterations reaches every game rather than
		whichever workers happened to pick the broadcast up.
		"""
		if resign_threshold is None:
			resign_threshold = self.cfg.get("resign_threshold", 0.0)
		pending = [(i, resign_threshold) for i in range(num_games)]
		# Prime each worker with a couple of games, then top up as results land.
		# Keeping the queue shallow is what makes stop_early cheap: at most
		# `2 * num_workers` games are committed at any moment.
		in_flight = 0
		def _dispatch(n):
			nonlocal in_flight
			for _ in range(n):
				if not pending:
					return
				self._task_q.put(("play", pending.pop(0)))
				in_flight += 1

		_dispatch(2 * self.num_workers)

		while in_flight > 0:
			try:
				msg = self._result_q.get(timeout=1.0)
			except queue.Empty:
				if not any(p.is_alive() for p in self._procs):
					raise RuntimeError("all self-play workers died")
				continue

			in_flight -= 1
			if msg[0] == "error":
				raise RuntimeError(f"self-play worker error:\n{msg[2]}")
			_kind, _gid, examples, result, moves, secs, audit = msg
			yield examples, result, moves, secs, audit

			if stop_early is not None and stop_early():
				pending.clear()
			else:
				_dispatch(1)
