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
    ("play",        game_id)           play one self-play game
    ("match",       (game_id, white))  current vs reference; *white* is True
                                       when the current net has white
    ("stop",        None)              exit

Replies (worker -> parent on ``result_q``):
    ("ready", rank,    None,     None,   None,  None)
    ("game",  game_id, examples, result, moves, seconds)
    ("match", game_id, score,    result, moves, seconds)
    ("error", game_id, traceback, None,  None,  None)
"""

import os
import queue
import signal
import time
import traceback

import chess
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

def play_game(mcts, max_moves=512, value_discount=1.0, temp_moves=30,
              temp_high=1.0, temp_low=0.1, resign_threshold=0.0,
              resign_plies=2, resign_disable_frac=0.1,
              search_value_weight=0.0):
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

	**Value targets.**  A pure game-outcome label is the same number for every
	position in a game, so in a corpus that is ~57% draws the constant 0.0 is
	genuinely the loss-minimising prediction and the value head converges to
	it.  *search_value_weight* blends in each position's own root Q, which
	differs from ply to ply, so two drawn games stop sharing one label and the
	constant stops being optimal.

	The blend is self-referential — root Q comes from the value head being
	trained — so it only helps once that head already carries signal.  Run it
	at 0.0 from a random initialisation and it reinforces the net's own noise;
	the intended use is after supervised pre-training (see src/pretrain.py).

	Games cut off at *max_moves* are the one case that takes the search value
	outright.  Those are not draws — they are unfinished — and labelling
	several hundred of their positions 0.0 is the single largest source of
	value-label noise left once resignation is on.
	"""
	board = chess.Board()
	history = []  # (encoded_state, policy, turn, root_q)

	resign_enabled = (resign_threshold < 0.0
	                  and np.random.random() >= resign_disable_frac)
	# Counted per colour: the root Q alternates POV every ply, so a single
	# counter would trip on two *different* sides each thinking they are lost.
	bad_turns = {chess.WHITE: 0, chess.BLACK: 0}
	resigned_by = None

	move_count = 0
	while not board.is_game_over() and move_count < max_moves:
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

	if resigned_by is not None:
		winner = not resigned_by
	elif board.is_checkmate():
		winner = not board.turn  # side to move is mated
	else:
		winner = None  # draw (or truncated at max_moves)

	# A game that ran out of moves is unfinished, not drawn.  Its outcome
	# label carries no information at all, so the search value replaces it
	# rather than being blended with it.
	truncated = (resigned_by is None and not board.is_game_over()
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
			value = root_q
		elif search_value_weight > 0.0:
			value = ((1.0 - search_value_weight) * value
			         + search_value_weight * root_q)
		examples.append((state, policy, value))

	if resigned_by is not None:
		# board.result() is "*" here — the game is decided but not over.
		# The trailing R keeps the resignation rate greppable in the log.
		result = "1-0 R" if winner == chess.WHITE else "0-1 R"
	else:
		result = board.result()
	return examples, result


def play_match(mcts_cur, mcts_ref, cur_is_white, max_moves=512,
               temp_moves=12):
	"""Play one arena game between two nets and score it for *mcts_cur*.

	Returns ``(score, result_string, moves)`` where score is 1.0 / 0.5 / 0.0
	from the current net's point of view.

	No Dirichlet noise here — this is meant to measure playing strength, not to
	generate training data.  Instead the first *temp_moves* plies are sampled at
	temperature 1.0 so the pair doesn't replay one identical game every time;
	after that both sides play their argmax move.
	"""
	board = chess.Board()
	moves = 0
	while not board.is_game_over() and moves < max_moves:
		cur_to_move = (board.turn == chess.WHITE) == cur_is_white
		searcher = mcts_cur if cur_to_move else mcts_ref
		temperature = 1.0 if moves < temp_moves else 0.0
		move, _ = searcher.search(board, temperature=temperature,
		                          add_noise=False)
		if move is None:
			break
		board.push(move)
		moves += 1

	if board.is_checkmate():
		# The side to move has been mated, so the other side won.
		white_won = board.turn == chess.BLACK
		score = 1.0 if white_won == cur_is_white else 0.0
	else:
		# Draw, or truncated at max_moves — scored as a draw either way.
		score = 0.5
	return score, board.result(), moves


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
				score, result, moves = play_match(
					mcts_eval, mcts_ref, cur_is_white,
					max_moves=cfg["max_moves"],
				)
				result_q.put(("match", game_id, score, result, moves,
				              time.perf_counter() - t0))
			except Exception:
				result_q.put(("error", game_id, traceback.format_exc(),
				              None, None, None))
			continue

		if kind == "play":
			game_id = payload
			t0 = time.perf_counter()
			try:
				examples, result = play_game(
					mcts,
					max_moves=cfg["max_moves"],
					value_discount=cfg["value_discount"],
					resign_threshold=cfg.get("resign_threshold", 0.0),
					resign_plies=cfg.get("resign_plies", 2),
					resign_disable_frac=cfg.get("resign_disable_frac", 0.1),
					search_value_weight=cfg.get("search_value_weight", 0.0),
				)
				result_q.put(("game", game_id, examples, result,
				              len(examples), time.perf_counter() - t0))
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

	def play(self, num_games, stop_early=None):
		"""Dispatch *num_games* and yield ``(examples, result, moves, secs)``.

		Results arrive in completion order, not submission order.  When
		*stop_early* is given and returns True, remaining undispatched games are
		dropped; games already in flight are still collected so no work is
		wasted and the queues are left clean for the next iteration.
		"""
		pending = list(range(num_games))
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
			_kind, _gid, examples, result, moves, secs = msg
			yield examples, result, moves, secs

			if stop_early is not None and stop_early():
				pending.clear()
			else:
				_dispatch(1)
