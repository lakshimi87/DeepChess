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
    ("weights", path)  reload fp16 weights from *path*
    ("play", game_id)  play one game, reply on ``result_q``
    ("stop", None)     exit

Replies (worker -> parent on ``result_q``):
    ("ready", rank,    None,     None,   None,  None)
    ("game",  game_id, examples, result, moves, seconds)
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
              temp_high=1.0, temp_low=0.1):
	"""Play one self-play game with *mcts* and return (examples, result).

	*mcts* is reused across games — :meth:`MCTS.search` builds a fresh root
	every call, so there is no state to reset, and reusing it keeps the pinned
	staging buffer alive instead of reallocating it once per game.
	"""
	board = chess.Board()
	history = []  # (encoded_state, policy, turn)

	move_count = 0
	while not board.is_game_over() and move_count < max_moves:
		temperature = temp_high if move_count < temp_moves else temp_low
		state = encode_board(board)
		move, policy = mcts.search(board, temperature=temperature, add_noise=True)
		if move is None:
			break
		history.append((state, policy, board.turn))
		board.push(move)
		move_count += 1

	if board.is_checkmate():
		winner = not board.turn  # side to move is mated
	else:
		winner = None  # draw (or truncated at max_moves)

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
			value *= value_discount ** (total - i)
		examples.append((state, policy, value))

	return examples, board.result()


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
		            batch_size=cfg["mcts_batch"])
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

		if kind == "play":
			game_id = payload
			t0 = time.perf_counter()
			try:
				examples, result = play_game(
					mcts,
					max_moves=cfg["max_moves"],
					value_discount=cfg["value_discount"],
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

	def __init__(self, num_workers, cfg, weights_path):
		self.num_workers = max(1, int(num_workers))
		self.cfg = dict(cfg)
		self.weights_path = weights_path
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
