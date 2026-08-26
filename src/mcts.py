import math
import random
import numpy as np
import torch
import chess

from . import _ext
from .board_utils import (
	encode_board, get_legal_move_indices, NUM_MOVES, NUM_PLANES,
)


def _ext_supports_fpu():
	"""Does the compiled extension's ``puct_select`` take an ``fpu_value``?

	The FPU argument was added after the first .so builds shipped, and a stale
	extension would silently reject the call.  Probing once here lets us fall
	back to the (slower) Python PUCT loop with a warning instead of crashing
	mid-search, and tells the user exactly what to do about it.
	"""
	if not _ext.AVAILABLE:
		return False
	try:
		_ext.impl.puct_select(
			np.ones(1, dtype=np.float32),
			np.zeros(1, dtype=np.int32),
			np.zeros(1, dtype=np.float32),
			1.5, 0.0,
		)
		return True
	except TypeError:
		import warnings
		warnings.warn(
			"src/_ext is built without first-play-urgency support — falling "
			"back to the pure-Python PUCT loop, which is several times slower. "
			"Run ./build_ext.sh to rebuild.",
			RuntimeWarning,
		)
		return False


_EXT_HAS_FPU = _ext_supports_fpu()


def _is_terminal_fast(board):
	"""Cheap terminal check used inside MCTS descent.

	Returns ``(terminal, value)`` where *value* is from the perspective of
	the side to move (-1 for being mated, 0 for any kind of draw).

	Skips the threefold/fivefold-repetition scan that ``Board.is_game_over``
	performs — that scan walks the whole move stack and shows up as ~30% of
	per-move CPU time in profiling.  ``play_game``'s outer loop still uses
	``board.is_game_over()`` for the *actual* game termination, so a missed
	repetition only affects rare leaves visited mid-search.
	"""
	if not any(board.generate_legal_moves()):
		return True, (-1.0 if board.is_check() else 0.0)
	if board.halfmove_clock >= 100:
		return True, 0.0
	if board.is_insufficient_material():
		return True, 0.0
	return False, None


class MCTSNode:
	"""Single node in the MCTS search tree.

	Children are stored as parallel arrays (moves, priors, visits, total_values)
	instead of a dict so the PUCT inner loop can be vectorised / dispatched to
	the C++ extension.
	"""

	__slots__ = ["moves", "priors", "visits", "total_values", "children_nodes",
	             "value"]

	def __init__(self):
		self.moves = []
		self.priors = None          # np.float32[N]
		self.visits = None          # np.int32[N]
		self.total_values = None    # np.float32[N]
		self.children_nodes = []    # list[MCTSNode | None]
		# The net's own value for this position, side-to-move POV.  Kept so
		# first-play urgency can be anchored to a clean evaluation instead of
		# the running edge mean, which carries in-flight virtual loss.
		self.value = 0.0

	@property
	def expanded(self):
		return self.moves is not None and len(self.moves) > 0


class MCTS:
	"""Monte Carlo Tree Search guided by a neural network.

	Uses PUCT for tree traversal and a dual-head NN (policy + value)
	for leaf evaluation and move priors.  When *batch_size* > 1 the search
	descends to *batch_size* leaves under virtual loss and evaluates them
	with a single NN forward pass — dramatically cutting per-simulation
	dispatch overhead on GPU / MPS.
	"""

	def __init__(self, model, device, num_simulations=800, c_puct=1.5,
	             batch_size=1, fpu_reduction=0.25, dirichlet_alpha=0.3,
	             dirichlet_eps=0.25):
		self.model = model
		self.device = device
		self.num_simulations = num_simulations
		self.c_puct = c_puct
		self.batch_size = max(1, int(batch_size))
		# First-play urgency: the Q value handed to a child that has never been
		# visited, expressed as (parent Q - fpu_reduction).  See _select_child.
		self.fpu_reduction = float(fpu_reduction)
		self.dirichlet_alpha = float(dirichlet_alpha)
		self.dirichlet_eps = float(dirichlet_eps)
		# Root evaluation from the most recent search(), side-to-move POV.
		# Kept as attributes rather than extra return values so the existing
		# search() call sites keep their two-value unpacking.  Read straight
		# after a search — self-play's resignation check is the only consumer.
		self.root_value = 0.0   # the net's own value for the root position
		self.root_q = 0.0       # visit-weighted mean over the root's edges
		self._use_ext = _ext.AVAILABLE and _EXT_HAS_FPU
		# Inputs are encoded as float32 on the host; cast to whatever dtype the
		# model expects (fp16 self-play clones run ~10% faster on MPS).
		try:
			p = next(model.parameters())
			self._model_dtype = p.dtype
			self._channels_last = p.dim() == 4 and p.is_contiguous(
				memory_format=torch.channels_last,
			)
		except StopIteration:
			self._model_dtype = torch.float32
			self._channels_last = False

		# Host staging buffer for eval batches.  Pinned memory turns the H2D
		# copy from a synchronous pageable transfer into an async DMA:
		# 256x20x8x8 measured 700us pageable vs 72us pinned+non_blocking.
		# Encoders write straight into this buffer, so np.stack disappears too.
		self._stage = None
		self._stage_np = None
		if device.type == "cuda":
			self._stage = torch.empty(
				(self.batch_size, NUM_PLANES, 8, 8),
				dtype=torch.float32, pin_memory=True,
			)
			self._stage_np = self._stage.numpy()

	# ------------------------------------------------------------------
	# Neural-network evaluation
	# ------------------------------------------------------------------

	def _upload(self, boards):
		"""Encode *boards* into device memory, returning the input tensor."""
		n = len(boards)
		if self._stage_np is not None and n <= self.batch_size:
			for i, b in enumerate(boards):
				self._stage_np[i] = encode_board(b)
			tensor = self._stage[:n].to(
				self.device, dtype=self._model_dtype, non_blocking=True,
			)
		else:
			states = np.stack([encode_board(b) for b in boards])
			tensor = torch.from_numpy(states).to(
				self.device, dtype=self._model_dtype,
			)
		if self._channels_last:
			tensor = tensor.contiguous(memory_format=torch.channels_last)
		return tensor

	@torch.inference_mode()
	def evaluate(self, board):
		"""Run the NN on *board* and return (policy, value).

		*policy* is a numpy array of shape (NUM_MOVES,) — softmax over
		all 4672 move slots.  *value* is a scalar from the current
		player's perspective.
		"""
		policies, values = self.evaluate_batch([board])
		return policies[0], float(values[0])

	@torch.inference_mode()
	def evaluate_batch(self, boards):
		"""Run the NN on a list of boards and return (policies, values).

		*policies* is a numpy array (B, NUM_MOVES); *values* is (B,).
		"""
		tensor = self._upload(boards)
		policy_logits, values = self.model(tensor)
		# .float() before .cpu() so downstream numpy stays fp32 even when the
		# model runs in fp16.  Both heads are fetched in one sync point.
		policies = torch.softmax(policy_logits, dim=1).float()
		values_t = values.squeeze(-1).float()
		policies_np = policies.cpu().numpy()
		values_np = values_t.cpu().numpy()
		return policies_np, values_np

	# ------------------------------------------------------------------
	# Tree operations
	# ------------------------------------------------------------------

	def _expand(self, node, policy, legal_moves, indices, add_noise=False,
	            node_value=0.0):
		"""Populate *node* with legal children priors drawn from *policy*."""
		n = len(legal_moves)
		node.value = float(node_value)
		priors = policy[indices].astype(np.float32, copy=False)
		prior_sum = float(priors.sum())
		if prior_sum > 0:
			priors = priors / prior_sum
		else:
			priors = np.full(n, 1.0 / n, dtype=np.float32)

		if add_noise and n > 0 and self.dirichlet_eps > 0.0:
			noise = np.random.dirichlet(
				[self.dirichlet_alpha] * n).astype(np.float32)
			eps = self.dirichlet_eps
			priors = (1.0 - eps) * priors + eps * noise

		node.moves = list(legal_moves)
		node.priors = priors
		node.visits = np.zeros(n, dtype=np.int32)
		node.total_values = np.zeros(n, dtype=np.float32)
		node.children_nodes = [None] * n

	def _select_child(self, node):
		"""Pick the child that maximises Q + U (PUCT) and return its index.

		Unvisited children get ``parent_Q - fpu_reduction`` rather than 0.
		Handing them a flat 0 treats every unexplored move as an even game,
		which in a lost position looks better than any explored move and in a
		won position looks worse than all of them — either way PUCT keeps
		peeling off to fresh moves and the visit counts end up nearly uniform.
		Since the visit distribution *is* the policy training target, a flat
		search means a flat target, the net learns flat priors, and the whole
		self-play loop settles into a fixed point where nothing sharpens.
		"""
		visits = node.visits
		total_values = node.total_values
		total = int(visits.sum())
		# Anchor on the node's own NN value rather than the mean over its edges:
		# during a batched descent every traversed edge carries +1.0 of virtual
		# loss, so an edge mean taken mid-batch can sit near -1 and would drive
		# the FPU baseline far below anything real.
		fpu = node.value - self.fpu_reduction

		if self._use_ext:
			return _ext.impl.puct_select(
				node.priors, visits, total_values, self.c_puct, fpu,
			)
		priors = node.priors
		sqrt_total = math.sqrt(total + 1)

		best_score = -float("inf")
		best = 0
		for i in range(len(priors)):
			v = int(visits[i])
			q = fpu if v == 0 else -(float(total_values[i]) / v)
			u = self.c_puct * float(priors[i]) * sqrt_total / (1 + v)
			s = q + u
			if s > best_score:
				best_score = s
				best = i
		return best

	# ------------------------------------------------------------------
	# Main search
	# ------------------------------------------------------------------

	def search(self, board, temperature=1.0, add_noise=False):
		"""Run MCTS from *board* and return (best_move, policy_target).

		*policy_target* is a numpy array (NUM_MOVES,) with the visit-count
		distribution — used as the training target.
		"""
		self.root_value = self.root_q = 0.0

		if board.is_game_over():
			return None, np.zeros(NUM_MOVES, dtype=np.float32)

		root = MCTSNode()

		# Expand root (single NN call — only once per search).
		policy, root_value = self.evaluate(board)
		legal_moves, indices = get_legal_move_indices(board)
		# Seed both with the raw net value so the forced-move and zero-visit
		# fast paths below still leave a meaningful root evaluation behind.
		self.root_value = self.root_q = float(root_value)

		# Fast path: forced move
		if len(legal_moves) == 1:
			target = np.zeros(NUM_MOVES, dtype=np.float32)
			target[indices[0]] = 1.0
			return legal_moves[0], target

		self._expand(root, policy, legal_moves, indices, add_noise=add_noise,
		             node_value=root_value)
		root_indices = list(indices)

		# ----- simulations (batched with virtual loss when batch_size>1) -----
		sims_done = 0
		while sims_done < self.num_simulations:
			this_batch = min(self.batch_size, self.num_simulations - sims_done)

			# Phase 1 — descent: pick `this_batch` leaves.  Virtual loss
			# (+1 to visits, +1 to total_values) applied at every edge
			# traversed so that subsequent descents within this batch
			# are pushed toward different paths.
			paths = []
			leaf_nodes = []
			leaf_boards = []
			leaf_terminal_values = []  # None if non-terminal, else float

			for _ in range(this_batch):
				node = root
				# stack=False: 0.73us vs 12.3us for a full copy.  MCTS only
				# ever pushes onto the scratch board and its terminal test
				# (_is_terminal_fast) deliberately skips repetition detection,
				# which is the only thing the move stack would be needed for.
				scratch = board.copy(stack=False)
				path = []
				term, term_val = _is_terminal_fast(scratch)
				while not term and node.expanded:
					idx = self._select_child(node)
					node.visits[idx] += 1
					node.total_values[idx] += 1.0  # virtual loss
					scratch.push(node.moves[idx])
					path.append((node, idx))
					child = node.children_nodes[idx]
					if child is None:
						child = MCTSNode()
						node.children_nodes[idx] = child
					node = child
					term, term_val = _is_terminal_fast(scratch)

				paths.append(path)
				if term:
					leaf_nodes.append(None)
					leaf_boards.append(None)
					leaf_terminal_values.append(term_val)
				else:
					leaf_nodes.append(node)
					leaf_boards.append(scratch)
					leaf_terminal_values.append(None)

			# Phase 2 — batched NN evaluation for non-terminal leaves.
			nn_idx = [i for i, v in enumerate(leaf_terminal_values) if v is None]
			if nn_idx:
				boards_to_eval = [leaf_boards[i] for i in nn_idx]
				policies, values_np = self.evaluate_batch(boards_to_eval)

				for bidx, i in enumerate(nn_idx):
					leaf_node = leaf_nodes[i]
					# Expand the leaf (skip if an earlier batch entry already did).
					if not leaf_node.expanded:
						leaf_legal, leaf_idx = get_legal_move_indices(leaf_boards[i])
						if leaf_legal:
							self._expand(leaf_node, policies[bidx], leaf_legal,
							             leaf_idx,
							             node_value=float(values_np[bidx]))
					leaf_terminal_values[i] = float(values_np[bidx])

			# Phase 3 — backprop: remove virtual loss and apply real value.
			for path, value in zip(paths, leaf_terminal_values):
				for parent, idx in reversed(path):
					parent.total_values[idx] -= 1.0  # undo virtual loss
					parent.total_values[idx] += value
					# visits were already incremented during descent
					value = -value

			sims_done += this_batch

		# ----- root value after search -----
		# total_values accumulate from the *child's* POV (see _select_child), so
		# the root's own Q is the negated visit-weighted mean.  Virtual loss is
		# fully unwound by the time the loop above exits, so this is clean.
		total_visits = int(root.visits.sum())
		if total_visits > 0:
			self.root_q = -float(root.total_values.sum()) / total_visits

		# ----- build policy target from visit counts -----
		policy_target = np.zeros(NUM_MOVES, dtype=np.float32)
		if total_visits == 0:
			for m, pidx in zip(legal_moves, root_indices):
				policy_target[pidx] = 1.0 / len(legal_moves)
			return random.choice(legal_moves), policy_target

		for pidx, v in zip(root_indices, root.visits):
			policy_target[pidx] = float(v) / total_visits

		# ----- select move -----
		if temperature <= 0.01:
			best_idx = int(np.argmax(root.visits))
			return root.moves[best_idx], policy_target

		visits_f = root.visits.astype(np.float64) ** (1.0 / temperature)
		probs = visits_f / visits_f.sum()
		chosen = np.random.choice(len(root.moves), p=probs)
		return root.moves[chosen], policy_target
