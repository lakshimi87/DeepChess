import math
import random
import numpy as np
import torch
import chess

from . import _ext
from .board_utils import encode_board, get_legal_move_indices, NUM_MOVES


class MCTSNode:
	"""Single node in the MCTS search tree.

	Children are stored as parallel arrays (moves, priors, visits, total_values)
	instead of a dict so the PUCT inner loop can be vectorised / dispatched to
	the C++ extension.
	"""

	__slots__ = ["moves", "priors", "visits", "total_values", "children_nodes"]

	def __init__(self):
		self.moves = []
		self.priors = None          # np.float32[N]
		self.visits = None          # np.int32[N]
		self.total_values = None    # np.float32[N]
		self.children_nodes = []    # list[MCTSNode | None]

	@property
	def expanded(self):
		return self.moves is not None and len(self.moves) > 0


class MCTS:
	"""Monte Carlo Tree Search guided by a neural network.

	Uses PUCT for tree traversal and a dual-head NN (policy + value)
	for leaf evaluation and move priors.
	"""

	def __init__(self, model, device, num_simulations=800, c_puct=1.5):
		self.model = model
		self.device = device
		self.num_simulations = num_simulations
		self.c_puct = c_puct
		self._use_ext = _ext.AVAILABLE

	# ------------------------------------------------------------------
	# Neural-network evaluation
	# ------------------------------------------------------------------

	@torch.no_grad()
	def evaluate(self, board):
		"""Run the NN on *board* and return (policy, value).

		*policy* is a numpy array of shape (NUM_MOVES,) — softmax over
		all 4672 move slots.  *value* is a scalar from the current
		player's perspective.
		"""
		state = encode_board(board)
		tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
		policy_logits, value = self.model(tensor)
		policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
		return policy, value.item()

	# ------------------------------------------------------------------
	# Tree operations
	# ------------------------------------------------------------------

	def _expand(self, node, policy, legal_moves, indices, add_noise=False):
		"""Populate *node* with legal children priors drawn from *policy*."""
		n = len(legal_moves)
		priors = policy[indices].astype(np.float32, copy=False)
		prior_sum = float(priors.sum())
		if prior_sum > 0:
			priors = priors / prior_sum
		else:
			priors = np.full(n, 1.0 / n, dtype=np.float32)

		if add_noise and n > 0:
			noise = np.random.dirichlet([0.3] * n).astype(np.float32)
			priors = 0.75 * priors + 0.25 * noise

		node.moves = list(legal_moves)
		node.priors = priors
		node.visits = np.zeros(n, dtype=np.int32)
		node.total_values = np.zeros(n, dtype=np.float32)
		node.children_nodes = [None] * n

	def _select_child(self, node):
		"""Pick the child that maximises Q + U (PUCT) and return its index."""
		if self._use_ext:
			return _ext.impl.puct_select(
				node.priors, node.visits, node.total_values, self.c_puct,
			)
		visits = node.visits
		total_values = node.total_values
		priors = node.priors
		total = int(visits.sum())
		sqrt_total = math.sqrt(total + 1)

		best_score = -float("inf")
		best = 0
		for i in range(len(priors)):
			v = int(visits[i])
			q = 0.0 if v == 0 else -(float(total_values[i]) / v)
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
		if board.is_game_over():
			return None, np.zeros(NUM_MOVES, dtype=np.float32)

		root = MCTSNode()

		# Expand root
		policy, _ = self.evaluate(board)
		legal_moves, indices = get_legal_move_indices(board)

		# Fast path: forced move
		if len(legal_moves) == 1:
			target = np.zeros(NUM_MOVES, dtype=np.float32)
			target[indices[0]] = 1.0
			return legal_moves[0], target

		self._expand(root, policy, legal_moves, indices, add_noise=add_noise)
		root_indices = list(indices)

		# ----- simulations -----
		for _ in range(self.num_simulations):
			node = root
			scratch = board.copy()
			path = []  # list[(parent_node, child_idx)]

			# Selection — walk existing tree
			while node.expanded and not scratch.is_game_over():
				idx = self._select_child(node)
				scratch.push(node.moves[idx])
				path.append((node, idx))
				child = node.children_nodes[idx]
				if child is None:
					child = MCTSNode()
					node.children_nodes[idx] = child
				node = child

			# Leaf evaluation
			if scratch.is_game_over():
				value = -1.0 if scratch.is_checkmate() else 0.0
			else:
				leaf_policy, value = self.evaluate(scratch)
				leaf_legal, leaf_idx = get_legal_move_indices(scratch)
				if leaf_legal:
					self._expand(node, leaf_policy, leaf_legal, leaf_idx)

			# Backpropagation (flip sign at each level)
			for parent, idx in reversed(path):
				parent.visits[idx] += 1
				parent.total_values[idx] += value
				value = -value

		# ----- build policy target from visit counts -----
		policy_target = np.zeros(NUM_MOVES, dtype=np.float32)
		total_visits = int(root.visits.sum())
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
