import math
import random
import numpy as np
import torch
import chess

from board_utils import encode_board, get_legal_move_indices, NUM_MOVES


class MCTSNode:
	"""Single node in the MCTS search tree."""

	__slots__ = ["visit_count", "total_value", "prior", "children"]

	def __init__(self, prior=0.0):
		self.visit_count = 0
		self.total_value = 0.0
		self.prior = prior
		self.children = {}  # chess.Move -> MCTSNode

	@property
	def value(self):
		if self.visit_count == 0:
			return 0.0
		return self.total_value / self.visit_count


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
		tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		policy_logits, value = self.model(tensor)
		policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
		return policy, value.item()

	# ------------------------------------------------------------------
	# Tree operations
	# ------------------------------------------------------------------

	def _expand(self, node, board, policy, legal_moves, indices, add_noise=False):
		"""Create children for *node* using *policy* restricted to legal moves."""
		priors = policy[indices]
		prior_sum = priors.sum()
		if prior_sum > 0:
			priors = priors / prior_sum
		else:
			priors = np.ones(len(legal_moves), dtype=np.float32) / len(legal_moves)

		if add_noise and len(legal_moves) > 0:
			noise = np.random.dirichlet([0.3] * len(legal_moves))
			priors = 0.75 * priors + 0.25 * noise

		for i, move in enumerate(legal_moves):
			node.children[move] = MCTSNode(prior=float(priors[i]))

	def _select_child(self, node):
		"""Pick the child that maximises Q + U (PUCT)."""
		total_visits = sum(c.visit_count for c in node.children.values())
		sqrt_total = math.sqrt(total_visits + 1)

		best_score = float("-inf")
		best_move = None
		best_child = None

		for move, child in node.children.items():
			q = -child.value  # negate: stored from child's perspective
			u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
			score = q + u
			if score > best_score:
				best_score = score
				best_move = move
				best_child = child

		return best_move, best_child

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

		self._expand(root, board, policy, legal_moves, indices, add_noise=add_noise)

		# Build move -> policy-index lookup once
		move_to_pidx = {m: indices[i] for i, m in enumerate(legal_moves)}

		# ----- simulations -----
		for _ in range(self.num_simulations):
			node = root
			scratch = board.copy()
			path = [node]

			# Selection — walk existing tree
			while node.children and not scratch.is_game_over():
				move, child = self._select_child(node)
				scratch.push(move)
				node = child
				path.append(node)

			# Leaf evaluation
			if scratch.is_game_over():
				value = -1.0 if scratch.is_checkmate() else 0.0
			else:
				leaf_policy, value = self.evaluate(scratch)
				leaf_legal, leaf_idx = get_legal_move_indices(scratch)
				self._expand(node, scratch, leaf_policy, leaf_legal, leaf_idx)

			# Backpropagation (flip sign at each level)
			for n in reversed(path):
				n.visit_count += 1
				n.total_value += value
				value = -value

		# ----- build policy target from visit counts -----
		policy_target = np.zeros(NUM_MOVES, dtype=np.float32)
		total_visits = sum(c.visit_count for c in root.children.values())
		if total_visits == 0:
			for m in legal_moves:
				policy_target[move_to_pidx[m]] = 1.0 / len(legal_moves)
			return random.choice(legal_moves), policy_target

		for move, child in root.children.items():
			policy_target[move_to_pidx[move]] = child.visit_count / total_visits

		# ----- select move -----
		if temperature <= 0.01:
			best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
		else:
			visits = np.array(
				[child.visit_count for child in root.children.values()],
				dtype=np.float64,
			)
			visits = visits ** (1.0 / temperature)
			probs = visits / visits.sum()
			moves_list = list(root.children.keys())
			best_move = moves_list[np.random.choice(len(moves_list), p=probs)]

		return best_move, policy_target
