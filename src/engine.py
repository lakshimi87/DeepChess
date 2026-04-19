"""Game engine — selects between the neural MCTS engine and a classical
minimax fallback depending on whether a trained checkpoint exists."""

import os
import chess
import torch

from .model import ChessNet
from .mcts import MCTS
from .paths import CHECKPOINTS_DIR

# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------
# Neural engine: number of MCTS simulations
NEURAL_SIMULATIONS = {"easy": 100, "normal": 400, "hard": 800}
# Classical engine: minimax search depth
CLASSICAL_DEPTH = {"easy": 2, "normal": 3, "hard": 4}


def get_device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


# ===================================================================
# Public wrapper
# ===================================================================

class Engine:
	"""Unified interface: loads NN+MCTS when a checkpoint is found,
	otherwise falls back to the classical engine."""

	def __init__(self, difficulty="normal", checkpoint_dir=None):
		self.difficulty = difficulty
		self.device = get_device()

		if checkpoint_dir is None:
			checkpoint_dir = CHECKPOINTS_DIR
		checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
		if os.path.exists(checkpoint_path):
			self.mode = "neural"
			ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
			num_res = ckpt.get("num_res_blocks", 10)
			num_fil = ckpt.get("num_filters", 128)
			self.model = ChessNet(num_res_blocks=num_res, num_filters=num_fil)
			self.model.load_state_dict(ckpt["model_state_dict"])
			self.model.to(self.device)
			self.model.eval()
			sims = NEURAL_SIMULATIONS.get(difficulty, 400)
			self.mcts = MCTS(self.model, self.device, num_simulations=sims)
		else:
			self.mode = "classical"
			self._classical = _ClassicalEngine(
				depth=CLASSICAL_DEPTH.get(difficulty, 3)
			)

	def get_move(self, board):
		if board.is_game_over():
			return None
		if self.mode == "neural":
			move, _ = self.mcts.search(board, temperature=0.1)
			return move
		return self._classical.get_move(board)


# ===================================================================
# Classical minimax engine (fallback before training)
# ===================================================================

# Piece-square tables — visual layout (rank 8 at top, rank 1 at bottom).
# For white pieces at square sq use table[sq ^ 56]; for black use table[sq].

_PAWN_TABLE = [
	  0,   0,   0,   0,   0,   0,   0,   0,
	 50,  50,  50,  50,  50,  50,  50,  50,
	 10,  10,  20,  30,  30,  20,  10,  10,
	  5,   5,  10,  25,  25,  10,   5,   5,
	  0,   0,   0,  20,  20,   0,   0,   0,
	  5,  -5, -10,   0,   0, -10,  -5,   5,
	  5,  10,  10, -20, -20,  10,  10,   5,
	  0,   0,   0,   0,   0,   0,   0,   0,
]

_KNIGHT_TABLE = [
	-50, -40, -30, -30, -30, -30, -40, -50,
	-40, -20,   0,   0,   0,   0, -20, -40,
	-30,   0,  10,  15,  15,  10,   0, -30,
	-30,   5,  15,  20,  20,  15,   5, -30,
	-30,   0,  15,  20,  20,  15,   0, -30,
	-30,   5,  10,  15,  15,  10,   5, -30,
	-40, -20,   0,   5,   5,   0, -20, -40,
	-50, -40, -30, -30, -30, -30, -40, -50,
]

_BISHOP_TABLE = [
	-20, -10, -10, -10, -10, -10, -10, -20,
	-10,   0,   0,   0,   0,   0,   0, -10,
	-10,   0,   5,  10,  10,   5,   0, -10,
	-10,   5,   5,  10,  10,   5,   5, -10,
	-10,   0,  10,  10,  10,  10,   0, -10,
	-10,  10,  10,  10,  10,  10,  10, -10,
	-10,   5,   0,   0,   0,   0,   5, -10,
	-20, -10, -10, -10, -10, -10, -10, -20,
]

_ROOK_TABLE = [
	  0,   0,   0,   0,   0,   0,   0,   0,
	  5,  10,  10,  10,  10,  10,  10,   5,
	 -5,   0,   0,   0,   0,   0,   0,  -5,
	 -5,   0,   0,   0,   0,   0,   0,  -5,
	 -5,   0,   0,   0,   0,   0,   0,  -5,
	 -5,   0,   0,   0,   0,   0,   0,  -5,
	 -5,   0,   0,   0,   0,   0,   0,  -5,
	  0,   0,   0,   5,   5,   0,   0,   0,
]

_QUEEN_TABLE = [
	-20, -10, -10,  -5,  -5, -10, -10, -20,
	-10,   0,   0,   0,   0,   0,   0, -10,
	-10,   0,   5,   5,   5,   5,   0, -10,
	 -5,   0,   5,   5,   5,   5,   0,  -5,
	  0,   0,   5,   5,   5,   5,   0,  -5,
	-10,   5,   5,   5,   5,   5,   0, -10,
	-10,   0,   5,   0,   0,   0,   0, -10,
	-20, -10, -10,  -5,  -5, -10, -10, -20,
]

_KING_MID_TABLE = [
	-30, -40, -40, -50, -50, -40, -40, -30,
	-30, -40, -40, -50, -50, -40, -40, -30,
	-30, -40, -40, -50, -50, -40, -40, -30,
	-30, -40, -40, -50, -50, -40, -40, -30,
	-20, -30, -30, -40, -40, -30, -30, -20,
	-10, -20, -20, -20, -20, -20, -20, -10,
	 20,  20,   0,   0,   0,   0,  20,  20,
	 20,  30,  10,   0,   0,  10,  30,  20,
]

_KING_END_TABLE = [
	-50, -40, -30, -20, -20, -30, -40, -50,
	-30, -20, -10,   0,   0, -10, -20, -30,
	-30, -10,  20,  30,  30,  20, -10, -30,
	-30, -10,  30,  40,  40,  30, -10, -30,
	-30, -10,  30,  40,  40,  30, -10, -30,
	-30, -10,  20,  30,  30,  20, -10, -30,
	-30, -30,   0,   0,   0,   0, -30, -30,
	-50, -30, -30, -30, -30, -30, -30, -50,
]

_PIECE_VALUES = {
	chess.PAWN: 100,
	chess.KNIGHT: 320,
	chess.BISHOP: 330,
	chess.ROOK: 500,
	chess.QUEEN: 900,
	chess.KING: 20000,
}

_PST = {
	chess.PAWN: _PAWN_TABLE,
	chess.KNIGHT: _KNIGHT_TABLE,
	chess.BISHOP: _BISHOP_TABLE,
	chess.ROOK: _ROOK_TABLE,
	chess.QUEEN: _QUEEN_TABLE,
}


class _ClassicalEngine:
	"""Minimax + alpha-beta pruning + quiescence search + piece-square tables."""

	def __init__(self, depth=3):
		self.depth = depth

	# ------------------------------------------------------------------
	# Evaluation
	# ------------------------------------------------------------------

	@staticmethod
	def _is_endgame(board):
		return (
			len(board.pieces(chess.QUEEN, chess.WHITE))
			+ len(board.pieces(chess.QUEEN, chess.BLACK))
		) == 0

	@staticmethod
	def evaluate(board):
		if board.is_checkmate():
			return -99999 if board.turn == chess.WHITE else 99999
		if board.is_stalemate() or board.is_insufficient_material():
			return 0

		endgame = _ClassicalEngine._is_endgame(board)
		score = 0

		for sq in chess.SQUARES:
			piece = board.piece_at(sq)
			if piece is None:
				continue
			value = _PIECE_VALUES[piece.piece_type]
			if piece.piece_type == chess.KING:
				table = _KING_END_TABLE if endgame else _KING_MID_TABLE
			else:
				table = _PST[piece.piece_type]
			if piece.color == chess.WHITE:
				score += value + table[sq ^ 56]
			else:
				score -= value + table[sq]

		return score

	# ------------------------------------------------------------------
	# Move ordering (MVV-LVA)
	# ------------------------------------------------------------------

	@staticmethod
	def _order_moves(board):
		def _score(move):
			s = 0
			if board.is_capture(move):
				victim = board.piece_at(move.to_square)
				attacker = board.piece_at(move.from_square)
				if victim and attacker:
					s += 10 * _PIECE_VALUES[victim.piece_type] - _PIECE_VALUES[attacker.piece_type]
				elif attacker:  # en passant
					s += 10 * _PIECE_VALUES[chess.PAWN] - _PIECE_VALUES[attacker.piece_type]
			if move.promotion:
				s += _PIECE_VALUES.get(move.promotion, 0)
			return s

		moves = list(board.legal_moves)
		moves.sort(key=_score, reverse=True)
		return moves

	# ------------------------------------------------------------------
	# Quiescence search (captures only)
	# ------------------------------------------------------------------

	def _quiescence(self, board, alpha, beta, maximizing):
		stand_pat = self.evaluate(board)

		if maximizing:
			if stand_pat >= beta:
				return beta
			alpha = max(alpha, stand_pat)
			for move in self._order_moves(board):
				if not board.is_capture(move):
					continue
				board.push(move)
				val = self._quiescence(board, alpha, beta, False)
				board.pop()
				alpha = max(alpha, val)
				if alpha >= beta:
					return beta
			return alpha
		else:
			if stand_pat <= alpha:
				return alpha
			beta = min(beta, stand_pat)
			for move in self._order_moves(board):
				if not board.is_capture(move):
					continue
				board.push(move)
				val = self._quiescence(board, alpha, beta, True)
				board.pop()
				beta = min(beta, val)
				if beta <= alpha:
					return alpha
			return beta

	# ------------------------------------------------------------------
	# Alpha-beta minimax
	# ------------------------------------------------------------------

	def _minimax(self, board, depth, alpha, beta, maximizing):
		if board.is_game_over():
			return self.evaluate(board)
		if depth == 0:
			return self._quiescence(board, alpha, beta, maximizing)

		moves = self._order_moves(board)

		if maximizing:
			max_eval = float("-inf")
			for move in moves:
				board.push(move)
				val = self._minimax(board, depth - 1, alpha, beta, False)
				board.pop()
				max_eval = max(max_eval, val)
				alpha = max(alpha, val)
				if beta <= alpha:
					break
			return max_eval
		else:
			min_eval = float("inf")
			for move in moves:
				board.push(move)
				val = self._minimax(board, depth - 1, alpha, beta, True)
				board.pop()
				min_eval = min(min_eval, val)
				beta = min(beta, val)
				if beta <= alpha:
					break
			return min_eval

	# ------------------------------------------------------------------
	# Top-level search
	# ------------------------------------------------------------------

	def get_move(self, board):
		best_move = None
		maximizing = board.turn == chess.WHITE

		if maximizing:
			best_val = float("-inf")
			for move in self._order_moves(board):
				board.push(move)
				val = self._minimax(board, self.depth - 1, float("-inf"), float("inf"), False)
				board.pop()
				if val > best_val:
					best_val = val
					best_move = move
		else:
			best_val = float("inf")
			for move in self._order_moves(board):
				board.push(move)
				val = self._minimax(board, self.depth - 1, float("-inf"), float("inf"), True)
				board.pop()
				if val < best_val:
					best_val = val
					best_move = move

		return best_move
