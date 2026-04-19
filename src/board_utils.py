"""Board encoding (18x8x8) and move indexing (4672 moves).

A C++ acceleration module is loaded when available; otherwise a pure-Python
implementation is used.  See src/_ext/ for the native source.
"""

import chess
import numpy as np

from . import _ext

NUM_MOVES = 4672  # 64 squares x 73 move types per square

# Direction vectors for queen-like moves (8 directions)
DIRECTIONS = [
	(0, 1),    # N
	(1, 1),    # NE
	(1, 0),    # E
	(1, -1),   # SE
	(0, -1),   # S
	(-1, -1),  # SW
	(-1, 0),   # W
	(-1, 1),   # NW
]

# Knight move vectors (8 possible jumps)
KNIGHT_MOVES = [
	(1, 2), (2, 1), (2, -1), (1, -2),
	(-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# Precompute lookups for performance
_KNIGHT_MOVE_INDEX = {km: i for i, km in enumerate(KNIGHT_MOVES)}
_DIRECTION_INDEX = {d: i for i, d in enumerate(DIRECTIONS)}


def _move_to_index_py(move):
	"""Pure-Python move-to-index.  Kept as a fallback when the C++
	extension is unavailable.  See module docstring for the encoding scheme."""
	from_sq = move.from_square
	to_sq = move.to_square
	from_file = chess.square_file(from_sq)
	from_rank = chess.square_rank(from_sq)
	to_file = chess.square_file(to_sq)
	to_rank = chess.square_rank(to_sq)

	file_diff = to_file - from_file
	rank_diff = to_rank - from_rank

	if move.promotion is not None and move.promotion != chess.QUEEN:
		promo_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[move.promotion]
		dir_idx = file_diff + 1
		move_type = 64 + promo_idx * 3 + dir_idx
		return from_sq * 73 + move_type

	diff = (file_diff, rank_diff)
	if diff in _KNIGHT_MOVE_INDEX:
		move_type = 56 + _KNIGHT_MOVE_INDEX[diff]
		return from_sq * 73 + move_type

	step_file = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
	step_rank = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)
	dir_idx = _DIRECTION_INDEX[(step_file, step_rank)]
	distance = max(abs(file_diff), abs(rank_diff))
	move_type = dir_idx * 7 + (distance - 1)
	return from_sq * 73 + move_type


if _ext.AVAILABLE:
	_move_to_index_raw = _ext.impl.move_to_index
else:
	_move_to_index_raw = None


def move_to_index(move):
	"""Convert a chess.Move to a policy index (0-4671)."""
	if _move_to_index_raw is not None:
		promo = move.promotion if move.promotion is not None else 0
		return _move_to_index_raw(move.from_square, move.to_square, promo)
	return _move_to_index_py(move)


def mirror_move(move):
	"""Mirror a move vertically for encoding from black's perspective."""
	from_sq = move.from_square ^ 56
	to_sq = move.to_square ^ 56
	return chess.Move(from_sq, to_sq, promotion=move.promotion)


def get_legal_move_indices(board):
	"""Return (legal_moves, policy_indices) for *board*.

	Moves are mirrored when black is to move so the policy vector is always
	from the current player's perspective.
	"""
	turn = board.turn
	legal_moves = list(board.legal_moves)
	if _move_to_index_raw is not None:
		if turn == chess.BLACK:
			indices = [
				_move_to_index_raw(m.from_square ^ 56, m.to_square ^ 56,
				                   m.promotion if m.promotion is not None else 0)
				for m in legal_moves
			]
		else:
			indices = [
				_move_to_index_raw(m.from_square, m.to_square,
				                   m.promotion if m.promotion is not None else 0)
				for m in legal_moves
			]
		return legal_moves, indices

	indices = []
	for move in legal_moves:
		encoded = mirror_move(move) if turn == chess.BLACK else move
		indices.append(_move_to_index_py(encoded))
	return legal_moves, indices


def _encode_board_py(board):
	"""Pure-Python board encoder.  See encode_board for plane layout."""
	if board.turn == chess.BLACK:
		board = board.mirror()

	state = np.zeros((18, 8, 8), dtype=np.float32)

	piece_types = [
		chess.PAWN, chess.KNIGHT, chess.BISHOP,
		chess.ROOK, chess.QUEEN, chess.KING,
	]

	for i, pt in enumerate(piece_types):
		for sq in board.pieces(pt, chess.WHITE):
			state[i][chess.square_rank(sq)][chess.square_file(sq)] = 1.0

	for i, pt in enumerate(piece_types):
		for sq in board.pieces(pt, chess.BLACK):
			state[i + 6][chess.square_rank(sq)][chess.square_file(sq)] = 1.0

	if board.has_kingside_castling_rights(chess.WHITE):
		state[12][:, :] = 1.0
	if board.has_queenside_castling_rights(chess.WHITE):
		state[13][:, :] = 1.0
	if board.has_kingside_castling_rights(chess.BLACK):
		state[14][:, :] = 1.0
	if board.has_queenside_castling_rights(chess.BLACK):
		state[15][:, :] = 1.0

	if board.ep_square is not None:
		state[16][chess.square_rank(board.ep_square)][chess.square_file(board.ep_square)] = 1.0

	state[17][:, :] = 1.0

	return state


def encode_board(board):
	"""Encode a chess.Board as an 18x8x8 float32 tensor.

	The board is rendered from the current player's perspective (mirrored when
	black is to move) so the network always sees the moving side as "white".

	Plane layout:
	   0-5:  Current player's pieces (P, N, B, R, Q, K)
	   6-11: Opponent's pieces       (P, N, B, R, Q, K)
	  12:    Our kingside castling rights
	  13:    Our queenside castling rights
	  14:    Their kingside castling rights
	  15:    Their queenside castling rights
	  16:    En passant square
	  17:    Ones (current-player indicator)
	"""
	if _ext.AVAILABLE:
		turn = board.turn
		if turn == chess.BLACK:
			board = board.mirror()
		return _ext.impl.encode_board(
			int(board.occupied_co[chess.WHITE]),
			int(board.occupied_co[chess.BLACK]),
			int(board.pawns),
			int(board.knights),
			int(board.bishops),
			int(board.rooks),
			int(board.queens),
			int(board.kings),
			bool(board.has_kingside_castling_rights(chess.WHITE)),
			bool(board.has_queenside_castling_rights(chess.WHITE)),
			bool(board.has_kingside_castling_rights(chess.BLACK)),
			bool(board.has_queenside_castling_rights(chess.BLACK)),
			board.ep_square if board.ep_square is not None else -1,
		)
	return _encode_board_py(board)
