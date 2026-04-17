import chess
import numpy as np

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


def move_to_index(move):
	"""Convert a chess.Move to a policy index (0-4671).

	Encoding (per source square, 73 move types):
	  0-55: queen-like moves (8 directions x 7 distances)
	  56-63: knight moves (8 jumps)
	  64-72: underpromotions (3 piece types x 3 directions)
	Queen promotions are encoded as regular queen-like moves.
	"""
	from_sq = move.from_square
	to_sq = move.to_square
	from_file = chess.square_file(from_sq)
	from_rank = chess.square_rank(from_sq)
	to_file = chess.square_file(to_sq)
	to_rank = chess.square_rank(to_sq)

	file_diff = to_file - from_file
	rank_diff = to_rank - from_rank

	# Underpromotion (knight, bishop, rook — queen is encoded as a regular move)
	if move.promotion is not None and move.promotion != chess.QUEEN:
		promo_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[move.promotion]
		dir_idx = file_diff + 1  # -1 -> 0, 0 -> 1, 1 -> 2
		move_type = 64 + promo_idx * 3 + dir_idx
		return from_sq * 73 + move_type

	# Knight move
	diff = (file_diff, rank_diff)
	if diff in _KNIGHT_MOVE_INDEX:
		move_type = 56 + _KNIGHT_MOVE_INDEX[diff]
		return from_sq * 73 + move_type

	# Queen-like move (sliding pieces, pawn pushes, king moves, castling, queen promotions)
	step_file = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
	step_rank = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)
	direction = (step_file, step_rank)
	dir_idx = _DIRECTION_INDEX[direction]
	distance = max(abs(file_diff), abs(rank_diff))
	move_type = dir_idx * 7 + (distance - 1)
	return from_sq * 73 + move_type


def mirror_move(move):
	"""Mirror a move vertically (flip ranks) for encoding from black's perspective."""
	from_sq = move.from_square ^ 56
	to_sq = move.to_square ^ 56
	return chess.Move(from_sq, to_sq, promotion=move.promotion)


def get_legal_move_indices(board):
	"""Get legal moves and their corresponding policy indices.

	Moves are mirrored when it is black's turn so the policy vector
	is always from the current player's perspective (as if white).

	Returns:
		legal_moves: list of chess.Move (original, unmapped moves)
		indices: list of int (policy indices, aligned with legal_moves)
	"""
	turn = board.turn
	legal_moves = list(board.legal_moves)
	indices = []
	for move in legal_moves:
		encoded = mirror_move(move) if turn == chess.BLACK else move
		indices.append(move_to_index(encoded))
	return legal_moves, indices


def encode_board(board):
	"""Encode board state as an 18x8x8 float32 tensor.

	The board is always encoded from the current player's perspective
	(mirrored so the current player appears as white).

	Planes:
	   0-5:  Current player's pieces (P, N, B, R, Q, K)
	   6-11: Opponent's pieces       (P, N, B, R, Q, K)
	  12:    Our kingside castling rights
	  13:    Our queenside castling rights
	  14:    Their kingside castling rights
	  15:    Their queenside castling rights
	  16:    En passant square
	  17:    Ones (current-player indicator)
	"""
	if board.turn == chess.BLACK:
		board = board.mirror()

	state = np.zeros((18, 8, 8), dtype=np.float32)

	piece_types = [
		chess.PAWN, chess.KNIGHT, chess.BISHOP,
		chess.ROOK, chess.QUEEN, chess.KING,
	]

	# Our pieces (white after mirroring)
	for i, pt in enumerate(piece_types):
		for sq in board.pieces(pt, chess.WHITE):
			state[i][chess.square_rank(sq)][chess.square_file(sq)] = 1.0

	# Their pieces (black after mirroring)
	for i, pt in enumerate(piece_types):
		for sq in board.pieces(pt, chess.BLACK):
			state[i + 6][chess.square_rank(sq)][chess.square_file(sq)] = 1.0

	# Castling rights (binary planes)
	if board.has_kingside_castling_rights(chess.WHITE):
		state[12][:, :] = 1.0
	if board.has_queenside_castling_rights(chess.WHITE):
		state[13][:, :] = 1.0
	if board.has_kingside_castling_rights(chess.BLACK):
		state[14][:, :] = 1.0
	if board.has_queenside_castling_rights(chess.BLACK):
		state[15][:, :] = 1.0

	# En passant
	if board.ep_square is not None:
		state[16][chess.square_rank(board.ep_square)][chess.square_file(board.ep_square)] = 1.0

	# Current player plane (always 1 after mirroring)
	state[17][:, :] = 1.0

	return state
