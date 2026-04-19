// DeepChess native acceleration.
//
// Implements the hot-path routines called during self-play and MCTS:
//   * encode_board    — build the 18x8x8 float32 input tensor
//   * move_to_index   — map (from, to, promotion) -> AlphaZero-style policy slot
//   * puct_select     — pick the argmax-PUCT child from parallel node arrays
//
// The goal is to eliminate per-node Python overhead in the MCTS inner loops;
// these three functions dominate the profile before batching.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Move encoding
// -----------------------------------------------------------------------------
//
// Layout mirrors src/board_utils.py _move_to_index_py.  Per source square (73
// slots):
//   0-55 : queen-like moves (8 directions x 7 distances)
//  56-63 : knight moves
//  64-72 : underpromotions (3 piece types x 3 directions)
// Queen promotions are encoded as queen-like moves.

// Square helpers (chess.Move uses 0..63 where square = rank*8 + file).
static inline int file_of(int sq) { return sq & 7; }
static inline int rank_of(int sq) { return sq >> 3; }

// Knight (file_diff, rank_diff) -> knight-index 0..7.
static int knight_index(int df, int dr) {
	// Must match KNIGHT_MOVES order in board_utils.py.
	static const int tbl[5][5] = {
		// df = -2, -1, 0, 1, 2   (indexed as df+2)
		// dr = -2
		{ -1, 4, -1, 3, -1 },
		// dr = -1
		{  5, -1, -1, -1, 2 },
		// dr =  0
		{ -1, -1, -1, -1, -1 },
		// dr =  1
		{  6, -1, -1, -1, 1 },
		// dr =  2
		{ -1, 7, -1, 0, -1 },
	};
	if (df < -2 || df > 2 || dr < -2 || dr > 2) return -1;
	return tbl[dr + 2][df + 2];
}

// Direction (step_file, step_rank) -> index matching DIRECTIONS in Python.
static int direction_index(int sf, int sr) {
	// (0, 1)=N, (1,1)=NE, (1,0)=E, (1,-1)=SE, (0,-1)=S,
	// (-1,-1)=SW, (-1,0)=W, (-1,1)=NW
	if (sf == 0 && sr == 1) return 0;
	if (sf == 1 && sr == 1) return 1;
	if (sf == 1 && sr == 0) return 2;
	if (sf == 1 && sr == -1) return 3;
	if (sf == 0 && sr == -1) return 4;
	if (sf == -1 && sr == -1) return 5;
	if (sf == -1 && sr == 0) return 6;
	if (sf == -1 && sr == 1) return 7;
	return -1;
}

// python-chess piece-type constants: PAWN=1..KING=6, QUEEN=5.
static int move_to_index(int from_sq, int to_sq, int promotion) {
	int from_file = file_of(from_sq), from_rank = rank_of(from_sq);
	int to_file = file_of(to_sq),   to_rank = rank_of(to_sq);

	int df = to_file - from_file;
	int dr = to_rank - from_rank;

	// Underpromotion (knight=2, bishop=3, rook=4; queen=5 falls through).
	if (promotion != 0 && promotion != 5) {
		int promo_idx;
		switch (promotion) {
			case 2: promo_idx = 0; break;  // knight
			case 3: promo_idx = 1; break;  // bishop
			case 4: promo_idx = 2; break;  // rook
			default: promo_idx = 0; break;
		}
		int dir_idx = df + 1;                    // -1, 0, 1 -> 0, 1, 2
		int move_type = 64 + promo_idx * 3 + dir_idx;
		return from_sq * 73 + move_type;
	}

	int kidx = knight_index(df, dr);
	if (kidx >= 0) {
		int move_type = 56 + kidx;
		return from_sq * 73 + move_type;
	}

	int sf = (df == 0) ? 0 : (df > 0 ? 1 : -1);
	int sr = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
	int dir_idx = direction_index(sf, sr);
	int distance = std::max(std::abs(df), std::abs(dr));
	int move_type = dir_idx * 7 + (distance - 1);
	return from_sq * 73 + move_type;
}

// -----------------------------------------------------------------------------
// Board encoding
// -----------------------------------------------------------------------------
//
// Caller passes the bitboards for each piece-type (the python-chess layout),
// already mirrored if it was black to move.  Castling rights and ep square
// refer to the post-mirror board.

static py::array_t<float> encode_board(
	uint64_t occ_w, uint64_t occ_b,
	uint64_t pawns, uint64_t knights, uint64_t bishops,
	uint64_t rooks, uint64_t queens, uint64_t kings,
	bool w_kside, bool w_qside, bool b_kside, bool b_qside,
	int ep_square
) {
	py::array_t<float> arr({18, 8, 8});
	auto buf = arr.mutable_unchecked<3>();

	// Zero-initialize (numpy allocated memory is not guaranteed to be zeroed).
	std::memset(arr.mutable_data(), 0, sizeof(float) * 18 * 8 * 8);

	const uint64_t type_bb[6] = { pawns, knights, bishops, rooks, queens, kings };

	// Planes 0-5: current player's pieces (white after mirroring).
	for (int t = 0; t < 6; ++t) {
		uint64_t bb = type_bb[t] & occ_w;
		while (bb) {
			int sq = __builtin_ctzll(bb);
			buf(t, rank_of(sq), file_of(sq)) = 1.0f;
			bb &= bb - 1;
		}
	}

	// Planes 6-11: opponent's pieces (black after mirroring).
	for (int t = 0; t < 6; ++t) {
		uint64_t bb = type_bb[t] & occ_b;
		while (bb) {
			int sq = __builtin_ctzll(bb);
			buf(t + 6, rank_of(sq), file_of(sq)) = 1.0f;
			bb &= bb - 1;
		}
	}

	// Castling rights (binary planes).
	auto fill_plane = [&](int p) {
		for (int r = 0; r < 8; ++r)
			for (int f = 0; f < 8; ++f)
				buf(p, r, f) = 1.0f;
	};
	if (w_kside) fill_plane(12);
	if (w_qside) fill_plane(13);
	if (b_kside) fill_plane(14);
	if (b_qside) fill_plane(15);

	// En passant square.
	if (ep_square >= 0 && ep_square < 64) {
		buf(16, rank_of(ep_square), file_of(ep_square)) = 1.0f;
	}

	// Current-player indicator plane (always 1 after mirroring).
	fill_plane(17);

	return arr;
}

// -----------------------------------------------------------------------------
// PUCT selection
// -----------------------------------------------------------------------------
//
// Given parallel arrays over the children of a node (priors, visits, total
// values) and the PUCT exploration constant, return the child index with
// maximum Q + U where
//   Q = -total_value / visits   (negated: stored from child's POV)
//   U = c_puct * prior * sqrt(sum_visits + 1) / (1 + visits)

static int puct_select(
	py::array_t<float, py::array::c_style | py::array::forcecast> priors,
	py::array_t<int32_t, py::array::c_style | py::array::forcecast> visits,
	py::array_t<float, py::array::c_style | py::array::forcecast> total_values,
	float c_puct
) {
	ssize_t n = priors.shape(0);
	const float*   p_arr = priors.data();
	const int32_t* v_arr = visits.data();
	const float*   t_arr = total_values.data();

	int32_t total_visits = 0;
	for (ssize_t i = 0; i < n; ++i) total_visits += v_arr[i];
	float sqrt_total = std::sqrt(static_cast<float>(total_visits) + 1.0f);

	int best = 0;
	float best_score = -std::numeric_limits<float>::infinity();
	for (ssize_t i = 0; i < n; ++i) {
		int32_t v = v_arr[i];
		float q = (v == 0) ? 0.0f : -(t_arr[i] / static_cast<float>(v));
		float u = c_puct * p_arr[i] * sqrt_total / (1.0f + static_cast<float>(v));
		float s = q + u;
		if (s > best_score) { best_score = s; best = static_cast<int>(i); }
	}
	return best;
}

// -----------------------------------------------------------------------------
// pybind11 module
// -----------------------------------------------------------------------------

PYBIND11_MODULE(chess_ext, m) {
	m.doc() = "DeepChess native acceleration (encode_board, move_to_index, "
	          "puct_select)";
	m.def("move_to_index", &move_to_index,
	      py::arg("from_square"), py::arg("to_square"), py::arg("promotion") = 0,
	      "Encode (from, to, promotion) as a policy slot in [0, 4672).");
	m.def("encode_board", &encode_board,
	      py::arg("occ_w"), py::arg("occ_b"),
	      py::arg("pawns"), py::arg("knights"), py::arg("bishops"),
	      py::arg("rooks"), py::arg("queens"), py::arg("kings"),
	      py::arg("w_kside"), py::arg("w_qside"),
	      py::arg("b_kside"), py::arg("b_qside"),
	      py::arg("ep_square"),
	      "Build the 18x8x8 float32 input tensor from the mirrored board's "
	      "bitboards.");
	m.def("puct_select", &puct_select,
	      py::arg("priors"), py::arg("visits"), py::arg("total_values"),
	      py::arg("c_puct"),
	      "Return the argmax-PUCT child index over parallel node arrays.");
}
