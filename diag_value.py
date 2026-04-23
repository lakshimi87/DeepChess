"""Clean diagnosis of value-head output.

Key question: why does validate_gt see +0.001 for every position when called
one at a time, but batched inference gives varied (often extreme) values?
"""
import os
import chess
import numpy as np
import torch

from src.board_utils import encode_board
from src.model import ChessNet
from src.paths import CHECKPOINTS_DIR
from src.engine import get_device

device = get_device()
ckpt_path = os.path.join(CHECKPOINTS_DIR, "latest.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
nr = ckpt.get("num_res_blocks", 10)
nf = ckpt.get("num_filters", 128)
model = ChessNet(num_res_blocks=nr, num_filters=nf)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

print(f"Checkpoint  : iter {ckpt.get('iteration','?')}  ({nr} blocks, {nf} filters)")
print(f"Device      : {device}\n")

positions = [
	("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "White up a queen"),
	("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1", "Black up a rook"),
	("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
	("rn1qkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "White up B+N"),
	("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", "Black up a queen"),
	("1k6/8/1K6/8/8/8/8/7R w - - 0 1", "Mate in 1 (R-h8)"),
	("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Italian opening"),
	("r3k2r/pppq1ppp/2n2n2/3pp3/3PP3/2N2N2/PPPQ1PPP/R3K2R w KQkq - 0 1", "Middle-game (rich)"),
]

states = np.stack([encode_board(chess.Board(f)) for f, _ in positions])
t_all = torch.from_numpy(states).to(device)

# --- 1. Single-position forwards (as validate_gt does) ----------
print("[A] Single-position forward (eval mode):")
vals_single = []
for fen, desc in positions:
	state = encode_board(chess.Board(fen))
	x = torch.from_numpy(state).unsqueeze(0).to(device)
	with torch.no_grad():
		_p, v = model(x)
	vals_single.append(v.item())
	print(f"  v={v.item():+.5f}  {desc}")

# --- 2. Batched forward -----------------------------------------
print("\n[B] Batched forward (all 8 together, eval mode):")
with torch.no_grad():
	_p, v = model(t_all)
vals_batch = v.detach().cpu().numpy().ravel()
for (fen, desc), val in zip(positions, vals_batch):
	print(f"  v={val:+.5f}  {desc}")

print("\n[A] vs [B] difference (should be ~0 in eval mode):")
for (fen, desc), a, b in zip(positions, vals_single, vals_batch):
	diff = a - b
	tag = "!!!" if abs(diff) > 1e-3 else "   "
	print(f"  {tag}  diff={diff:+.5f}  single={a:+.5f}  batch={b:+.5f}  {desc}")

# --- 3. Re-run single on CPU for comparison ---------------------
print("\n[C] Single forward on CPU (model copied to CPU):")
cpu_model = ChessNet(num_res_blocks=nr, num_filters=nf)
cpu_model.load_state_dict(ckpt["model_state_dict"])
cpu_model.eval()
for fen, desc in positions:
	state = encode_board(chess.Board(fen))
	x = torch.from_numpy(state).unsqueeze(0)
	with torch.no_grad():
		_p, v = cpu_model(x)
	print(f"  v={v.item():+.5f}  {desc}")

# --- 4. Batched on CPU ------------------------------------------
print("\n[D] Batched forward on CPU:")
with torch.no_grad():
	_p, v = cpu_model(torch.from_numpy(states))
for (fen, desc), val in zip(positions, v.detach().numpy().ravel()):
	print(f"  v={val:+.5f}  {desc}")

# --- 5. Minimum batch size needed for MPS to work ---------------
print("\n[E] MPS: try varying batch sizes (first position only):")
first_fen = positions[0][0]
for bs in [1, 2, 3, 4, 8, 16]:
	state = encode_board(chess.Board(first_fen))
	x = torch.from_numpy(state).unsqueeze(0).repeat(bs, 1, 1, 1).to(device)
	with torch.no_grad():
		_p, v = model(x)
	vals = v.detach().cpu().numpy().ravel()
	print(f"  bs={bs:2d}  first_v={vals[0]:+.5f}  last_v={vals[-1]:+.5f}")
