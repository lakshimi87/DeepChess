"""Trace activations layer-by-layer through the value head to pinpoint collapse."""
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

positions = [
	("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "White up Q"),
	("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", "Black up Q"),
	("r3k2r/pppq1ppp/2n2n2/3pp3/3PP3/2N2N2/PPPQ1PPP/R3K2R w KQkq - 0 1", "Middle-game"),
]

states = np.stack([encode_board(chess.Board(f)) for f, _ in positions])
t = torch.from_numpy(states).to(device)

# Hook every value-head operation
captured = {}

def hook_factory(name):
	def hook(mod, inp, out):
		captured[name] = out.detach().cpu().numpy()
	return hook

hooks = []
hooks.append(model.value_conv.register_forward_hook(hook_factory("value_conv")))
hooks.append(model.value_bn.register_forward_hook(hook_factory("value_bn")))
hooks.append(model.value_fc1.register_forward_hook(hook_factory("value_fc1")))
hooks.append(model.value_fc2.register_forward_hook(hook_factory("value_fc2")))

# Also capture last residual block output (shared trunk)
hooks.append(model.res_blocks[-1].register_forward_hook(hook_factory("trunk_out")))

with torch.no_grad():
	p, v = model(t)

print("Output v:")
for (_f, d), val in zip(positions, v.cpu().numpy().ravel()):
	print(f"  v={val:+.5f}  {d}")

print("\n[Trunk output (N, 128, 8, 8)] — variance per position:")
t_out = captured["trunk_out"]
for (_f, d), feat in zip(positions, t_out):
	print(f"  min={feat.min():+.3f} max={feat.max():+.3f} "
	      f"mean={feat.mean():+.3f} std={feat.std():+.3f}  {d}")

print("\n[value_conv output (N, 1, 8, 8)]:")
vc = captured["value_conv"]
for (_f, d), feat in zip(positions, vc):
	print(f"  min={feat.min():+.3f} max={feat.max():+.3f} "
	      f"mean={feat.mean():+.3f} std={feat.std():+.3f}  {d}")

print("\n[value_bn output (N, 1, 8, 8)]:")
vb = captured["value_bn"]
for (_f, d), feat in zip(positions, vb):
	print(f"  min={feat.min():+.3f} max={feat.max():+.3f} "
	      f"mean={feat.mean():+.3f} std={feat.std():+.3f}  {d}")

print("\n[After ReLU on value_bn output]:")
vb_relu = np.maximum(vb, 0)
for (_f, d), feat in zip(positions, vb_relu):
	print(f"  min={feat.min():+.3f} max={feat.max():+.3f} "
	      f"mean={feat.mean():+.3f} std={feat.std():+.3f} "
	      f"nonzero={int((feat>0).sum())}/64  {d}")

print("\n[value_fc1 output (N, 128)] (pre-ReLU):")
vf1 = captured["value_fc1"]
for (_f, d), feat in zip(positions, vf1):
	relu = np.maximum(feat, 0)
	print(f"  min={feat.min():+.3f} max={feat.max():+.3f} "
	      f"mean={feat.mean():+.3f} std={feat.std():+.3f}  "
	      f"post-ReLU nonzero={int((relu>0).sum())}/128  {d}")

print("\n[value_fc2 output (N, 1)] pre-tanh:")
vf2 = captured["value_fc2"]
for (_f, d), feat in zip(positions, vf2):
	print(f"  val={feat.ravel()[0]:+.5f}  tanh={np.tanh(feat.ravel()[0]):+.5f}  {d}")

for h in hooks:
	h.remove()

# --- Sanity: what are BN running stats? ---
print("\n[BN running stats on value_bn]:")
print(f"  running_mean = {model.value_bn.running_mean.item():+.5f}")
print(f"  running_var  = {model.value_bn.running_var.item():+.5f}")
print(f"  weight       = {model.value_bn.weight.item():+.5f}")
print(f"  bias         = {model.value_bn.bias.item():+.5f}")
