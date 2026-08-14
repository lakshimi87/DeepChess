import torch
import torch.nn as nn
import torch.nn.functional as F

from .board_utils import NUM_MOVES, NUM_PLANES

# Move-type planes per from-square (NUM_MOVES = 64 squares x 73 move types).
_MOVE_PLANES = NUM_MOVES // 64  # == 73


class ResidualBlock(nn.Module):
	"""Pre-activation residual block: conv -> BN -> ReLU -> conv -> BN + skip -> ReLU."""

	def __init__(self, num_filters):
		super().__init__()
		self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(num_filters)
		self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(num_filters)

	def forward(self, x):
		residual = x
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out = F.relu(out + residual)
		return out


class ChessNet(nn.Module):
	"""AlphaZero-style dual-head neural network for chess.

	Architecture:
	  Input  : NUM_PLANES x 8 x 8 board encoding
	  Body   : 3x3 conv -> residual tower (num_res_blocks blocks)
	  Policy : 3x3 conv -> 1x1 conv(73 planes) -> 4672 logits (no FC)
	  Value  : 1x1 conv(32) -> FC(256) -> FC(1) -> tanh

	Policy head is fully convolutional (AlphaZero-style).  The final 1x1 conv
	emits 73 move-type planes over the 8x8 board; flattening as
	(from_square, move_type) — index = from_sq*73 + move_type — matches
	board_utils.move_to_index exactly, so each logit's spatial position is its
	move's from-square.  This shares weights across the board instead of the
	old flatten->Linear(2048, 4672) head, which alone held ~9.6M parameters
	(73% of the whole net) with no spatial weight sharing.

	Note: value head uses 32 channels (not the 1 channel from the AlphaZero
	paper).  A single-channel value head puts a scalar BN with one γ/β pair
	right before the FC, and when γ collapses to ~0 the head emits a constant
	value for every position — the classic "value collapse" failure mode.
	32 channels gives the head enough capacity to survive ordinary training
	without that degenerate fixed point.
	"""

	def __init__(self, num_res_blocks=16, num_filters=192):
		super().__init__()
		self.num_res_blocks = num_res_blocks
		self.num_filters = num_filters

		# Input convolution
		self.conv_input = nn.Conv2d(NUM_PLANES, num_filters, 3, padding=1, bias=False)
		self.bn_input = nn.BatchNorm2d(num_filters)

		# Residual tower
		self.res_blocks = nn.ModuleList([
			ResidualBlock(num_filters) for _ in range(num_res_blocks)
		])

		# Policy head — fully convolutional (no FC).
		self.policy_conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
		self.policy_bn = nn.BatchNorm2d(num_filters)
		self.policy_conv2 = nn.Conv2d(num_filters, _MOVE_PLANES, 1)

		# Value head — 32 channels + wider FC
		self.value_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
		self.value_bn = nn.BatchNorm2d(32)
		self.value_fc1 = nn.Linear(32 * 64, 256)
		self.value_fc2 = nn.Linear(256, 1)

	def forward(self, x):
		# Input block
		out = F.relu(self.bn_input(self.conv_input(x)))

		# Residual tower
		for block in self.res_blocks:
			out = block(out)

		# Policy head — (B, 73, 8, 8) reshaped to (B, 4672) as
		# index = from_sq*73 + move_type so it lines up with board_utils.
		p = F.relu(self.policy_bn(self.policy_conv1(out)))
		p = self.policy_conv2(p)
		p = p.permute(0, 2, 3, 1).reshape(p.size(0), -1)

		# Value head
		v = F.relu(self.value_bn(self.value_conv(out)))
		# reshape (not view): under channels_last the NHWC strides aren't
		# view-compatible with a flat (B, 32*64) layout.
		v = v.reshape(v.size(0), -1)
		v = F.relu(self.value_fc1(v))
		v = torch.tanh(self.value_fc2(v))

		return p, v
