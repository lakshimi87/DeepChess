import torch
import torch.nn as nn
import torch.nn.functional as F

from .board_utils import NUM_MOVES, NUM_PLANES


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
	  Policy : 1x1 conv(32) -> FC -> 4672 logits
	  Value  : 1x1 conv(32) -> FC(256) -> FC(1) -> tanh

	Note: value head uses 32 channels (not the 1 channel from the AlphaZero
	paper).  A single-channel value head puts a scalar BN with one γ/β pair
	right before the FC, and when γ collapses to ~0 the head emits a constant
	value for every position — the classic "value collapse" failure mode.
	32 channels gives the head enough capacity to survive ordinary training
	without that degenerate fixed point.
	"""

	def __init__(self, num_res_blocks=10, num_filters=128):
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

		# Policy head
		self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
		self.policy_bn = nn.BatchNorm2d(32)
		self.policy_fc = nn.Linear(32 * 64, NUM_MOVES)

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

		# Policy head
		p = F.relu(self.policy_bn(self.policy_conv(out)))
		p = p.view(p.size(0), -1)
		p = self.policy_fc(p)

		# Value head
		v = F.relu(self.value_bn(self.value_conv(out)))
		v = v.view(v.size(0), -1)
		v = F.relu(self.value_fc1(v))
		v = torch.tanh(self.value_fc2(v))

		return p, v
