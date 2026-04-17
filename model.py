import torch
import torch.nn as nn
import torch.nn.functional as F

from board_utils import NUM_MOVES


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
	  Input  : 18 x 8 x 8 board encoding
	  Body   : 3x3 conv -> residual tower (num_res_blocks blocks)
	  Policy : 1x1 conv(32) -> FC -> 4672 logits
	  Value  : 1x1 conv(1)  -> FC(128) -> FC(1) -> tanh

	Default configuration (~12.5 M parameters):
	  num_res_blocks = 10, num_filters = 128
	"""

	def __init__(self, num_res_blocks=10, num_filters=128):
		super().__init__()
		self.num_res_blocks = num_res_blocks
		self.num_filters = num_filters

		# Input convolution
		self.conv_input = nn.Conv2d(18, num_filters, 3, padding=1, bias=False)
		self.bn_input = nn.BatchNorm2d(num_filters)

		# Residual tower
		self.res_blocks = nn.ModuleList([
			ResidualBlock(num_filters) for _ in range(num_res_blocks)
		])

		# Policy head
		self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
		self.policy_bn = nn.BatchNorm2d(32)
		self.policy_fc = nn.Linear(32 * 64, NUM_MOVES)

		# Value head
		self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
		self.value_bn = nn.BatchNorm2d(1)
		self.value_fc1 = nn.Linear(64, 128)
		self.value_fc2 = nn.Linear(128, 1)

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
