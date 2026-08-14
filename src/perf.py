"""Process-wide performance knobs.

Call :func:`configure` once at the top of any entry point (train, play,
validate, self-play worker) *before* building models.

The single most important knob here is ``torch.set_num_threads(1)``.
Self-play spends its time in single-position CPU work (python-chess move
generation) plus small GPU forward passes; there is no large CPU tensor op
to parallelise.  With PyTorch's default (one thread per core) the OpenMP
pool spin-waits at a barrier on every tiny CPU op, which measured at ~11
cores of pure burn and made MCTS *2x slower* than with a single thread:

    threads=20, mcts_batch=32  : 160 ms/move
    threads=1,  mcts_batch=32  :  83 ms/move
    threads=1,  mcts_batch=128 :  59 ms/move
"""

import os

import torch


def configure(num_threads=1, cudnn_benchmark=True, tf32=True):
	"""Apply the standard performance settings for this project."""
	# Belt-and-braces: some BLAS backends read the env var, not the torch API.
	os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
	os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
	torch.set_num_threads(num_threads)
	try:
		torch.set_num_interop_threads(num_threads)
	except RuntimeError:
		pass  # already initialised — harmless

	if torch.cuda.is_available():
		# Fixed input shapes (we pad eval batches), so autotuning pays off
		# once and then every conv picks the fastest kernel.
		torch.backends.cudnn.benchmark = cudnn_benchmark
		if tf32:
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True


def to_inference(model, device, half=True, channels_last=True):
	"""Return *model* configured for fast inference on *device*.

	``channels_last`` + ``cudnn.benchmark`` measured 1.22-1.41x on the
	forward pass for this 16x192 tower vs plain NCHW fp16.
	"""
	model = model.to(device).eval()
	if half and device.type in ("cuda", "mps"):
		model = model.half()
	if channels_last and device.type == "cuda":
		model = model.to(memory_format=torch.channels_last)
	return model
