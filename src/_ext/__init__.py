"""Native C++ extension loader.

Exposes ``impl`` (the compiled module) and ``AVAILABLE`` (bool).  Falls back
silently when the extension hasn't been built — callers must branch on
``AVAILABLE`` and use their pure-Python path in that case.

Build with ``./build_ext.sh`` (or ``pip install pybind11 && python -m pip
install --no-build-isolation -e .``).
"""

try:
	from . import chess_ext as impl
	AVAILABLE = True
except ImportError:
	impl = None
	AVAILABLE = False
