"""Build script for the C++ acceleration extension.

Usage:
    pip install pybind11
    python setup.py build_ext --inplace

The compiled ``chess_ext.*.so`` ends up next to ``src/_ext/__init__.py`` which
loads it lazily.  The Python code falls back to a pure-Python implementation
if the extension is missing.
"""

import sys

from setuptools import setup, Extension

try:
	import pybind11
except ImportError:
	sys.stderr.write(
		"pybind11 is required to build the C++ extension.\n"
		"Install with: pip install pybind11\n"
	)
	raise

ext_modules = [
	Extension(
		"src._ext.chess_ext",
		sources=["src/_ext/chess_ext.cpp"],
		include_dirs=[pybind11.get_include()],
		language="c++",
		extra_compile_args=["-O3", "-std=c++17", "-Wall"],
	),
]

setup(
	name="deepchess-ext",
	version="0.1.0",
	description="DeepChess native acceleration",
	ext_modules=ext_modules,
)
