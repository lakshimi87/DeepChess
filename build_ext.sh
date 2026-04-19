#!/bin/bash
# Build the native C++ acceleration extension.  Idempotent — safe to re-run.
set -e
cd "$(dirname "$0")"

source "$HOME/venvs/torch/bin/activate"

python -m pip install --quiet pybind11
python setup.py build_ext --inplace
echo
echo "Extension built.  Look for src/_ext/chess_ext.*.so"
