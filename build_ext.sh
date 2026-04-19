#!/bin/bash
# Build the native C++ acceleration extension.  Idempotent — safe to re-run.
set -e
cd "$(dirname "$0")"

if [ -d .venv ]; then
	source .venv/bin/activate
fi

python -m pip install --quiet pybind11
python setup.py build_ext --inplace
echo
echo "Extension built.  Look for src/_ext/chess_ext.*.so"
