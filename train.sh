#!/bin/bash
set -e
cd "$(dirname "$0")"
source "$HOME/venvs/torch/bin/activate"

exec python -m src.train "$@"
