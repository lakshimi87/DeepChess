#!/bin/bash
set -e
cd "$(dirname "$0")"
source "$HOME/venvs/torch/bin/activate"

DIFFICULTY="${1:-normal}"
exec python -m src.main --difficulty "$DIFFICULTY"
