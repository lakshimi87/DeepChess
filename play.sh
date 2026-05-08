#!/bin/bash
set -e
cd "$(dirname "$0")"
source "$HOME/venvs/torch/bin/activate"

DIFFICULTY="${1:-normal}"
COLOR="${2:-random}"
exec python -m src.main --difficulty "$DIFFICULTY" --color "$COLOR"
