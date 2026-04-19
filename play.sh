#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

DIFFICULTY="${1:-normal}"
exec python -m src.main --difficulty "$DIFFICULTY"
