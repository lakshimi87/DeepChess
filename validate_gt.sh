#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

exec python -m src.validate_gt "$@"
