#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

exec python validate_gt.py "$@"
