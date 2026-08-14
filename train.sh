#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

# -u: unbuffered stdout.  Without it Python block-buffers when stdout is a
# pipe or file, so `./train.sh > train.log` shows nothing for many minutes and
# a `nohup` log can sit empty for an entire run.
exec python -u -m src.train "$@"
