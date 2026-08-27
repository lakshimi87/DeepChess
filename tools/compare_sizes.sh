#!/bin/bash
# Pre-train several tower sizes on the same labelled data and print the
# held-out numbers side by side.
#
# The decision this settles is whether capacity is binding.  The signal is
# the sign of (held-out CE - training CE): run4's held-out policy CE ran
# *below* its training CE, which is what a net that is not straining against
# its parameter count looks like, and is why scaling the tower then would
# have bought nothing.  If a bigger tower now improves held-out loss, the
# dense Stockfish labels have moved the binding constraint and the extra
# parameters are earning their self-play slowdown.
set -eu
cd "$(dirname "$0")/.."
DATA=${DATA:-data/bootstrap/labels}
EPOCHS=${EPOCHS:-2}
OUT=${OUT:-/tmp/size_cmp}
mkdir -p "$OUT"
for spec in "8 128" "16 192" "20 256"; do
    set -- $spec
    b=$1; f=$2
    echo "=============== ${b}x${f} ==============="
    .venv/bin/python -u -m src.pretrain --data-dir "$DATA" \
        --res-blocks "$b" --filters "$f" --epochs "$EPOCHS" \
        --out "$OUT/pre_${b}x${f}.pt" 2>&1 | grep -E "Model|Target entropy|epoch|Positions"
done
