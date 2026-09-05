#!/bin/bash
# One round of expert iteration: the net's own positions, Stockfish's labels.
#
#   nohup setsid tools/expert_round.sh > logs_expert_round.log 2>&1 &
#
# Generation and labelling are pipelined rather than sequential.  The generator
# renames each shard into place when it is full and the labeller runs in
# --watch mode, so Stockfish starts on shard 0 while the net is still playing
# shard 1.  Sequential would idle 12 cores for hours and then idle the GPU for
# hours; overlapped, both stay busy and the round finishes in about the time
# the slower stage takes on its own.
#
# The script waits for any labelling job already running before it starts, so
# it can be queued behind one without oversubscribing the machine.
set -u
cd "$(dirname "$0")/.."

WAIT_PID="${WAIT_PID:-}"           # existing job to queue behind, if any
TARGET="${TARGET:-3000000}"
SHARD="${SHARD:-500000}"
GEN_WORKERS="${GEN_WORKERS:-12}"
LAB_WORKERS="${LAB_WORKERS:-14}"
SIMS="${SIMS:-400}"
DEPTH="${DEPTH:-14}"
CKPT="${CKPT:-checkpoints/pretrained_70M.pt}"
POS_DIR="${POS_DIR:-data/expert/positions}"
LAB_DIR="${LAB_DIR:-data/expert/labels}"

echo "=== expert iteration round: started $(date -Is) ==="
echo "checkpoint $CKPT   target $TARGET positions   ${SIMS} sims"
echo "generate ${GEN_WORKERS} workers -> $POS_DIR"
echo "label    ${LAB_WORKERS} workers depth ${DEPTH} -> $LAB_DIR"

if [ -n "$WAIT_PID" ]; then
	echo "--- waiting for PID $WAIT_PID to finish before starting"
	while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
	echo "--- PID $WAIT_PID gone at $(date -Is); starting"
fi

mkdir -p "$POS_DIR" "$LAB_DIR"

.venv/bin/python -u -m src.gen_positions \
	--checkpoint "$CKPT" \
	--out-dir "$POS_DIR" \
	--target-positions "$TARGET" \
	--shard-size "$SHARD" \
	--workers "$GEN_WORKERS" \
	--simulations "$SIMS" &
GEN_PID=$!
echo "--- generator PID $GEN_PID"

# 60s: long enough that the labeller is not spinning on an empty directory
# through the ~90 minutes the first shard takes, short enough that it picks a
# finished shard up promptly.
.venv/bin/python -u tools/label_sf.py \
	--in-dir "$POS_DIR" --out-dir "$LAB_DIR" \
	--depth "$DEPTH" --workers "$LAB_WORKERS" --watch 60 &
LAB_PID=$!
echo "--- labeller PID $LAB_PID"

wait "$GEN_PID"
GEN_RC=$?
echo "--- generation finished rc=$GEN_RC at $(date -Is)"

# --watch never exits on its own, so drain it here: wait until every shard on
# disk has a label file, then stop it.  Checked by name rather than by count so
# a shard that appeared late is not missed.
echo "--- draining labeller"
while true; do
	pending=0
	for p in "$POS_DIR"/pos_*.tsv; do
		[ -e "$p" ] || continue
		l="$LAB_DIR/$(basename "${p/pos_/lab_}")"
		[ -e "$l" ] || pending=$((pending + 1))
	done
	if [ "$pending" -eq 0 ]; then break; fi
	if ! kill -0 "$LAB_PID" 2>/dev/null; then
		echo "!!! labeller exited with $pending shard(s) unlabelled"
		break
	fi
	echo "    $pending shard(s) still to label — $(date +%H:%M)"
	sleep 300
done
kill "$LAB_PID" 2>/dev/null
wait "$LAB_PID" 2>/dev/null

echo "=== round finished $(date -Is) ==="
wc -l "$POS_DIR"/pos_*.tsv "$LAB_DIR"/lab_*.tsv 2>/dev/null | tail -20
echo
echo "Next: pre-train on bootstrap + this corpus.  --human-weight 0 is not"
echo "optional here — the played-move column is the net's own move, so the"
echo "default 0.30 would put a third of the policy target on the net copying"
echo "itself.  --result-weight 0 too: games cut at the move limit write a 0"
echo "in the result column that means 'unknown', not 'draw'."
