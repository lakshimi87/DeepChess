#!/bin/bash
# Extract positions from every downloaded month, newest download last.
# Waits for download_lichess.sh to mark a month complete (.done) before
# touching it, so a partially transferred .zst is never fed to zstd.
# Shard numbering continues across months: the extractor is given a distinct
# --shard-offset per month so two months never write pos_00000.tsv.
set -u
cd "$(dirname "$0")/.."
OUT=data/bootstrap/positions
mkdir -p "$OUT"
for m in "$@"; do
    f="data/bootstrap/pgn/lichess_db_standard_rated_$m.pgn.zst"
    stamp="$OUT/.extracted_$m"
    [ -f "$stamp" ] && { echo "[skip] $m already extracted"; continue; }
    echo "[wait] $m download"
    while [ ! -f "$f.done" ]; do sleep 60; done
    # Continue numbering after whatever is already on disk.
    next=$(( $(ls "$OUT" | grep -c '^pos_.*\.tsv$') ))
    echo "[extr] $m -> shards from $next"
    zstd -dc "$f" | .venv/bin/python tools/fetch_lichess.py \
        --out-dir "$OUT" --min-elo 1800 --shard-size 1000000 \
        --workers 10 --shard-start "$next" && touch "$stamp"
    echo "[extr] $m done"
done
echo "[all extracted]"
