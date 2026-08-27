#!/bin/bash
# Fetch Lichess monthly PGN dumps one at a time into data/bootstrap/pgn/.
# The link here runs at ~1.5 MB/s, so a month takes 30-150 min; keeping the
# .zst on disk means re-extracting with different sampling settings later
# costs minutes instead of another download.  curl -C - resumes a partial file.
set -u
cd "$(dirname "$0")/.."
mkdir -p data/bootstrap/pgn
for m in "$@"; do
    f="data/bootstrap/pgn/lichess_db_standard_rated_$m.pgn.zst"
    if [ -f "$f.done" ]; then echo "[skip] $m"; continue; fi
    echo "[get ] $m -> $f"
    curl -sL -C - -o "$f" \
        "https://database.lichess.org/standard/lichess_db_standard_rated_$m.pgn.zst" \
        && touch "$f.done" && echo "[done] $m ($(du -h "$f" | cut -f1))"
done
echo "[all done]"
