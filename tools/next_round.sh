#!/bin/bash
# Retrain and re-measure each time the labelled corpus reaches a new size.
#
# Runs unattended: launch it detached and it keeps producing rounds while the
# labeller feeds it, appending one row per round to results.md.  Each round is
# pre-training from scratch on everything labelled so far -- not a resume --
# because the point is to measure what a given corpus size is worth, and a
# warm start would confound that with however long the previous round trained.
set -u
cd "$(dirname "$0")/.."
mkdir -p .run
echo $$ > .run/round.pid
trap 'rm -f .run/round.pid' EXIT
TARGETS=${TARGETS:-"2 4 8 16 32 48 70"}   # millions of labelled positions
EPOCHS=${EPOCHS:-3}
GAMES=${GAMES:-30}
SIMS=${SIMS:-400}

[ -f results.md ] || cat > results.md <<'HDR'
# DeepChess 부트스트랩 결과

고전 엔진(depth 3 미니맥스)과의 대전 성적이 목표 지표.
GT는 837문제 스위트(1σ ±1.7점), 홀드아웃 월에서 Stockfish depth 18로 검증.

| 라운드 | 라벨 국면 | 모델 | GT 837 | 대전 (W-L-D) | 승률 | held-out value MSE |
| --- | --- | --- | --- | --- | --- | --- |
| 기준선 run4 | 0 (self-play만) | 8x128 | 380/837 (45%) | 0-15-4 | 10.5% | 0.39~0.45 (baseline 초과) |
| 사전학습 1차 | 0.6M | 16x192 | 629/837 (75%) | 3-10-17 | 38.3% | 0.168 (baseline 0.470) |
HDR

for t in $TARGETS; do
    tag="${t}M"
    ck="checkpoints/pretrained_${tag}.pt"
    log="logs_round_${tag}.log"
    [ -f "$ck" ] && { echo "[skip] $tag 이미 완료"; continue; }

    echo "[wait] 라벨 샤드 ${t}개 대기중 …"
    while [ "$(ls data/bootstrap/labels/lab_*.tsv 2>/dev/null | wc -l)" -lt "$t" ]; do
        sleep 300
        # The labeller can finish the whole corpus before reaching a target;
        # don't wait forever for shards that will never exist.
        if ! kill -0 "$(cat .run/label.pid 2>/dev/null)" 2>/dev/null; then
            have=$(ls data/bootstrap/labels/lab_*.tsv 2>/dev/null | wc -l)
            [ "$have" -lt "$t" ] && { echo "[stop] 라벨러 종료, ${have}M에서 마감"; exit 0; }
        fi
    done

    echo "[run ] 라운드 $tag 사전학습 시작 $(date '+%H:%M')"
    .venv/bin/python -u -m src.pretrain --data-dir data/bootstrap/labels \
        --epochs "$EPOCHS" --out "$ck" > "$log" 2>&1 || { echo "[fail] $tag"; continue; }

    echo "[run ] 라운드 $tag 측정 시작 $(date '+%H:%M')"
    .venv/bin/python -u -m src.validate_gt --checkpoint "$ck" \
        --games "$GAMES" --simulations "$SIMS" >> "$log" 2>&1

    gt=$(grep -oE "TOTAL +[0-9]+/[0-9]+ +\( *[0-9]+%\)" "$log" | head -1 \
         | sed -E 's/TOTAL +//; s/ +\( */ (/')
    res=$(grep -oE "W=[0-9]+ +L=[0-9]+ +D=[0-9]+.*win rate [0-9.]+%" "$log" | head -1)
    wl=$(echo "$res" | grep -oE "W=[0-9]+ +L=[0-9]+ +D=[0-9]+" | tr -s ' ' | sed 's/W=//;s/L=//;s/D=//;s/ /-/g')
    wr=$(echo "$res" | grep -oE "win rate [0-9.]+%" | sed 's/win rate //')
    mse=$(grep -oE "value MSE [0-9.]+ +\(baseline [0-9.]+" "$log" | tail -1 \
          | sed -E 's/value MSE //; s/ +\(baseline /  (baseline /')
    printf "| %s | %s | 16x192 | %s | %s | %s | %s) |\n" \
        "$tag" "${tag} 국면" "${gt:-?}" "${wl:-?}" "${wr:-?}" "${mse:-?}" >> results.md
    echo "[done] 라운드 $tag → results.md 기록됨"
done
echo "[all rounds done]"
