#!/bin/bash
# At-a-glance state of the bootstrap pipeline.  Safe to run any time.
cd "$(dirname "$0")/.."
echo "══════════════════════════════════════════════════════════"
echo "  DeepChess 부트스트랩 상태   $(date '+%Y-%m-%d %H:%M')"
echo "══════════════════════════════════════════════════════════"

# Check by pidfile, not by pattern: `pgrep -f next_round.sh` also matches
# this script's own command line when it is launched from a wrapper shell,
# which reports every job as running whether or not it is.
running() {
    local pf=".run/$1.pid"
    [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null \
        && echo "실행중 (PID $(cat "$pf"))" || echo "정지"
}

pos=$(ls data/bootstrap/positions/pos_*.tsv 2>/dev/null | wc -l)
lab=$(ls data/bootstrap/labels/lab_*.tsv 2>/dev/null | wc -l)
part=$(wc -l < data/bootstrap/labels/*.tmp 2>/dev/null | tail -1)
echo "  추출 국면    : ${pos}백만 (샤드 ${pos}개)"
echo "  라벨 완료    : ${lab}백만 (샤드 ${lab}개, 작업중 ${part:-0})"
echo "  라벨러       : $(running label)"
echo "  다음 라운드  : $(running round)"
echo "  디스크       : $(du -sh data 2>/dev/null | cut -f1)"

echo "──────────────────────────────────────────────────────────"
echo "  결과 기록 (results.md):"
if [ -f results.md ]; then tail -12 results.md | sed 's/^/  /'; else echo "  아직 없음"; fi
echo "──────────────────────────────────────────────────────────"
echo "  로그: logs_label.log  logs_round*.log  results.md"
echo "  체크포인트: $(ls -t checkpoints/pretrained*.pt 2>/dev/null | head -3 | tr '\n' ' ')"
