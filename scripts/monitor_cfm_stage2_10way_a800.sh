#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="/share_data/zhangyurui/flow-matching-test-a2d-v2/runs"
QUEUE_ID="${1:-cfm_stage2_10way_20260722_194500}"
QUEUE_DIR="$RUN_ROOT/$QUEUE_ID"

printf '=== queue results ===\n'
cat "$QUEUE_DIR/results.tsv"

printf '\n=== active trainer ===\n'
trainer_pids="$(pgrep -f '[f]low_matching_test.train.*cfm_1090ep_stage2' || true)"
if [[ -n "$trainer_pids" ]]; then
  # One parent trainer plus num_workers DataLoader children is expected.
  ps -o pid=,ppid=,stat=,etime=,cmd= -p $(tr '\n' ' ' <<<"$trainer_pids")
else
  printf 'no active trainer\n'
fi

printf '\n=== GPU ===\n'
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader

printf '\n=== completed epoch metrics ===\n'
find "$RUN_ROOT" -maxdepth 2 -type f -path "*/cfm_1090ep_stage2_*_20260722_194500/metrics.jsonl" \
  -print0 | sort -z | while IFS= read -r -d '' metrics; do
    run_name="$(basename "$(dirname "$metrics")")"
    printf '%s\t' "$run_name"
    tail -n 1 "$metrics"
  done

printf '\n=== launcher tail ===\n'
tail -n 30 "$QUEUE_DIR/launcher.log"
