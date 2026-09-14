#!/usr/bin/env bash
set -Eeuo pipefail

run_dir=${1:?usage: watch_a2d_rdt_run.sh RUN_DIR}
launcher_pid=$(cat "$run_dir/launcher.pid")
watch_log=$run_dir/watchdog.log

while kill -0 "$launcher_pid" 2>/dev/null; do
  now=$(date +%s)
  modified=$(stat -c %Y "$run_dir/train.log" 2>/dev/null || echo 0)
  stale_seconds=$((now - modified))
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits | head -1)
  printf '%s pid_alive=1 log_stale_seconds=%s gpu=%s\n' \
    "$(date --iso-8601=seconds)" "$stale_seconds" "$gpu" >> "$watch_log"
  if (( stale_seconds > 600 )); then
    printf '%s stale training log detected\n' "$(date --iso-8601=seconds)" \
      > "$run_dir/WARNING_STALE"
  fi
  sleep 60
done

if grep -q '"status":"failed"' "$run_dir/status.json" 2>/dev/null; then
  cp "$run_dir/status.json" "$run_dir/ALERT_FAILED.json"
fi
