#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/share_data/zhangyurui/flow-matching-test-a2d-v2/code"
RUN_ROOT="/share_data/zhangyurui/flow-matching-test-a2d-v2/runs"
QUEUE_ID="${1:-cfm_stage2_10way_20260722_194500}"
QUEUE_DIR="$RUN_ROOT/$QUEUE_ID"
QUEUE_PID="${2:-100811}"
LOG="$QUEUE_DIR/finalizer.log"

on_error() {
  local exit_code=$?
  printf '[%s] finalizer failed with exit_code=%s; inspect %s\n' \
    "$(date --iso-8601=seconds)" "$exit_code" "$LOG" > "$QUEUE_DIR/FINALIZATION_FAILED"
  exit "$exit_code"
}
trap on_error ERR

while kill -0 "$QUEUE_PID" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "$QUEUE_DIR/QUEUE_COMPLETE" ]]; then
  printf '[%s] queue exited without QUEUE_COMPLETE\n' "$(date --iso-8601=seconds)" > "$QUEUE_DIR/FINALIZATION_FAILED"
  exit 1
fi

cd "$REPO_ROOT"
source ./activate_a800.sh
python -u -m scripts.finalize_cfm_stage2_10way \
  --repo-root "$REPO_ROOT" \
  --run-root "$RUN_ROOT" \
  --queue-id "$QUEUE_ID" \
  2>&1 | tee "$LOG"
