#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="/share_data/zhangyurui/flow-matching-test-a2d-v2/runs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="$RUN_ROOT/cfm_vit_5ep_ablation_${STAMP}.launcher.log"

source "$REPO_ROOT/activate_a800.sh"
ulimit -n 65536
export WANDB_MODE=online

run_one() {
  local variant="$1"
  local config="$2"
  local run_name="cfm_1090ep_vit_${variant}_5ep_seed42_${STAMP}"
  local log_path="$RUN_ROOT/${run_name}.train.log"

  printf '[%s] starting %s\n' "$(date --iso-8601=seconds)" "$run_name" | tee -a "$LAUNCH_LOG"
  python -u -m flow_matching_test.train \
    --config "$REPO_ROOT/$config" \
    "training.run_name=$run_name" \
    2>&1 | tee "$log_path"
  printf '[%s] completed %s\n' "$(date --iso-8601=seconds)" "$run_name" | tee -a "$LAUNCH_LOG"
}

# One A800 is available. Run sequentially to preserve batch size and avoid GPU
# contention; set -e prevents the second experiment from starting after a failure.
run_one "frozen" "configs/a2d_450gb_cfm_a800_vit_frozen_5ep.yaml"
run_one "finetune_01x" "configs/a2d_450gb_cfm_a800_vit_finetune_01x_5ep.yaml"
