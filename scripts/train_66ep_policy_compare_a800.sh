#!/usr/bin/env bash
set -Eeuo pipefail

WORK="${WORK:-/share_data/zhangyurui/flow-matching-test-a2d-v2}"
CODE_DIR="${CODE_DIR:-$WORK/code}"
CONFIG_DIR="${CONFIG_DIR:-$WORK/runtime_configs}"
LOG_DIR="${LOG_DIR:-$WORK/logs/policy_compare_66ep}"
STAMP="$(date +%Y%m%d_%H%M%S)"

source "$CODE_DIR/activate_a800.sh"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

run_policy() {
  local name="$1"
  local config="$2"
  local log="$LOG_DIR/${STAMP}_${name}.log"
  echo "POLICY_RUN_START name=$name config=$config log=$log"
  python -u -m flow_matching_test.train --config "$config" 2>&1 | tee "$log"
  echo "POLICY_RUN_DONE name=$name log=$log"
}

run_policy rs_imle "$CONFIG_DIR/a2d_parallel_1507_imle_a800.yaml"
run_policy diffusion "$CONFIG_DIR/a2d_parallel_1507_diffusion_a800.yaml"
