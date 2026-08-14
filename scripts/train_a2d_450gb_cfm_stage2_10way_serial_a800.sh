#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="/share_data/zhangyurui/flow-matching-test-a2d-v2"
RUN_ROOT="$WORK_ROOT/runs"
CONFIG="$REPO_ROOT/configs/a2d_450gb_cfm_a800_stage2_3ep.yaml"
STAMP="${STAGE2_STAMP:-$(date +%Y%m%d_%H%M%S)}"
QUEUE_NAME="cfm_stage2_10way_${STAMP}"
QUEUE_DIR="$RUN_ROOT/$QUEUE_NAME"
QUEUE_LOG="$QUEUE_DIR/launcher.log"
RESULTS="$QUEUE_DIR/results.tsv"
PLAN="$QUEUE_DIR/plan.tsv"
LOCK_FILE="$RUN_ROOT/.a800_training.lock"

mkdir -p "$QUEUE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'Another training queue owns %s; refusing to start.\n' "$LOCK_FILE" >&2
  exit 75
fi

source "$REPO_ROOT/activate_a800.sh"
ulimit -n 65536
export WANDB_MODE=online

mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
if ((${#GPU_PIDS[@]} > 0)); then
  printf 'GPU already has compute processes: %s\n' "${GPU_PIDS[*]}" >&2
  exit 76
fi

printf 'variant\tstatus\trun_name\tstarted_at\tfinished_at\n' > "$RESULTS"
cat > "$PLAN" <<'EOF'
variant	head_lr	backbone_multiplier	weight_decay	betas	grad_clip	inference_steps
v01_lr1e5_bb01	1e-5	0.1	1e-4	0.9,0.95	1.0	10
v02_lr2e5_bb01	2e-5	0.1	1e-4	0.9,0.95	1.0	10
v03_lr5e5_bb01	5e-5	0.1	1e-4	0.9,0.95	1.0	10
v04_lr2e5_frozen	2e-5	0.0/frozen	1e-4	0.9,0.95	1.0	10
v05_lr2e5_bb005	2e-5	0.05	1e-4	0.9,0.95	1.0	10
v06_lr2e5_bb02	2e-5	0.2	1e-4	0.9,0.95	1.0	10
v07_lr2e5_no_wd	2e-5	0.1	0	0.9,0.95	1.0	10
v08_lr2e5_wd1e3	2e-5	0.1	1e-3	0.9,0.95	1.0	10
v09_lr2e5_beta2_099	2e-5	0.1	1e-4	0.9,0.99	1.0	10
v10_lr2e5_clip05_20s	2e-5	0.1	1e-4	0.9,0.95	0.5	20
EOF

active_variant="queue_setup"
active_run=""
started_at=""
on_error() {
  local exit_code=$?
  local failed_at
  failed_at="$(date --iso-8601=seconds)"
  printf '%s\tFAILED(%s)\t%s\t%s\t%s\n' \
    "$active_variant" "$exit_code" "$active_run" "$started_at" "$failed_at" >> "$RESULTS"
  printf '[%s] queue stopped after %s failed with exit %s\n' \
    "$failed_at" "$active_variant" "$exit_code" | tee -a "$QUEUE_LOG"
  exit "$exit_code"
}
trap on_error ERR

run_one() {
  local variant="$1"
  shift
  active_variant="$variant"
  active_run="cfm_1090ep_stage2_${variant}_3ep_seed42_${STAMP}"
  started_at="$(date --iso-8601=seconds)"
  local log_path="$QUEUE_DIR/${variant}.train.log"

  printf '[%s] starting %s\n' "$started_at" "$active_run" | tee -a "$QUEUE_LOG"
  python -u -m flow_matching_test.train \
    --config "$CONFIG" \
    "training.run_name=$active_run" \
    "$@" \
    2>&1 | tee "$log_path"
  local finished_at
  finished_at="$(date --iso-8601=seconds)"
  printf '%s\tCOMPLETED\t%s\t%s\t%s\n' \
    "$variant" "$active_run" "$started_at" "$finished_at" >> "$RESULTS"
  printf '[%s] completed %s\n' "$finished_at" "$active_run" | tee -a "$QUEUE_LOG"
}

# Equal budget for every candidate: 3 epochs = 15,267 optimizer steps.
# Only one process is launched at a time; set -e and pipefail stop the queue on failure.
run_one v01_lr1e5_bb01       training.lr=0.00001 training.backbone_lr_multiplier=0.1
run_one v02_lr2e5_bb01       training.lr=0.00002 training.backbone_lr_multiplier=0.1
run_one v03_lr5e5_bb01       training.lr=0.00005 training.backbone_lr_multiplier=0.1
run_one v04_lr2e5_frozen     training.lr=0.00002 training.backbone_lr_multiplier=0.0 model.freeze_encoder_backbone=true
run_one v05_lr2e5_bb005      training.lr=0.00002 training.backbone_lr_multiplier=0.05
run_one v06_lr2e5_bb02       training.lr=0.00002 training.backbone_lr_multiplier=0.2
run_one v07_lr2e5_no_wd      training.lr=0.00002 training.weight_decay=0.0
run_one v08_lr2e5_wd1e3      training.lr=0.00002 training.weight_decay=0.001
run_one v09_lr2e5_beta2_099  training.lr=0.00002 'training.betas=[0.9,0.99]'
run_one v10_lr2e5_clip05_20s training.lr=0.00002 training.grad_clip=0.5 model.num_inference_steps=20

trap - ERR
printf '[%s] all 10 training runs completed\n' "$(date --iso-8601=seconds)" | tee -a "$QUEUE_LOG"
touch "$QUEUE_DIR/QUEUE_COMPLETE"
