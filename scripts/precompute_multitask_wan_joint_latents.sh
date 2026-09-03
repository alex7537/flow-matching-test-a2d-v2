#!/bin/bash
set -euo pipefail

DATA_DIR="${1:-/share_data/zhangyurui/flow-matching-test-a2d-v2/code/datasets/a2d_v3_multitask_box300_bottle300_seed42}"
CACHE_DIR="${2:-/share_data/zhangyurui/flow-matching-test-a2d-v2/cache/wan_latents/a2d_v3_multitask_box300_bottle300_joint_wam_v2}"
VAE_PATH="${3:-/share_data/zhangyurui/.cache/huggingface/local-models/Wan2.2-TI2V-5B-vae/Wan2.2_VAE.pth}"
WAN_REPO="${4:-/share_data/zhangyurui/.cache/open-source/Wan2.2}"

python -m scripts.precompute_wan_joint_latents \
  --data-dir "$DATA_DIR" \
  --cache-dir "$CACHE_DIR" \
  --vae-checkpoint "$VAE_PATH" \
  --wan-runtime-repo "$WAN_REPO" \
  --video-key rgb_head \
  --condition-steps 9 \
  --future-steps 16 \
  --future-offset-steps 1 \
  --batch-size 4
