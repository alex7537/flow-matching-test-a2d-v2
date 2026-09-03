#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /absolute/path/to/mixed_cfm_checkpoint.ckpt [data_dir] [latent_cache_dir]" >&2
  exit 2
fi

INIT_CKPT="$1"
DATA_DIR="${2:-/share_data/zhangyurui/flow-matching-test-a2d-v2/code/datasets/a2d_v3_multitask_box300_bottle300_seed42}"
CACHE_DIR="${3:-/share_data/zhangyurui/flow-matching-test-a2d-v2/cache/wan_latents/a2d_v3_multitask_box300_bottle300_wan22_9plus16_v1}"
CONFIG="configs/a2d_v3_multitask_video_aux_posttrain_10ep.yaml"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "checkpoint not found: $INIT_CKPT" >&2
  exit 2
fi
if [[ ! -f "$DATA_DIR/dataset_manifest.json" || ! -f "$DATA_DIR/split_manifest.json" ]]; then
  echo "dataset manifests not found under: $DATA_DIR" >&2
  exit 2
fi
if [[ ! -f "$CACHE_DIR/video_latent_manifest.json" ]]; then
  echo "latent cache is not ready: $CACHE_DIR/video_latent_manifest.json" >&2
  echo "run scripts/precompute_multitask_wan_latents.sh first" >&2
  exit 2
fi

python -m flow_matching_test.train \
  --config "$CONFIG" \
  "training.init_from=$INIT_CKPT" \
  "data.data_dir=$DATA_DIR" \
  "data.video_latent_cache_dir=$CACHE_DIR"
