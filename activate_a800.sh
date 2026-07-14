#!/usr/bin/env bash

# Source this file before every setup, validation, or training command on A800.
export WORK=/share_data/zhangyurui/flow-matching-test-a2d-v2
export VIRTUAL_ENV="$WORK/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Keep persistent caches and artifacts inside the project directory.
export HOME="$WORK/home"
export WORK_TMPDIR="$WORK/tmp"
# W&B uses local IPC under TMPDIR; CFS-backed temp directories can time out.
export TMPDIR="/tmp/flow-matching-test-${UID}"
export XDG_CACHE_HOME="$WORK/cache/xdg"
export PIP_CACHE_DIR="$WORK/cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_REQUIRE_VIRTUALENV=true
export PYTHONNOUSERSITE=1
export HF_HOME="$WORK/cache/huggingface"
export TORCH_HOME="$WORK/cache/torch"
export MPLCONFIGDIR="$WORK/cache/matplotlib"
export WANDB_DIR="$WORK/logs/wandb"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

mkdir -p \
  "$HOME" \
  "$WORK_TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  "$WANDB_DIR"

mkdir -p -m 700 "$TMPDIR"
