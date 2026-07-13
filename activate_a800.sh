#!/usr/bin/env bash

# Source this file before every setup, validation, or training command on A800.
export WORK=/share_data/zhangyurui/flow-matching-test-a2d-v2
export VIRTUAL_ENV="$WORK/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Keep every writable cache and runtime artifact inside the project directory.
export HOME="$WORK/home"
export TMPDIR="$WORK/tmp"
export XDG_CACHE_HOME="$WORK/cache/xdg"
export PIP_CACHE_DIR="$WORK/cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_REQUIRE_VIRTUALENV=true
export PYTHONNOUSERSITE=1
export HF_HOME="$WORK/cache/huggingface"
export TORCH_HOME="$WORK/cache/torch"
export MPLCONFIGDIR="$WORK/cache/matplotlib"
export WANDB_DIR="$WORK/logs/wandb"

mkdir -p \
  "$HOME" \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  "$WANDB_DIR"
