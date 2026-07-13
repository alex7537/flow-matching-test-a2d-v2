#!/bin/bash
set -euo pipefail

source /share_data/projects/dev-algorithm/miniconda3/bin/activate
conda activate psipolicy-env-v2

cd /home/psibot/Downloads/flow-matching-test

pwd
which python

# Optional:
# export CUDA_VISIBLE_DEVICES=0

python -m flow_matching_test.train \
  --config configs/minimal_rgb_flow_timm.yaml \
  data.data_dir=/path/to/success_hdf5_dir \
  data.rgb_root=/path/to/rgb_root \
  training.device=cuda \
  training.batch_size=16 \
  training.num_workers=8 \
  training.num_epochs=20 \
  visualization.rerun.enabled=true \
  visualization.rerun.spawn=false
