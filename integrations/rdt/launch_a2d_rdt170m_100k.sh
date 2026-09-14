#!/usr/bin/env bash
set -Eeuo pipefail

base=/share_data/zhangyurui/rdt-a2d-170m-smoke
source_root=$base/source
data_dir=/share_data/zhangyurui/flow-matching-test-a2d-v2/code/datasets/a2d_v3_multitask_box300_bottle300_seed42__rdt_v1_emptylang
rdt_model=$base/cache/huggingface/hub/models--robotics-diffusion-transformer--rdt-170m/snapshots/8aa386cac3bbfd9540676c75b3d767cc7f88a10a
siglip_model=$base/cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3
python_bin=/share_data/zhangyurui/flow-matching-test-a2d-v2/venv/bin/python
run_name=rdt170m_a2d_v3_multitask600_emptylang_masked_100k_seed42_20260912_v2
output_dir=$base/runs/$run_name

if [[ -e "$output_dir" ]]; then
  echo "refusing to reuse output directory: $output_dir" >&2
  exit 17
fi
mkdir -p "$output_dir/wandb"

export PYTHONPATH=$base/python-packages:$source_root
export HF_HOME=$base/cache/huggingface
export A2D_RDT_DATA_DIR=$data_dir
export A2D_RDT_TASK_INSTRUCTIONS=1
export A2D_RDT_LANG_EMBED_DIR=$data_dir/language
export WANDB_MODE=online
export WANDB_DIR=$output_dir/wandb
export WANDB_ENTITY=z1135783608-psibot
export WANDB_PROJECT=a2d-flow-matching
export WANDB_NAME=$run_name

printf '{"status":"running","pid":%s,"started_at":"%s"}\n' \
  "$$" "$(date --iso-8601=seconds)" > "$output_dir/status.json"
printf '%s\n' "$$" > "$output_dir/launcher.pid"

cd "$source_root"
set +e
"$python_bin" -m accelerate.commands.launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  main.py \
  --pretrained_model_name_or_path "$rdt_model" \
  --pretrained_vision_encoder_name_or_path "$siglip_model" \
  --output_dir "$output_dir" \
  --seed 42 \
  --train_batch_size 8 \
  --sample_batch_size 4 \
  --num_sample_batches 8 \
  --max_train_steps 100000 \
  --checkpointing_period 5000 \
  --checkpoints_total_limit 10 \
  --sample_period 1000 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 5000 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --dataloader_num_workers 4 \
  --dataset_type finetune \
  --load_from_hdf5 \
  --precomp_lang_embed \
  --allow_tf32 \
  --set_grads_to_none \
  --report_to wandb \
  2>&1 | tee "$output_dir/train.log"
rc=${PIPESTATUS[0]}
set -e

status=failed
if [[ $rc -eq 0 ]]; then
  status=completed
fi
printf '{"status":"%s","exit_code":%s,"finished_at":"%s"}\n' \
  "$status" "$rc" "$(date --iso-8601=seconds)" > "$output_dir/status.json"
exit "$rc"
