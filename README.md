# Flow Matching Test

面向 A2D 机器人抓取数据的连续动作策略训练与部署仓库。当前已验证主线是双 RGB（可选 proprio）条件的 Flow Matching；RS-IMLE 与 Diffusion Policy 保留为对比实现。

## 当前主线

```text
processed HDF5
  -> A2DProcessedWindowDataset
  -> RGB / proprio condition tokens
  -> Flow Matching action policy
  -> 16 × 13 absolute-joint action chunk
  -> checkpoint / verified rollout bundle
```

当前 V3 数据契约：

```text
observation/qpos = arm2_pos(7) + hand2_pos(6)
                  实际手臂状态 + 实际手部状态

action           = arm2_pos(7) + hand2_pos_target(6)
                  实际手臂轨迹 + 手部 commanded target
```

V3 的目的不是把 target 当作 observation，而是把稳定的手部控制意图作为未来 action GT，避免模型模仿接触、回弹和跟踪误差造成的实际手指抖动。action 语义会写入 dataset、normalizer、checkpoint、bundle 与 rollout contract，混用旧语义时直接报错。

## 时间窗口与尾部标签

默认配置：

```text
history_steps       = 1
action_offset_steps = 1
action_horizon      = 16
```

所以训练对齐为：

```text
obs[t] -> action[t+1 : t+17]
obs[t+1] -> action[t+2 : t+18]
```

窗口以 stride 1 逐帧滑动。episode 尾部不足 16 步时重复最后一个真实 action 以保持固定形状，同时返回 `action_mask`；padding 位置不参与 loss。`include_tail_padded_windows: true` 会保留每个 episode 最后 15 个窗口，其中包含 lift 的窗口还会设置 `is_lift=true`，训练集可通过 `lift_oversample_factor` 增加曝光。

## 模型

Observation condition：

- 每路 RGB 经共享 CNN 或 timm/ViT encoder 产生视觉 tokens；
- RGB+proprio 模型将归一化 13 维实际 qpos 投影为一个 proprio token；
- 纯 RGB 模型设置 `model.use_proprio: false`。

Flow Matching 训练：

```text
clean action A, noise Z, flow time τ
Xτ = (1-τ)Z + τA
target velocity = A - Z

noisy action tokens --self-attention--+
                                      +--> velocity head --> masked MSE
observation tokens ----cross-attention+
```

推理从随机 action noise 开始，按配置的 CFM steps 积分得到未来 action chunk。

## Policy

配置通过 `policy.type` 选择：

```yaml
policy:
  type: flow_matching  # flow_matching | imle | diffusion
```

三种 policy 的训练目标与数值空间不同，loss 不可直接横向排名。正式比较必须固定数据、split、网络、step budget 与 seed，并使用 rollout 成功率、action MSE 和推理延迟。

## 快速开始

环境检查：

```bash
python3 -m flow_matching_test.check_timm_env \
  --model-name vit_small_r26_s32_224 \
  --pretrained
```

CPU smoke test：

```bash
python3 -m flow_matching_test.train \
  --config configs/cpu_smoke.yaml \
  data.data_dir=/path/to/processed_dataset \
  training.output_dir=/tmp/a2d_cpu_smoke
```

V3 RGB+proprio 100 epochs：

```bash
python3 -u -m flow_matching_test.train \
  --config configs/a2d_450gb_v3_hybrid_hand_target_cfm_a800_100ep_scratch.yaml
```

V3 纯 RGB 100 epochs：

```bash
python3 -u -m flow_matching_test.train \
  --config configs/a2d_450gb_v3_hybrid_hand_target_cfm_a800_rgb_only_100ep_scratch.yaml
```

训练目录包含：

```text
config_resolved.yaml
metrics.jsonl
latest.ckpt
best_val_loss.ckpt
best_action_mse.ckpt
best_ema_action_mse.ckpt
summary.json
```

实时查看每轮指标：

```bash
tail -F /path/to/run/metrics.jsonl
```

## 导出部署 bundle

优先按 rollout 目标选择 action-MSE checkpoint，不要默认最后一轮最好：

```bash
python3 -m flow_matching_test.export_bundle \
  --ckpt /path/to/best_ema_action_mse.ckpt \
  --out /path/to/eval_bundle \
  --weights-variant ema \
  --execute-horizon 16 \
  --data-version DATA_VERSION \
  --archive
```

Bundle manifest 会记录 action 语义、raw/EMA 权重、checkpoint 选择标准、训练 epoch/step、数据版本和内部文件 SHA256。部署 bundle 不是可恢复训练的完整 checkpoint。

## 验证

```bash
python3 -m pytest -q
git diff --check
```

数据更新时至少核对：

- episode、split 与 manifest SHA；
- action 语义和 13 维 layout；
- base/effective samples、tail/lift/transition windows；
- steps/epoch、total steps 与 warmup；
- checkpoint 与 bundle provenance。

## 文档与入口

- [`CHANGE.md`](CHANGE.md)：按时间倒序记录重要更新；
- [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md)：数据、split、padding、oversampling 与标签契约；
- [`docs/TRAINING_PLANNING_GUIDE.md`](docs/TRAINING_PLANNING_GUIDE.md)：epochs、steps、warmup 与 LR schedule；
- [`docs/TRAINING_TRICKS_GUIDE.md`](docs/TRAINING_TRICKS_GUIDE.md)：训练技巧与停止/续训判断；
- [`docs/POLICY_COMPARISON_PROTOCOL.md`](docs/POLICY_COMPARISON_PROTOCOL.md)：多 policy 公平对比；
- [`artifacts_index.md`](artifacts_index.md)：外部模型和评测产物索引；
- [`train+deploy/handoff_runbook.md`](train+deploy/handoff_runbook.md)：部署交付流程；
- `flow_matching_test/a2d_dataset.py`：训练 Dataset；
- `flow_matching_test/policies/`：policy 接口与实现；
- `flow_matching_test/train.py`：训练、验证、EMA 与 checkpoint；
- `flow_matching_test/export_bundle.py`：自包含部署包；
- `rollout/`：Isaac Sim rollout 与结果汇总。

训练数据、checkpoint、W&B 目录和 bundle 本体不进入 Git；只在 `artifacts_index.md` 中登记位置、SHA256、数据版本与 Git commit。
