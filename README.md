# Flow Matching Test

基于 A2D 机器人抓取数据训练连续关节动作策略。当前主线是双 RGB、可选 proprio 条件的 Flow Matching，输出未来 16 步绝对 joint action。

## 数据

当前使用 `a2d-450GB` 的 1,090 个成功 episodes：

```text
原始帧数             180,087
V3 exact-dedup 后    179,160
train / val          981 / 109 episodes
基础训练窗口          160,206
有效训练样本          220,026
验证样本              17,864
```

V3 数据语义：

```text
observation/qpos = arm2_pos(7) + hand2_pos(6)
                  实际手臂 joint + 实际手部 joint

action label     = arm2_pos(7) + hand2_pos_target(6)
                  实际手臂轨迹 + 手部 commanded target
```

V2 的手部标签使用实际 `hand2_pos`，包含接触、回弹和跟踪误差。V3 改为学习 `hand2_pos_target`，让模型学习稳定的手部控制意图；observation 仍使用机器人当前的实际 joint state。

## 输入与输出

当前有两组受控实验：

| 实验 | 输入 | 输出 |
|---|---|---|
| RGB+proprio | `rgb_head`、`rgb_right_hand`、当前 13 维实际 qpos | `[16,13]` action chunk |
| RGB-only | `rgb_head`、`rgb_right_hand` | `[16,13]` action chunk |

时间对齐：

```text
obs[t]   -> action[t+1 : t+17]
obs[t+1] -> action[t+2 : t+18]
```

窗口 stride 为 1，相邻训练样本共享 15 个未来 action。

## 模型与策略

- 视觉 encoder：ImageNet 预训练 `vit_small_r26_s32_224`；
- 两路相机共用 ViT，每路保留 49 个 spatial tokens；
- proprio 版本将归一化 13 维 qpos 投影为一个 384 维 token；
- action model：4 层 Transformer，`d_model=384`、`n_head=4`；
- policy：Conditional Flow Matching；训练预测 velocity，推理使用 5 步 Euler 积分；
- ViT 全量可训练，backbone LR 是 action head LR 的 `0.1×`；
- EMA 权重用于独立 checkpoint 选择与 rollout 对比。

仓库同时保留 RS-IMLE 和 Diffusion Policy 实现，但当前抓取主线使用 Flow Matching。不同 policy 的 loss 数值不能直接比较，最终以同协议 rollout 为准。

## 训练预算

RGB+proprio 与 RGB-only 使用相同数据和预算，只改变 `use_proprio`：

```text
batch size             32
steps / epoch          6,876
epochs                 100
total steps            687,600
warmup steps           34,380（前5轮）
head peak LR           1e-4
ViT peak LR            1e-5
schedule               linear warmup + cosine decay
seed                   42
```

Sampling：

```text
transition windows     16,520，train 2×
lift windows           43,300，train 2×
tail padded windows    train 14,715 / val 1,635
```

训练集启用图像 augmentation；验证集不增强、不 oversample，并使用固定 seed。每轮记录 train/val、continuous/keyframe/lift loss、sample action MSE、EMA 指标、梯度和学习率。

## 已完成的关键改动

1. `action_offset_steps=1`：当前 observation 预测下一帧开始的 16 步动作；
2. exact-dedup keep-last：删除完全重复的 joint frame，同时保留重复段最后一帧；
3. tail padding + `action_mask`：保留 episode 最后 15 个窗口，padding 不参与 loss，避免丢失最终 lift；
4. lift/transition oversampling：增加关键动作在训练中的曝光；
5. V3 hybrid action：arm 学习实际平滑轨迹，hand 学习 commanded target；
6. action contract：dataset、stats、checkpoint、bundle、rollout 均校验 V2/V3 语义；
7. `use_proprio` 开关：在完全相同预算下比较 RGB+proprio 与纯 RGB；
8. deterministic validation、EMA、watchdog 和原子 checkpoint 保存。
9. task-level grasp retry：在 chunk-level replanning 外增加 `attempt → verify → recover → re-attempt`；未接近或没有形成稳定多指接触时恢复到安全预抓取位，清空 policy 历史并更换 sampling seed 后重抓。默认关闭，校准恢复位后启用。

详细历史见 [`CHANGE.md`](CHANGE.md)。

## 训练

RGB+proprio：

```bash
python3 -u -m flow_matching_test.train \
  --config configs/a2d_450gb_v3_hybrid_hand_target_cfm_a800_100ep_scratch.yaml
```

纯 RGB：

```bash
python3 -u -m flow_matching_test.train \
  --config configs/a2d_450gb_v3_hybrid_hand_target_cfm_a800_rgb_only_100ep_scratch.yaml
```

重要产物：

```text
metrics.jsonl
latest.ckpt
best_val_loss.ckpt
best_action_mse.ckpt
best_ema_action_mse.ckpt
summary.json
```

部署时优先测试 action-MSE checkpoint，不默认最后一轮最好。部署 bundle 会记录 action 语义、raw/EMA 权重、epoch/step、数据版本及 SHA256；bundle 不包含 optimizer，不能用于完整 resume。

## 文档

- [`CHANGE.md`](CHANGE.md)：倒序更新记录；
- [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md)：数据、padding、mask 与 oversampling；
- [`docs/TRAINING_PLANNING_GUIDE.md`](docs/TRAINING_PLANNING_GUIDE.md)：epochs、steps、warmup 与 LR；
- [`docs/TRAINING_TRICKS_GUIDE.md`](docs/TRAINING_TRICKS_GUIDE.md)：训练与停止判断；
- [`docs/TASK_LEVEL_GRASP_RETRY.md`](docs/TASK_LEVEL_GRASP_RETRY.md)：task-level 抓取重试、恢复与安全门禁；
- [`artifacts_index.md`](artifacts_index.md)：外部训练和部署产物索引。

训练数据、checkpoint、W&B 目录和 bundle 本体不进入 Git。
