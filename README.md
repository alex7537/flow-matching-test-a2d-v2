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

## Wan 未来视频辅助分支

`feat/v3-wan-video-aux-v1` 在原有 CFM 动作策略上增加训练期的未来视频 latent 监督，但不改变部署输入和动作输出：

```text
动作输入：rgb_head[t] + rgb_right_hand[t] + proprio[t]
动作目标：action[t+1:t+17]（Python 半开区间）         [B,16,13]

视频条件：rgb_head[t-8:t]                          9 帧
视频目标：rgb_head[t+1:t+17]（Python 半开区间）      16 帧
              │
              ▼ 冻结 Wan2.2 VAE
25 帧 latent                                          [B,48,7,14,14]
├── condition latent                                  3 步
└── future latent GT                                  4 步

observation tokens + clean action
              │
              ▼ 共享 Action Transformer + Future Latent Head
predicted future latent                               [B,48,4,14,14]
```

动作分支仍使用 CFM velocity 回归：

```text
a_t = (1-t) * noise + t * action
velocity_target = action - noise
L_action = MSE(predicted_velocity, velocity_target)
```

视频分支使用未来 latent MSE：

```text
L_video = MSE(predicted_future_latent, frozen_Wan_future_latent)
L_total = L_action + lambda_video * L_video
```

当前 V1 设置 `lambda_video=0.01`。这表示 loss **系数**为 `1:0.01`，不是 `1:0.1`；但系数比例不等于实际优化贡献。真实一步 smoke 得到：

```text
L_action                 0.03347
L_video                  0.93516
0.01 * L_video           0.00935
L_total                  0.04282

loss 数值贡献约为：
action : weighted video = 3.58 : 1
```

如果要求某个 batch 上的目标贡献比例为 `L_action : lambda*L_video = R : 1`，可用：

```text
lambda = L_action / (R * L_video)
```

按上述 smoke，若目标是实际贡献 `10:1`，`lambda≈0.0036`；若直接设置 `lambda=0.1`，加权视频 loss 约为 `0.0935`，会达到动作 loss 的约 `2.8×`。因此正式实验应至少比较 `0.003 / 0.01 / 0.03`，并同时观察两项独立梯度和 rollout，而不是只按名义系数判断。

Wan VAE 完全冻结，不进入 optimizer、EMA 或部署 bundle。视频辅助 head 和共享 Action Transformer/视觉 encoder 接收视频梯度；CFM velocity head 只接收动作梯度。推理时仍然只运行双 RGB/proprio 条件的五步 CFM，不加载 Wan VAE，也不生成视频。

尾部样本继续通过 `action_mask` 保留真实动作监督；不足 16 个真实未来视频帧时，`video_valid_mask=false`，该样本的视频 loss 为零，避免把重复 padding 当成未来 GT。

该分支的在线 Wan VAE 编码只用于 smoke。正式长训前应预计算冻结 latent，否则每个 epoch 都会重复解码视频和运行 VAE。

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
9. enhanced proprio：在原 qpos token 上零初始化叠加 joint delta、previous action 与 hand target-actual error，支持从 V3 best checkpoint 兼容 warm start。

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

Enhanced proprio 20-epoch fine-tune：

```bash
python3 -u -m flow_matching_test.train \
  --config configs/a2d_450gb_v3_enhanced_proprio_cfm_a800_20ep_init_best.yaml
```

Wan 未来视频辅助 V1 smoke/原型：

```bash
python3 -u -m flow_matching_test.train \
  --config configs/a2d_450gb_v3_video_aux_cfm_a800_v1.yaml
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
- [`docs/ENHANCED_PROPRIO.md`](docs/ENHANCED_PROPRIO.md)：动态 proprio 输入、warm start 与训练预算；
- [`docs/V3_DIFFUSION_SCALED_LINEAR.md`](docs/V3_DIFFUSION_SCALED_LINEAR.md)：V3 DP scaled-linear schedule、监控与匹配预算；
- [`docs/WAN_VIDEO_AUX_V1.md`](docs/WAN_VIDEO_AUX_V1.md)：冻结 Wan VAE 的 9+16 视频辅助目标、mask 与运行边界；
- [`artifacts_index.md`](artifacts_index.md)：外部训练和部署产物索引。

训练数据、checkpoint、W&B 目录和 bundle 本体不进入 Git。
