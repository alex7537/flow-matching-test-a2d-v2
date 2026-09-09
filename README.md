# A2D Continuous-Action Policy Research

本仓库围绕 A2D 机器人抓取任务比较四条连续动作生成路线：Conditional Flow Matching、Diffusion Policy、IMLE，以及训练期加入未来视频 latent 监督的 Video-Aux 世界模型原型。共同目标是根据双相机 RGB 和可选 proprio，生成未来 16 步、每步 13 维的绝对关节目标。

`main` 是数据契约、共享模型组件、训练入口和路线总览；专项分支只维护各自实验，不要求 README 完全相同。

## 共同契约

```text
observation:
  rgb_head[t]
  rgb_right_hand[t]
  proprio[t] = arm actual(7) + hand actual(6)    # 可关闭

V3 action label:
  arm actual(7) + hand commanded target(6)

time alignment:
  obs[t] -> action[t+1:t+17]                     # Python半开区间，共16步

output:
  action chunk [B,16,13]
```

共享视觉/动作主干：

```text
每路 RGB [B,1,3,224,224]
        ↓ 共享 Hybrid ViT: vit_small_r26_s32_224
ResNetV2-26 → [B,2048,7,7]
        ↓ 1×1 Conv2d: 2048→384
49个 patch embeddings + 1个可学习 CLS
        ↓ ViT Transformer
默认保留每路49个上下文化 spatial tokens

两路 spatial: 49 + 49
+ 一个可选 proprio token: 1
= 99个 observation-condition tokens
        ↓
4层 Action Transformer
d_model=384, n_head=4
        ↓
连续 action chunk [16,13]
```

`7×7` 是图像二维特征网格，不是分割结果或机器人 XYZ 空间；49表示空间位置数，384才是每个位置的特征维度。ViT 内部同时存在 CLS，但默认 `spatial` 路线在完整 ViT 处理后丢弃 CLS。CLS 消融改为每路只传1个全局 token，因此 condition 从99降为3；ViT 内部仍会计算所有 patch。完整形状、代码入口和替代 encoder 边界见 [`docs/VISION_ENCODER.md`](docs/VISION_ENCODER.md)。

数据侧共同使用 exact-dedup keep-last、tail padding + `action_mask`、train-only transition/lift oversampling、episode-level split、deterministic validation、EMA、watchdog 和原子 checkpoint。

## 四条训练路线

| 路线 | 学习目标 | 推理 | 当前状态 |
|---|---|---|---|
| CFM baseline → enhanced-proprio | 回归 noise→action 的 velocity；后训练阶段增强状态条件 | 5步 Euler/ODE | 主线；旧 V3 已完成，新 b23v2 正在训练 |
| Diffusion Policy | 回归加入动作的 Gaussian noise | 15步 DDIM | 旧 V3 已完成正式训练 |
| IMLE | 每个 GT 从多 latent 候选中选择最近 action 回归 | latent 一步生成 | 旧 V3 已完成正式训练 |
| Video-Aux 世界模型原型 | CFM action loss + 冻结 Wan VAE future-latent loss | 部署仍只运行5步 CFM | 实现与 smoke 通过，尚未正式长训 |

不同路线的 loss 数值没有直接可比性。比较时必须锁定 dataset/split、action contract、ViT、batch、optimizer steps、seed、rollout 初始状态和成功判据，最终由同协议闭环 rollout 仲裁。

### 1. CFM baseline 与 enhanced-proprio

CFM baseline 采样 `noise` 与 `t∈(0,1]`：

```text
a_t = (1-t) * noise + t * action
velocity_target = action - noise
L_cfm = masked_MSE(predicted_velocity, velocity_target)
```

推理从 Gaussian action noise 开始，执行 5 次 Euler 更新。当前标准 proprio 只包含当前 13 维实际 qpos，没有显式 velocity/acceleration/jerk smoothing、low-pass filter 或 temporal ensemble。

Enhanced-proprio 是同一路线的第二阶段，不是第五种 policy。它从 CFM best checkpoint 做 model-only initialization，并把以下投影零初始化后叠加到同一个 proprio token：

```text
joint_delta
previous_action
hand_tracking_error
```

### 2. Diffusion Policy

DP 在离散噪声等级上构造 noisy action，网络预测 Gaussian noise：

```text
100个训练噪声等级
scaled-linear beta: 0.001 -> 0.2
15步DDIM推理
L_dp = MSE(predicted_noise, sampled_noise)
```

逐步去噪不等于显式动作平滑；当前同样没有 jerk、acceleration 或低通损失。

### 3. IMLE

IMLE 对每个 observation 采样多个 latent，生成候选 action chunk，并只回归离 GT 最近的候选：

```text
z_1 ... z_20 ~ N(0,I)
a_j = f(observation, z_j)
j* = argmin_j distance(a_j, GT)
L_imle = MSE(a_j*, GT)
```

它通过 latent 候选覆盖多模态动作，不进行 CFM/DP 的多步去噪。

### 4. Video-Aux 世界模型原型

专项分支在 CFM 上增加：

```text
9帧过去 head RGB + 16帧未来 head RGB
        ↓ frozen Wan2.2 VAE
3个 condition latent + 4个 future latent

L_total = L_cfm + 0.01 * L_video
```

当前只是训练期辅助目标：部署不加载 Wan VAE、不生成视频。正式长训前仍需完成离线 latent cache 与 `λ`/梯度比例消融。

## 数据版本

### A2D 1,090-episode V3

```text
episodes                 1,090
V3 exact-dedup frames    179,160
train / val              981 / 109 episodes
effective train samples  220,026
steps/epoch @ batch32    6,876
```

该数据已用于 CFM、enhanced-proprio、DP 和 IMLE 的历史实验。

### b23v2 654-episode V3：2026-09-01 新任务

```text
dataset version          b23v2_rgb_v3_arm_executed_hand_commanded_exact_dedup_keep_last
episodes                 654
frames before dedup      109,530
frames removed           580
frames after dedup       108,950
train / val              589 / 65 episodes
base train windows       97,560
effective train samples  132,759
val samples              10,736
steps/epoch @ batch32    4,149
epochs                   100
total optimizer steps    414,900
warmup steps             20,740 (5%)
```

当前实验保持旧 V3 CFM baseline 架构不变，只替换成新的任务数据分布：双 RGB + 当前 proprio、从零训练、ViT `0.1×` LR、transition/lift `2×`、无 enhanced-proprio、无显式动作平滑。

- 配置：[`configs/a2d_b23v2_v3_hybrid_hand_target_cfm_a800_100ep_scratch.yaml`](configs/a2d_b23v2_v3_hybrid_hand_target_cfm_a800_100ep_scratch.yaml)
- W&B：[online run `z61net93`](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/z61net93)
- 启动时间：2026-09-01；README 只记录已启动事实，最终结论以完成后的 `summary.json`、checkpoint provenance 和 rollout 为准。

数据 HDF5、checkpoint、W&B 本地目录和 deployment bundle 不进入 Git。

## 分支职责

| 分支 | README/代码职责 |
|---|---|
| `main` | 四条训练路线总览、共享数据/训练/部署契约 |
| [`feat/v3-wan-video-aux-v1`](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/feat/v3-wan-video-aux-v1) | 冻结 Wan VAE 的 9+16 future-video auxiliary 原型 |
| [`feat/task-level-grasp-retry`](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/feat/task-level-grasp-retry) | 推理侧 attempt→verify→recover→retry 状态机；不是新训练 policy |
| [`docs/grasp-success-gallery`](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/docs/grasp-success-gallery) | 成功视频、GIF gallery 与 rollout 展示 |
| [`agent/add-robot-ml-loop-instance`](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/agent/add-robot-ml-loop-instance) | robot-ML loop 实例与生命周期编排实验 |

DP 与 IMLE 当前是仓库内的策略路线，而不是两个独立远端产品分支；不要为了 README 复制出空分支。

## 训练

新 b23v2 CFM：

```bash
python3 -u -m flow_matching_test.train \
  --config configs/a2d_b23v2_v3_hybrid_hand_target_cfm_a800_100ep_scratch.yaml
```

历史 1,090-episode V3 CFM：

```bash
python3 -u -m flow_matching_test.train \
  --config configs/a2d_450gb_v3_hybrid_hand_target_cfm_a800_100ep_scratch.yaml
```

训练产物：

```text
config_resolved.yaml
metrics.jsonl
latest.ckpt
best_val_loss.ckpt
best_action_mse.ckpt
best_ema_action_mse.ckpt
summary.json
failure.json（仅失败时）
```

部署优先比较 `best_action_mse`、`best_ema_action_mse` 与必要的 periodic/latest checkpoint；bundle 不包含 optimizer，不能用于完整 resume。

## 文档

- [`CHANGE.md`](CHANGE.md)：倒序更新记录；
- [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md)：JPEG 预处理、V3 action、dedup、padding、mask 与 sampling；
- [`docs/TRAINING_PLANNING_GUIDE.md`](docs/TRAINING_PLANNING_GUIDE.md)：epochs、steps、warmup 与 LR；
- [`docs/TRAINING_TRICKS_GUIDE.md`](docs/TRAINING_TRICKS_GUIDE.md)：训练技巧、监控与停止判断；
- [`docs/VISION_ENCODER.md`](docs/VISION_ENCODER.md)：Hybrid ViT、ResNet特征图、spatial/CLS token与替代视觉encoder合同；
- [`artifacts_index.md`](artifacts_index.md)：外部 checkpoint、bundle 与实验产物索引。
