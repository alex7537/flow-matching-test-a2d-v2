# A2D CFM + Wan Future-Video Auxiliary V1

> **分支职责：**本 README 只描述 `feat/v3-wan-video-aux-v1` 世界模型原型。四条策略路线、共享数据契约和当前正式训练记录见 [`main` 总 README](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/main)。

本分支 `feat/v3-wan-video-aux-v1` 专门验证一件事：在已经可用的 A2D Conditional Flow Matching 动作策略上，加入冻结 Wan2.2 VAE 的未来视频 latent 辅助监督，是否能让策略形成更强的动作后果表征，并最终提高抓取 rollout 成功率。

它是一个 **action policy + future-video auxiliary loss**，不是完整的联合视频—动作 WAM。训练时使用视频监督；部署时仍然只输入双相机 RGB 与 proprio，只输出 16 步关节动作，不生成视频、不加载 Wan VAE。

当前状态：在线原型和离线 cached-latent 后训练链路均已实现；全仓测试、真实 Wan VAE 和一批 train+val smoke 已通过。当前等待 box300+bottle300 混合任务 CFM 训练结束，再用它的 checkpoint 做 model-only post-training；尚未启动世界模型正式长训。

## 1. 数据契约

以当前时间点 `t` 为锚点：

| 数据 | 时间范围 | 形状 | 用途 |
|---|---|---:|---|
| `rgb_head[t]` | 当前帧 | `[B,1,3,224,224]` | 动作 observation |
| `rgb_right_hand[t]` | 当前帧 | `[B,1,3,224,224]` | 动作 observation |
| proprio | 当前实际 13 维 joint state | token dim `384` | 动作 observation |
| action GT | `action[t+1:t+17]` | `[B,16,13]` | CFM velocity 监督 |
| video condition | `rgb_head[t-8:t+1]` | `[B,9,3,224,224]` | Wan 过去视频条件 |
| video future GT | `rgb_head[t+1:t+17]` | `[B,16,3,224,224]` | 未来视频 latent 监督 |

上述区间采用 Python 半开区间，因此 `t+1:t+17` 正好包含 16 帧。

动作 label 继续使用 V3 hybrid 语义：

```text
arm target  = 实际执行的 7 维 arm joint
hand target = 下发的 6 维 hand commanded joint target
```

## 2. 系统架构

```text
                           ┌──────────────────────────────┐
rgb_head[t] ──────────────►│                              │
rgb_right_hand[t] ────────►│ ViT + proprio token encoder │
proprio/enhanced proprio ─►│                              │
                           └──────────────┬───────────────┘
                                          │
                              observation tokens [B,N,384]
                                          │
                     ┌────────────────────┴────────────────────┐
                     │                                         │
                     ▼                                         ▼
          CFM action objective                     future-video objective
                                                               │
noise/action interpolation [B,16,13]       rgb_head[t-8:t+17], 25 frames
                     │                                         │
             Action Transformer                         frozen Wan2.2 VAE
                     │                                         │
       predicted velocity [B,16,13]              latent [B,48,7,14,14]
                     │                              ├─ condition: 3 steps
            masked action MSE                      └─ future GT: 4 steps
                     │                                         │
                     │                       last condition latent
                     │                       + clean-action context
                     │                       + 4 step embeddings
                     │                                         │
                     │                           Future Latent Head
                     │                                         │
                     │                      predicted future latent
                     │                           [B,48,4,14,14]
                     │                                         │
                     │                              video latent MSE
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          │
                         L_total = L_action + λ_video L_video
```

核心实现位于：

- [`flow_matching_test/policies/video_aux.py`](flow_matching_test/policies/video_aux.py)：冻结 Wan codec、Future Latent Head 与双损失；
- [`flow_matching_test/policies/flow_matching.py`](flow_matching_test/policies/flow_matching.py)：共享 observation tokens 和 Action Transformer features；
- [`flow_matching_test/a2d_dataset.py`](flow_matching_test/a2d_dataset.py)：9+16 视频窗口、cached latent、动作 mask 与视频 mask；
- [`scripts/precompute_wan_video_latents.py`](scripts/precompute_wan_video_latents.py)：可断点复用的离线 Wan latent cache；
- [`configs/a2d_v3_multitask_video_aux_posttrain_10ep.yaml`](configs/a2d_v3_multitask_video_aux_posttrain_10ep.yaml)：混合任务后训练配置。

## 3. CFM 动作目标

真实动作记为 `a`，随机高斯噪声记为 `ε`，随机时间为 `t∈(0,1]`：

```text
a_t = (1-t) ε + t a
velocity_target = a - ε
L_action = masked_MSE(v_pred(a_t, observation, t), a - ε)
```

`action_mask` 会排除 episode 尾部重复 padding，只让真实 action timestep 参与动作损失。

## 4. 未来视频目标

Wan VAE 的时间压缩满足：

```text
latent_steps = 1 + (rgb_frames - 1) / 4
25 RGB frames -> 7 latent steps
9 condition frames -> 3 latent steps
16 future frames -> 4 latent steps
```

V1 从三个 condition latent 中取最后一个 `[B,48,14,14]`，再融合：

- 同一组 observation tokens；
- clean action 经过共享 Action Transformer 得到的 context；
- 四个未来 latent timestep embedding。

Future Latent Head 输出 `[B,48,4,14,14]`，与冻结 Wan VAE 产生的未来 latent GT 计算 MSE：

```text
L_video = MSE(z_future_pred, stop_gradient(z_future_gt))
```

Wan VAE 全程 `no_grad`，不进入 optimizer、EMA、checkpoint 或部署 bundle。

## 5. Loss 权重

当前 V1：

```text
L_total = 1.0 * L_action + 0.01 * L_video
```

`0.01` 是名义系数，不代表视频分支只有动作分支 1% 的实际影响。真实一步 smoke：

```text
L_action                 = 0.03347
L_video                  = 0.93516
0.01 * L_video           = 0.00935
L_total                  = 0.04282

action : weighted video  ≈ 3.58 : 1
video 占 total loss      ≈ 21.8%
```

如果直接设置 `λ_video=0.1`，同一批数据上的加权视频 loss 将约为 `0.0935`，约是动作 loss 的 `2.8×`，可能让辅助任务反过来主导动作训练。

若希望某个 batch 上的 loss 数值贡献满足 `action:video = R:1`：

```text
λ_video = L_action / (R * L_video)
```

按当前 smoke，实际贡献目标为 `10:1` 时，`λ_video≈0.0036`。正式消融建议先比较：

```text
λ_video ∈ {0.003, 0.01, 0.03}
```

loss 数值占比不等于梯度占比；正式实验还需分别记录 action/video loss 对共享 Transformer 与 ViT 的梯度范数。

## 6. 梯度归属

| 模块 | Action loss | Video loss | 状态 |
|---|---:|---:|---|
| ViT / observation encoder | ✓ | ✓ | 训练，backbone LR `0.1×` |
| proprio adapters | ✓ | ✓ | 训练 |
| Action Transformer | ✓ | ✓ | 训练 |
| CFM velocity head | ✓ | — | 训练 |
| Future Latent Head | — | ✓ | 从零训练 |
| Wan2.2 VAE | — | — | 冻结、外置 |

这个辅助目标的真正作用，是把“未来视觉是否合理”的梯度传回共享 observation encoder 与 Action Transformer，而不是训练 Wan VAE。

## 7. 尾部与 mask

动作和视频使用独立有效性规则：

```text
action_mask:
  每个未来 action timestep 是否真实

video_valid_mask:
  是否存在完整的 16 个真实未来视频帧
```

若 episode 尾部只剩若干真实动作：

- 真实动作仍参与 `L_action`；
- padding 动作由 `action_mask` 排除；
- 视频张量重复最后一帧保持固定形状；
- 整个视频目标由 `video_valid_mask=false` 排除，不把 padding 当作未来 GT。

## 8. 训练与推理边界

### 训练

```text
双 RGB + proprio
+ 9 帧过去 head RGB
+ 16 帧未来 head RGB
+ 16 步 action GT
        ↓
CFM action loss + Wan future latent loss
```

### 推理/部署

```text
当前双 RGB + proprio
        ↓
CFM 从随机动作噪声开始进行 5 步 Euler/ODE 积分
        ↓
输出 action chunk [16,13]
```

推理路径不读取 9+16 视频窗口、不运行 Wan VAE、不调用 Future Latent Head，也不生成未来视频。因此它仍然是 action policy，而不是可递归 rollout 的完整世界模型。

## 9. 环境边界

动作训练环境必须保持 Python 包优先级。Wan runtime 及其 site-packages 只在 codec 首次使用时追加加载。

不要把整个 WAM 虚拟环境放到训练进程 `PYTHONPATH` 最前面；已经验证过的失败包括：

- 旧 NumPy 抢先加载导致 checkpoint 报 `ModuleNotFoundError: numpy._core`；
- WAM OpenCV 抢先加载导致 `ImportError: libGL.so.1`。

## 10. 当前验证状态

- 全仓测试：`55 passed`；
- 真实 Wan VAE：25 帧成功编码为 7 个 latent timestep；
- 真实 V3 数据：9+16 窗口、尾部 mask 和动作 mask 均通过；
- CFM checkpoint 升级：只允许新 `video_aux_head.*` 缺失，model-only warm start 成功；
- 一步 A800 train + val：前向、反向和指标记录成功；
- bundle round-trip：推理不依赖 Wan VAE。

详细验证边界见 [`docs/WAN_VIDEO_AUX_V1.md`](docs/WAN_VIDEO_AUX_V1.md)。

## 11. 运行

在线原型配置（只用于 smoke）：

```bash
python3 -u -m flow_matching_test.train \
  --config configs/a2d_450gb_v3_video_aux_cfm_a800_v1.yaml
```

正式后训练先在空闲 A800 上生成约 84,771 个有效窗口（未压缩 fp16 约 7.43 GiB）：

```bash
bash scripts/precompute_multitask_wan_latents.sh
```

混合任务 CFM 训练完成后，只需传入选定 checkpoint：

```bash
bash scripts/launch_multitask_wan_video_aux_posttrain.sh \
  /absolute/path/to/best_action_mse.ckpt
```

这是 model-only init：继承模型权重，新建 video head，并重新建立 optimizer、5% warmup 与 10-epoch cosine schedule；不是从零训练，也不是恢复旧 optimizer/scheduler 的 resume。

## 12. 下一步实验

1. 等当前混合 CFM 结束后，在空闲 A800 生成完整 latent cache 并做一批 cached train+val smoke；
2. 用选定 mixed-CFM checkpoint 启动 `λ=0.01` 的 10-epoch 后训练；
3. 增加 action/video 分别作用于共享网络的梯度范数日志；
4. 以相同 checkpoint、数据、step budget 比较 `λ={0,0.003,0.01,0.03}`；
5. 同协议比较 action MSE、lift 指标与闭环抓取成功率；只有辅助分支证明有效后，再决定是否升级为联合视频—动作 Flow Matching WAM。
