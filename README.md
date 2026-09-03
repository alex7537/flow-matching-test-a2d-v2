# A2D Joint Latent WAM — Scratch V1

> **分支职责：**`feat/a2d-v3-joint-latent-wam-scratch-v1` 在个人
> `alex7537/flow-matching-test-a2d-v2` 仓库内实现独立的紧凑型视频—动作联合
> Flow Matching。它不加载旧 CFM checkpoint，也不依赖公司 `psi-policy` 源码。

这条路线与父分支 `feat/v3-wan-video-aux-v1` 不同：父分支是在训练好的 CFM
上增加 future-video MSE；本分支让 future-video latent 和 action 同时从噪声生成，
并通过同一组 Transformer self-attention 相互作用。

当前状态：本地代码、cache V2、双 loss、联合 ODE 和梯度测试完成；全仓测试
`61 passed`，并用真实600条数据索引和临时latent完成CPU trainer单步集成。
尚未生成真实Wan cache，尚未连接A800，也未启动训练。

## 数据合同

以锚点 `t` 为中心：

| 张量 | 时间/形状 | 角色 |
|---|---|---|
| 当前双 RGB | `[B,1,3,224,224] × 2` | 当前视觉条件 |
| 当前 proprio | `[B,13]` | 当前机器人状态条件 |
| head RGB history | `t-8:t`，9帧 | Wan因果视频前缀 |
| action GT | `t+1:t+16`，`[B,16,13]` | action Flow目标 |
| future head RGB | `t+1:t+16`，16帧 | future-video latent目标 |

动作仍使用V3语义：arm为7维executed joint，hand为6维commanded target。

公开Wan2.2 VAE把25帧压缩成：

```text
condition_prefix [B,48,3,14,14]
future_target    [B,48,4,14,14]
```

VAE只在离线cache阶段运行，训练时读取float16 latent。Cache manifest绑定数据
manifest SHA、episode content hash、VAE SHA、预处理和9+16时序合同。

## 模型

```text
condition video prefix ──► latent patch tokens ─┐
current dual RGB + proprio ─► condition tokens ├─► cross-attention condition
                                                │
noisy future-video latent ─► 196 video tokens ─┐│
                                               ├┴─► shared Transformer blocks
noisy action [16,13] ──────► 16 action tokens ─┘
                                                    ├─► video velocity
                                                    └─► action velocity
```

Future-video token和action token在shared self-attention中互相读取。已知视频前缀、
双RGB和proprio作为condition被cross-attention读取。

该模型使用本仓库的小型`d_model=384` Transformer，不是复制Wan TI2V-5B DiT。
Wan只提供公开、冻结的视频latent codec。

## 双Flow Matching

共享采样时间 `τ`，两种模态使用独立高斯噪声：

```text
z_τ = (1-τ) ε_video  + τ z_future
a_τ = (1-τ) ε_action + τ a_future

target_video_velocity  = z_future - ε_video
target_action_velocity = a_future - ε_action
```

损失独立计算后相加：

```text
L_total = λ_action L_action_flow + λ_video L_video_flow
```

Smoke配置暂用`λ_action=1.0, λ_video=0.1`。正式权重必须依据初始loss量级、
共享Transformer梯度范数和rollout结果校准，不能直接沿用video-aux的`0.01`。

## 尾部窗口

- action继续使用逐帧`action_mask [16]`；
- video使用逐latent-step`video_future_mask [4]`；
- 只有完全由4个真实未来帧构成的latent step参与video loss；
- 无效future-video tokens被self-attention key padding mask隔离，不能污染action；
- 因此最后的lift action仍可参与训练，而padding视频不成为监督目标。

## Scratch边界

```yaml
training:
  init_from: null
  resume_from: null
```

从零初始化Joint Transformer和video/action heads。当前RGB ViT仍可使用ImageNet
预训练；Wan VAE使用公开预训练权重并冻结。因此“scratch”指不继承任何A2D
policy/CFM权重，不是随机初始化视觉基础模型和视频codec。

## 入口

- `flow_matching_test/policies/joint_wam.py`：Joint Transformer、双loss和联合ODE；
- `flow_matching_test/a2d_dataset.py`：cache V2、tail mask和训练batch；
- `scripts/precompute_wan_joint_latents.py`：公开Wan2.2 VAE latent cache；
- `configs/a2d_v3_multitask_joint_wam_scratch_smoke.yaml`：受限smoke配置；
- `configs/a2d_v3_multitask_joint_wam_scratch_100ep.yaml`：正式100-epoch配置；
- `scripts/queue_joint_wam_after_cfm.py`：前序任务完成后依次执行cache、三级smoke和正式训练的一次性状态机；
- `tests/test_joint_wam_policy.py`：双loss、跨模态梯度、联合采样；
- `tests/test_joint_wam_latent_cache.py`：cache、尾部和数据绑定。

离线cache命令：

```bash
bash scripts/precompute_multitask_wan_joint_latents.sh
```

受限smoke命令：

```bash
python -m flow_matching_test.train \
  --config configs/a2d_v3_multitask_joint_wam_scratch_smoke.yaml
```

## 剩余门禁

1. 等当前主线训练释放开发机GPU；
2. 在隔离代码快照中安装公开`Wan-Video/Wan2.2`；
3. 验证真实`9+16 → 3+4` latent及cache断点续算；
4. 运行1-step、100-step、1-epoch smoke并检查显存、NaN、两个loss与梯度；
5. 实现rollout侧9帧历史编码后，才允许导出部署bundle；
6. 根据smoke吞吐和有效样本数规划正式训练预算。

本分支不会自动推送。任何公司remote操作必须经过`corporate-git-push-gate`
展示目标后，再由用户进行一次独立确认。
