# A2D Joint Latent WAM

本分支从零训练一个紧凑型视频—动作联合Flow Matching模型：它读取当前双相机RGB、
当前关节状态和过去9帧头部视频，在同一个Transformer中联合生成未来16帧头部视频的
latent与未来16步、每步13维的关节动作。

Wan2.2只提供冻结的视频VAE。真正学习视频动力学和动作策略的是本仓库的
`d_model=384` Joint Transformer，不是Wan 5B DiT。

## 一眼看懂完整架构

```text
已知条件（模型可以看见）
────────────────────────────────────────────────────────────────────
当前head RGB [B,1,3,224,224]
      └─ shared Hybrid ViT ───────────────────→ [B,49,384] ─┐
当前right-hand RGB [B,1,3,224,224]                           │
      └─ shared Hybrid ViT ───────────────────→ [B,49,384] ─┤
当前actual proprio [B,13]                                   │
      └─ Linear(13→384) + LayerNorm ──────────→ [B,1,384] ──┤
过去9帧head RGB [B,9,3,224,224]                             │
      └─ frozen Wan2.2 VAE                                  │
         → latent [B,48,3,14,14]                            │
      └─ Conv2d(48→384,k=2,s=2)                             │
         → 3×7×7 = 147 tokens [B,147,384] ──────────────────┤
                                                            ↓
                                       condition [B,246,384]

需要生成（训练时有GT，推理时从Gaussian noise开始）
────────────────────────────────────────────────────────────────────
未来16帧head RGB
      └─ frozen Wan2.2 VAE → future latent [B,48,4,14,14]
      └─ Flow interpolation + Conv2d(48→384,k=2,s=2)
         → 4×7×7 = 196 video tokens [B,196,384] ────────────┐
未来动作GT [B,16,13]                                        │
      └─ Flow interpolation + Linear(13→384)                │
         → 16 action tokens [B,16,384] ─────────────────────┤
Flow timestep τ
      └─ sinusoidal embedding + MLP → [B,1,384]             │
                                                            ↓
                                   joint tokens [B,212,384]
                                                            │
                       ┌────────────────────────────────────┘
                       ↓
                Joint Transformer ×4
                hidden=384, heads=4, head_dim=96
                ├─ self-attention：video/action互相读取
                ├─ cross-attention：生成tokens读取condition
                └─ MLP
                       │
             final LayerNorm并按token位置切分
              ┌────────┴────────┐
              ↓                 ↓
       前196个video tokens    后16个action tokens
       Linear(384→192)        Linear(384→13)
       + PixelShuffle              │
              ↓                    ↓
 video velocity             action velocity
 [B,48,4,14,14]             [B,16,13]
```

## 输入、监督目标与随机量

| 类别 | 内容 | 形状 | 训练时来源 | 推理时来源 |
|---|---|---|---|---|
| 当前视觉条件 | head + right-hand RGB | 各`[B,1,3,224,224]` | 数据集 | Isaac当前观测 |
| 当前状态条件 | actual arm7 + hand6 | `[B,13]` | 数据集 | Isaac当前关节状态 |
| 视频历史条件 | 过去9帧head RGB | `[B,9,3,224,224]` | 离线VAE cache | 在线Wan VAE编码 |
| 动作目标 | arm actual7 + hand commanded6 | `[B,16,13]` | GT | 不存在，从noise生成 |
| 视频目标 | 未来16帧head RGB latent | `[B,48,4,14,14]` | GT cache | 不存在，从noise生成 |
| 动作随机量 | Gaussian action noise | `[B,16,13]` | 随机采样 | 随机初值 |
| 视频随机量 | Gaussian video noise | `[B,48,4,14,14]` | 随机采样 | 随机初值 |

视频历史和视频未来都只使用`rgb_head`。right-hand相机仍参与当前时刻的ViT视觉条件，
但没有进入9帧视频历史，也不是未来视频预测目标。

## 三类encoder如何变成统一token

### 1. 当前双相机：Hybrid ViT

两路相机共享同一个`timm vit_small_r26_s32_224`：

```text
RGB 224×224
→ ResNetV2-26卷积stem
→ feature map [B,2048,7,7]
→ 1×1 Conv2d投影到384通道
→ 49个spatial embeddings
→ ViT blocks上下文化
→ 每路保留49个[B,49,384] spatial tokens
```

`7×7`是图像空间网格，不是分割mask或机器人XYZ坐标。ViT使用ImageNet预训练权重，
训练时以主学习率的`0.1×`更新。

### 2. 当前proprio：线性投影

```text
[B,13]
→ Linear(13→384)
→ LayerNorm
→ [B,1,384]
```

这一步只负责把13维关节状态映射到Transformer的384维token空间；它不是数据归一化，
也不独立建模时间关系。

### 3. 过去头部视频：Wan VAE + patch embedding

```text
9帧head RGB
→ frozen causal Wan2.2 VAE
→ [B,48,3,14,14]
→ 每个latent时间步做Conv2d(kernel=2,stride=2,48→384)
→ [B,3×7×7,384] = [B,147,384]
```

这里的3是latent时间步，不是只取3张RGB。9帧先经过VAE约4倍时间压缩，再得到3个
latent steps；每个step的`14×14`latent再切成`7×7`个空间token。

## Transformer中的两套Q/K/V

每一层都按以下顺序执行：

```text
1. self-attention(joint, joint, joint)
2. cross-attention(joint, condition, condition)
3. MLP(joint)
```

### Self-attention：未来视频和动作互相影响

```text
Q = joint tokens [B,212,384]
K = joint tokens [B,212,384]
V = joint tokens [B,212,384]
```

模块内部通过不同的`W_Q/W_K/W_V`投影为4个head：

```text
Q/K/V → [B,4,212,96]
attention score → [B,4,212,212]
```

212个token包含196个future-video token和16个action token。因此：

- action query可以读取video key/value，动作会参考当前正在生成的视觉未来；
- video query可以读取action key/value，视频未来也可以参考当前正在生成的动作；
- 这就是“联合视频—动作模型”，而不是两个互不相关的head并排训练。

### Cross-attention：生成结果读取已知条件

```text
Q = 更新后的joint tokens [B,212,384]
K = condition tokens      [B,246,384]
V = condition tokens      [B,246,384]

attention score → [B,4,212,246]
```

246个condition token由`49 head + 49 hand + 1 proprio + 147 past-video`组成。每个未来
视频/动作token都可以选择读取不同的当前图像区域、关节状态和过去视频位置。

Cross-attention只直接更新212个生成token，condition token本身不会在该层被反向改写；
但训练梯度仍会通过K/V投影传回ViT、proprio projection和video patch embedding。

## 双Flow Matching与梯度

动作和视频共享同一个随机时间`τ`，但使用独立Gaussian noise：

```text
a_τ = (1-τ) ε_action + τ a_future
z_τ = (1-τ) ε_video  + τ z_future

target_action_velocity = a_future - ε_action
target_video_velocity  = z_future - ε_video
```

最终loss：

```text
L_total = 1.0 × L_action_flow + 0.1 × L_video_flow
```

| 模块 | action loss | video loss | 状态 |
|---|---:|---:|---|
| Wan2.2 VAE | 否 | 否 | 公开预训练、冻结，用于离线cache与在线condition编码 |
| Hybrid ViT | 是 | 是 | ImageNet初始化，`1e-5`微调 |
| proprio projection | 是 | 是 | 从零训练 |
| video patch embedding | 是 | 是 | 从零训练 |
| Joint Transformer | 是 | 是 | 从零训练 |
| action head | 是 | 否 | 从零训练 |
| video velocity head | 否 | 是 | 从零训练 |

`0.1`只表示标量loss权重，不表示视频梯度一定恰好占总梯度的10%。token数量、误差尺度和
共享路径都会影响真实梯度比例。

尾部样本使用两套mask：`action_mask [B,16]`只监督真实动作，
`video_future_mask [B,4]`只监督由完整未来RGB组成的latent steps。这样保留最后的lift动作，
同时不把重复padding视频当作真实未来。

## 训练与推理的区别

### 训练

为了避免每个batch重复运行大VAE，先离线处理：

```text
过去9帧 + 未来16帧 = 25帧head RGB
→ frozen Wan2.2 VAE一次编码
→ 7个latent steps
→ 前3步condition_prefix + 后4步future_target
```

现有V1数据为box 300 + bottle 300，共600 episodes、540 train / 60 val。cache保存约
93,771个唯一窗口，float16未压缩容量约11.5GiB；transition/lift oversampling只重复索引，
不复制cache。

### 推理

对应推理仓库分支：
[`fk-issac-logistics/feat/joint-wam-online-rollout-v1`](https://github.com/alex7537/fk-issac-logistics/tree/feat/joint-wam-online-rollout-v1)。

```text
每个Isaac物理step保存head RGB
→ 每个环境维护独立9帧队列
→ Wan VAE在线编码condition_prefix
→ action和future-video latent分别从Gaussian noise开始
→ 5步联合ODE
→ 输出action [B,16,13]和future latent [B,48,4,14,14]
```

当前部署只执行动作，并把生成的future latent暂存在内存`last_video_latent`。尚未调用Wan
decoder，也没有写出预测视频MP4。若要可视化，应把3步condition prefix和4步generated
future拼成7步latent后解码完整序列，再截取未来16帧。

第一次推理没有真实9帧历史，目前使用第一帧左填充到9帧；这是cold-start近似。正式测试
前还必须证明在线9帧编码与训练cache的condition prefix在容差内一致。

## 与基础CFM的核心区别

| 项目 | 基础CFM | Joint WAM |
|---|---|---|
| 视觉时间范围 | 当前双RGB | 当前双RGB + 过去9帧head |
| 生成目标 | 16步action | 16步action + 未来16帧head latent |
| condition tokens | 99 | 246 |
| generated tokens | 16 | 212 |
| Transformer | 4层、384维、4 heads | 同规格，但同时处理video/action |
| loss | action Flow | action Flow + `0.1×` video Flow |
| VAE | 无 | frozen Wan2.2 VAE |
| 部署外部依赖 | 无 | 固定SHA的Wan checkpoint/runtime |
| 推理结果 | action | action + 未解码future latent |

更多token和视频loss不保证动作一定更好。是否有效必须通过相同dataset、seed、checkpoint
选择、Isaac场景、pose、execute horizon和成功判据下的CFM对照实验判断。

## 数据路线

- V1：使用现有V3混合JPEG数据和exact-dedup时间线，已完成训练与schema-v3 bundle导出；
- V2：计划从两个task的原始连续RGB重建，不按joint相等删除视频帧，以保留物体滑动、
  晃动和掉落等“关节不动但视觉变化”的动态。

V2启动前必须先找到与box同协议的bottle原始RGB，避免两个任务使用不对称的视频时间线。

## 代码入口

- `configs/a2d_v3_multitask_joint_wam_scratch_100ep.yaml`：完整训练配置；
- `scripts/precompute_wan_joint_latents.py`：25帧Wan latent cache；
- `flow_matching_test/a2d_dataset.py`：窗口、cache读取与两套mask；
- `flow_matching_test/policies/joint_wam.py`：token化、两套attention、双head和双loss；
- `flow_matching_test/policies/flow_matching.py`：共享Transformer block；
- `flow_matching_test/export_bundle.py`：schema-v3 Joint WAM bundle；
- `tests/test_joint_wam_policy.py`：联合模型、梯度与采样测试；
- `tests/test_joint_wam_latent_cache.py`：cache合同测试。

## 当前验证边界

已完成：正式scratch训练、latent cache、双loss梯度测试、schema-v3 bundle、严格权重图加载、
在线9帧adapter与静态单测。

尚未完成：在线prefix parity、Wan decoder视频导出、Isaac prediction-only、闭环抓取成功率，
以及相同协议下Joint WAM相对基础CFM的收益验证。
