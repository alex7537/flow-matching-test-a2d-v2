# A2D Joint Latent WAM

本分支在个人仓库中实现紧凑型视频—动作联合Flow Matching：模型根据当前双相机
RGB、当前proprio和head相机过去9帧，同时生成未来视频latent与16步关节动作。

它不加载旧CFM checkpoint，也不依赖公司`psi-policy`代码。Wan2.2只作为公开、
冻结的视频VAE；Joint Transformer、video head和action head在本仓库中从零训练。

## 路线概览

| 路线 | 数据 | 目的 | 状态 |
|---|---|---|---|
| CFM baseline | 当前双RGB + proprio → action | 已有动作策略对照 | 已有训练结果 |
| Joint WAM V1 | 现有V3混合数据 | 先验证视频历史和双Flow loss是否有效 | 已完成训练、bundle导出与静态推理接入 |
| Joint WAM V2 | 两个task的原始连续RGB重新处理 | 恢复严格时间间隔，训练更可靠的视频动力学 | 计划中 |

V1和V2使用相同模型；区别主要在数据时间线和图像处理质量。只有V1产生正向证据后，
才投入成本重建V2。

## V1：现有V3混合数据

当前数据来自原始约500GB HDF5的离线处理结果，不是ViT特征：

```text
原始RGB/depth HDF5
→ 保留head与right-hand RGB
→ resize到224×224，JPEG quality 92
→ exact joint dedup keep-last
→ V3 action：arm actual 7D + hand commanded target 6D
→ box 300 + bottle 300
→ 600 episodes / 99,171帧 / 约2.1GB
```

固定拆分为540 train、60 val。训练窗口使用：

| 张量 | 时间与形状 | 角色 |
|---|---|---|
| 当前head RGB | `[B,1,3,224,224]` | ViT视觉条件 |
| 当前right-hand RGB | `[B,1,3,224,224]` | ViT视觉条件 |
| 当前qpos | `[B,13]` | proprio条件 |
| head RGB history | `t-8:t`，9帧 | 视频历史条件 |
| action GT | `t+1:t+16`，`[B,16,13]` | action目标 |
| head RGB future | `t+1:t+16`，16帧 | future-video目标 |

### Wan latent cache

公开Wan2.2 VAE将每个9+16窗口编码为：

```text
condition_prefix [B,48,3,14,14]
future_target    [B,48,4,14,14]
```

VAE仅在预计算阶段运行。93,771个唯一窗口以float16保存，未压缩容量约11.5GiB；
训练中的transition/lift oversampling只重复索引，不复制cache。

Cache变大主要因为相邻25帧窗口高度重叠，而不是产生了新的独立数据。manifest绑定
dataset SHA、episode content hash、Wan VAE SHA、预处理和时序合同。

## 模型架构

```text
当前双RGB ── ViT ───────────────────────────────┐
当前qpos ── Linear + LayerNorm ─────────────────┤
过去9帧head ── frozen Wan VAE ── prefix tokens ┤
                                                 ├─ condition
噪声future-video latent ── 196 video tokens ───┐│
                                               ├┴─ Shared Transformer ×4
噪声action [16,13] ────── 16 action tokens ────┘
                                                    ├─ Video velocity head
                                                    └─ Action velocity head
```

Future-video token与action token进入同一组self-attention，因此两种预测可以互相读取；
当前RGB、proprio和过去视频prefix通过cross-attention提供条件。

本模型是`d_model=384`的紧凑Joint WAM，不是Wan TI2V-5B DiT。Wan只提供公开VAE
latent空间。

## 双Flow Matching

共享随机时间`τ`，视频和动作使用独立高斯噪声：

```text
z_τ = (1-τ) ε_video  + τ z_future
a_τ = (1-τ) ε_action + τ a_future

target_video_velocity  = z_future - ε_video
target_action_velocity = a_future - ε_action
```

两个loss分别计算：

```text
L_total = 1.0 × L_action_flow + 0.1 × L_video_flow
```

`0.1`是V1起点，不代表视频梯度恰好占10%。正式判断需要同时看两个loss、共享
Transformer梯度、video controllability和闭环抓取成功率。

## 参数更新

| 模块 | 初始化 | 学习率/梯度 |
|---|---|---|
| Wan2.2 VAE | 公开预训练 | 冻结；只生成cache |
| RGB ViT | ImageNet预训练 | `1e-5`，即主学习率0.1× |
| Joint Transformer | 随机 | `1e-4`；接收两个loss |
| Video/Action heads | 随机 | `1e-4` |

Scratch定义为`init_from=null`、`resume_from=null`，即不继承任何A2D策略权重；不是
随机初始化Wan VAE或ViT。

## 尾部与mask

- `action_mask [16]`只监督真实未来动作；
- `video_future_mask [4]`只监督由完整4帧组成的future latent step；
- 无效视频token不会作为self-attention key；
- 最后的lift动作仍能参与训练，padding视频不会成为GT。

## V2：原始连续RGB重处理

V2的目标不是简单使用更大文件，而是恢复更可信的视觉时间线：

```text
两个task的原始RGB episode
→ 保留原始frame index、timestamp与固定FPS
→ 不按joint相等删除视频帧
→ head/right-hand采用相同resize、压缩和相机协议
→ 对齐V3 hand commanded action
→ 重新建立600-episode 1:1数据与train/val split
→ 重新生成Wan latent cache
```

重新处理的主要收益是保留“joint不动但物体仍滑动、晃动或掉落”的视觉变化。原始
500GB不能直接随机读取训练：旧HDF5 chunk布局有明显读放大，仍需顺序转换为随机读取
友好的中间数据。

V2当前阻塞项：本地500GB主要对应box来源；必须先定位同协议的bottle原始RGB，避免
box使用连续原始视频、bottle继续使用旧JPEG/dedup时间线的不对称训练。

## 执行计划

V1使用一次性状态机，在当前混合CFM完整结束后执行：

```text
验证前序summary且无failure
→ 确认旧trainer退出、GPU空闲
→ 安装固定commit的公开Wan runtime与独立venv
→ 生成latent cache
→ 1-step smoke
→ 100-step smoke
→ 1-epoch smoke
→ 全部通过后启动100-epoch scratch训练
```

正式预算：`N_eff=115,265`、batch 32、3,603 steps/epoch、100 epochs、
360,300 total steps、18,015 warmup steps、cosine decay。任一门禁失败即停止，
不自动重试。

## 当前证据与边界

- 真实600条数据索引、Wan latent cache、正式 scratch 训练和 schema-v3 bundle 导出已经完成；
- action/video loss均能更新共享Transformer，跨模态梯度测试通过；
- 导出模型与部署图已做严格 state-dict key/shape 核对；
- `fk-issac-logistics/feat/joint-wam-online-rollout-v1` 已实现9帧在线历史与 Wan VAE 条件编码；
- 仍需在目标GPU完成在线 prefix 与训练 cache 的数值一致性检查，以及 Isaac prediction-only/闭环 smoke；
- video loss下降不等于动作更好，最终仍由同协议rollout成功率裁决。

## 代码入口

- `flow_matching_test/policies/joint_wam.py`：联合Transformer、双loss、联合ODE；
- `flow_matching_test/a2d_dataset.py`：V3窗口、cache V2与尾部mask；
- `scripts/precompute_wan_joint_latents.py`：公开Wan VAE cache；
- `configs/a2d_v3_multitask_joint_wam_scratch_100ep.yaml`：正式训练配置；
- `scripts/queue_joint_wam_after_cfm.py`：前序→cache→smoke→正式训练状态机；
- `tests/test_joint_wam_policy.py`、`tests/test_joint_wam_latent_cache.py`：核心验证。

本分支的个人GitHub同步与任何公司remote操作是两套权限。公司remote写入必须经过
`corporate-git-push-gate`展示精确目标后，由用户另行确认。
