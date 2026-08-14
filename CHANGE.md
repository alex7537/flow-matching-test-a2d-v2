# Change Log

本文件按时间倒序记录项目的重要更新；后续每次完成代码、数据、训练或部署交付后，在顶部追加一条，并记录对应 Git commit 与验收结果。

## 2026-07-22｜1,090-episode ViT 0.1× Level 0 测试包落地

- 从 A800 的 best epoch 4 / step 25,445 导出 bundle v2，固定 `offset=1`、双 RGB
  `640×480`、16×13 action chunk、execute horizon 16 与 CFM 5-step 推理合同。
- 使用固定 val[0] frame 0、seed `20260721` 重新加载 bundle，16×13 输出最大绝对误差
  为 `0.0`，有限值与动作范围哨兵通过；目录内 `sha256sum -c SHA256SUMS` 全绿。
- 本机交付目录为 `~/rollout_handoff/level0_prep_1090ep_cfm_vit_finetune_01x_step25445/`；
  单文件 tar SHA-256 为 `f0eeb5d75bf92d93dc54b46d4c40c5b00ec474d04b49bb51646b0bc510c4cd4c`。
- 因训练工作树非 clean，测试包额外携带 `training_source.patch`、untracked source、
  resolved config、metrics、summary 和 A800 requirements lock，禁止只按 Git HEAD 复现。

## 2026-07-21｜A2D 450GB 五轮 ViT 冻结 / 0.1× 消融完成

- 基于 1,090 episodes、162,844 个 oversampling 后有效训练窗口，完成 Frozen 与
  Fine-tune 0.1× 的同预算实验：各 5 epochs / 25,445 steps / warmup 1,272，
  `action_offset_steps=1`，两组均无 failure。
- 0.1× 最终 `val_loss=0.022643`、`val_keyframe_loss=0.030506`、
  `val_sample_action_mse=0.005561`，相对 Frozen 分别改善 7.19%、6.63%、16.75%；
  当前将 0.1× 记为 offline baseline，最终选择等待固定协议 rollout。
- 两组 cosine LR 已衰减到 0，不原样续训。若 rollout 证明仍需增加预算，只为胜出
  候选建立独立 stage 2 schedule；Frozen 保留为 rollout 对照。
- 新增 `reports/cfm_a2d_450gb_vit_freeze_ablation_20260721.md`，记录完整曲线判断、
  W&B run、checkpoint/config SHA 与 dirty-worktree 可复现性风险。

## 2026-07-20｜Frozen ViT 下一帧动作模型与推理包

- 完成 Frozen ViT 与 ViT Fine-tune 0.1× 的受控消融；Frozen 的 best val flow loss 为 `0.041467`，Fine-tune 为 `0.061160`。微调虽然获得更低 train loss，但 validation gap 更大，因此当前 66-episode 数据规模选择冻结预训练 ViT。
- 明确当前实现中的“视觉 adapter”边界：仓库提供 token 维度对齐 adapter，但 `vit_small_r26_s32_224` 的输出维度正好等于 `d_model=384`，本配置下 adapter 为 `Identity`；ViT 冻结后，由 condition 位置编码、proprio projection 与 Flow Matching Transformer 学习任务适配。
- `a78fa8f` 将动作窗口改为下一帧契约：`obs[t] → action[t+1:t+17]`。完整窗口过滤、分段标签、checkpoint/resume provenance 和 bundle 配置均同步携带 `action_offset_steps=1`；旧 offset=0 checkpoint 仅保留作消融证据。
- A800 完成 `cfm_66ep_vit_frozen_next_action_seed42`：30 epochs / 7320 steps，best `val_loss=0.051097`（epoch 21），best `val_sample_action_mse=0.013971`（epoch 27）；W&B run 为 `arzpl1mx`。
- 导出并校验 `cfm_frozen_next_action_offset1_seed42_best.tgz`：归档 SHA-256 为 `6225396967997d96ff4911e186612a7535091f9395d4f424f38481c56afd2175`，manifest 锚定 `a78fa8f89e504e18100c424bdf1216540be8dec7`。
- 使用真实成功轨迹 `episode_000002_success.hdf5` frame 0 完成本地 CPU 推理，输出为 `(16,13)` float32，全部有限且动作范围哨兵通过；下一阶段是接入仿真推理 server 与 `set angle` 执行闭环。

## 2026-07-16｜三 policy 部署产物登记

- `9f8e60a` 修复 Diffusion 的 DDIM 末步数值放大：将预测的 clean action 裁剪到训练归一化契约 `[-1,1]`，使原本无效的 DP val sample MSE（约 `5820.76`）恢复为 `0.01376`；CFM/RS-IMLE 经出口对称性检查仅有轻微学习型越界，保持不裁剪。
- 已导出并逐项 SHA-256 验证 RS-IMLE best（epoch 24 / step 6150）与 Diffusion best（epoch 29 / step 7380）的 Level 0 自包含包；二者均使用相同 val[0] frame 0、seed `20260715` 的参考输入/输出，bundle manifest 锚定 `9f8e60a`。
- 三 policy 的 loss/sample MSE 只作为各自数值健康证据，禁止横向排名；最终优劣统一由 ws-05 上的同协议 rollout 成功率裁决。

## 2026-07-16｜A800 凭据边界修正

- 修正旧诊断：`activate_a800.sh` 会将 `HOME` 指向共享 CFS 的 `$WORK/home`，凭据读取路径变化而非“容器重启清空 `/root/.netrc`”更能解释此前登录状态丢失；现已审计该目录，无 `.netrc`、history、`.ssh` 或 credential 残留，并将目录权限由 `755` 收紧为 `700`。
- `activate_a800.sh` 改为仅从容器本地 `/root/.secrets/wandb_api_key` 可选加载 W&B key，保留 TI-ONE Secret 作为优先方案；密钥不得写入 Git、shell history 或任何 `/share_data` 路径。

## 2026-07-15｜66 条目标训练与 4090 交付封卷

- 新建 `artifacts_index.md`，登记 66 条 best bundle、Level 0 自包含交付包、val 可视化与 W&B run 的位置、SHA-256 和代码血统；`.gitignore` 归拢产物目录并补充 `*.tgz`，README 固化“产物本体不入库、身份信息进索引”的原则。
- `README.md` 固化 GitHub Flow 协作约定：`main` 为唯一长期且可部署的事实源，所有改动使用 `<type>/<description>` 短命分支 + PR + Squash and merge，合并后删除分支，禁止直接 push `main`；`.gitignore` 同时隔离 `模型存档/` 与 `结果-图片/`。
- GitHub 邀请与飞书文件两条通道均已完成对 Richard 的物理交接，本方动作清零，项目转入等待 ws-05 `LEVEL0_REPORT` 至最终 6×5 rollout 矩阵的报告回流阶段。
- `目标架构.md` 新增跨机交付原则：禁止正式交付裸 ckpt，必须使用带 bundle、参考输入/输出、环境 lock 与 SHA 清单的自包含目录，逐跳校验并通过 Level 0 同 seed 对账后才可 rollout。
- Level 0 自包含交付目录已从 A800 完整同步至 psibot `~/rollout_handoff/level0_prep_66ep_step4674/`，TGZ、NPZ、JSON、lock 四项 `sha256sum -c SHA256SUMS` 全绿；runbook 已收口为“单目录、单次传输、单条校验”。
- 新增 `train+deploy/handoff_runbook.md`，作为 ws-05 交接入口，串联交付物、环境、Level 0→3 执行顺序、红线与汇报格式。
- `b843b0f`：A800 Level 0 参考包完成，包含 66 条 best bundle、固定输入 NPZ、16×13 同 seed 输出 JSON、A800 lock 与四项 `SHA256SUMS`；4090 agent 方案交付标记完成。
- `e5ce3ca`：新增参考推理生成器与 `ws05_rollout_task_v2.md`，冻结 Level 0→3、USD mimic、robot-local 坐标及 6 条 val × 5 次 rollout 验收协议。
- `333f3b5`：完成 6 条 val 的 pregrasp/grasp 关键点可视化；grasp 手部平均 MSE 为 `0.00141 rad²`，3 条 arm-pregrasp 弱位姿留给 rollout 裁决。
- `6e5a561`：约 20GB/66 条完整 lift 数据完成 A800 CFM 训练；30 epochs / 7380 steps，best 为 epoch 18 / step 4674，`val_loss=0.06124`、`val_sample_action_mse=0.01386`。
- best bundle TGZ SHA-256：`ebe1d1eb52de01e25d8c581c357c5bd71c5f5ccb30717c7a065812579335ecd8`；训练与执行方案均已收口，ws-05 只需按 v2 任务书执行。

## 2026-07-15｜数据契约与策略对比

- `314e364`：新 A2D episode 的 `/meta` group 改为可选读取，同时保留完整性审计链路。
- `bf55a18`：RS-IMLE 增加逐通道加权的双向 rollout 候选选择。
- `c02763d`：冻结 CFM、RS-IMLE、Diffusion 三策略公平对比协议；训练 loss 禁止跨策略直接比较。

## 2026-07-14｜策略解耦与 W&B

- `c704c04`、`f35c55a`、`bc6b0a5`：完成统一 policy 接口、RS-IMLE/Diffusion 基线、policy 专属过拟合探针及 bundle/rollout 兼容。
- `c35d6ee`、`cfd4749`、`87d06da`：完成 W&B offline/online 分级、同步 IPC 修复、服务端 history 行数验收与环境版本钉死。
- `172bc8f`：固化 v0.2 训练与部署架构快照。

## 2026-07-13｜A800 全链路基线

- `1a22c53`：A2D 数据入库、CFM 训练、rollout harness 与 bundle v2 首次入库。
- `0aab671`、`482adb7`、`c03a1de`：完成固定 split/stats provenance、A800 preflight、checkpoint resume 与 CUDA RNG state 恢复。
- 六条历史基线完成 20 epochs / 460 steps，形成首个带完整 provenance 的可部署 bundle。
