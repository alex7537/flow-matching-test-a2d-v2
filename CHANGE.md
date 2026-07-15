# Change Log

本文件按时间倒序记录项目的重要更新；后续每次完成代码、数据、训练或部署交付后，在顶部追加一条，并记录对应 Git commit 与验收结果。

## 2026-07-15｜66 条目标训练与 4090 交付封卷

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
