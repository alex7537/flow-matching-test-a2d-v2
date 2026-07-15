# CFM / RS-IMLE / Diffusion 公平对比协议

## 目的与启动条件

本协议用于比较 `flow_matching`、`imle`、`diffusion` 三种 policy；6 条数据只做实现正确性与过拟合探针，正式结论必须等待约 640 条数据完成版本化入库、固定 split，并在 rollout Level 3 与 hand `6→11` 映射验收通过后产生。

## 冻结项

三个正式 run 必须使用相同的：

- dataset version、dataset manifest SHA-256、stats digest 与 episode 级 split manifest；
- 双相机输入 `rgb_head + rgb_right_hand`、proprio/action 语义、history 和 action horizon；
- timm backbone、Transformer 宽度/层数/头数、数据增强、optimizer、LR schedule、batch size、总 optimizer step 预算与 seed；
- checkpoint、bundle、182 网格 rollout 场景、执行 horizon、随机种子和成功判据。

算法固有参数允许不同，但必须进入 config/provenance：CFM 的 Euler steps、RS-IMLE 的 `n_samples_per_condition/epsilon`、Diffusion 的 noise schedule 与 DDIM steps。比较按相同 optimizer steps 而不是 epoch 或 wall-clock 时间终止。

“同网络主体”限定为相同 Transformer 骨架和 observation 管线；CFM/Diffusion 的时间条件与 RS-IMLE 的 latent 注入属于方法本征差异，不视为额外混淆变量。各方法必须使用其标准推理形态：CFM 用 5-step Euler（既有 5-vs-16 复读实验未见劣化）、Diffusion 用 DDIM、RS-IMLE 在重规划时用 32 候选的双向连续性选择。考虑 setpoint 语义，RS-IMLE 衔接距离默认 `arm_weight=0、hand_weight=1`，权重与候选数必须进入 config；该机制只在 `execute_horizon < chunk_size` 时生效，并在 Level 4 的 horizon=8/4 对照中单独评估。

## 评价与记录

1. 主指标：相同 182 网格上的 rollout success rate，并报告 approach/close/lift 分阶段失败。
2. 次指标：`val_sample_action_mse` 量级；多模态任务下不作为唯一优劣判据。
3. 实用指标：相同 A800/4090 环境、batch=1、预热后统计端到端 policy 推理延迟的 median/P95。
4. 禁止跨 policy 比较 flow loss、RS-IMLE min-distance loss 与 diffusion epsilon loss；三者目标空间不同。
5. W&B 正式 run 统一挂团队 entity，以 `policy_type` 分组；config 必须记录 git SHA、dataset/split/stats digest、训练 step 预算与 policy 专属参数。

正式结论至少同时给出 rollout 成功率、sample MSE、推理延迟和资源成本；任一 run 的数据、网络主体或 step 预算不一致时标记为不可比，不进入主表。
