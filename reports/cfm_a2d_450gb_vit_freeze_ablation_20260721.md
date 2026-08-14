# A2D 450GB CFM ViT 冻结 / 0.1× 微调消融报告

- 日期：2026-07-21
- 数据：`a2d_450gb_rgb_v1`，1,090 episodes（981 train / 109 validation）
- 策略：Conditional Flow Matching
- 动作契约：`obs[t] -> action[t+1:t+17]`，`action_offset_steps=1`
- 采样：146,324 基础训练窗口，factor 2 后 162,844 有效样本
- 控制变量：seed 42、batch 32、5 epochs、25,445 steps、warmup 1,272
- 调度：linear warmup + cosine decay，最终 LR 为 0
- 代码 HEAD：`fd2b7512affaff00ef03165953978c8e85ab20b2`（训练时工作树包含未提交修改，见“可复现性说明”）

## 实验定义

| 组别 | ViT 参数 | Backbone LR | Head LR | 训练时长 |
|---|---:|---:|---:|---:|
| Frozen | `requires_grad=false` | 0 | `1e-4` | 3,411.463 s |
| Fine-tune 0.1× | 参与反向传播 | `1e-5` | `1e-4` | 4,098.970 s |

除 ViT 是否训练及其参数组 LR 外，两组共享 dataset/split/stats、网络、增强、seed、
batch、总 steps、warmup、cosine 终点、validation 和 checkpoint 协议。

## 最终结果

两组均正常完成 5 epochs / 25,445 steps，无 `failure.json`；两组的最低
`val_loss` 和最低 `val_sample_action_mse` 均出现在最终 epoch 4。

| 指标（越低越好） | Frozen | Fine-tune 0.1× | 0.1× 相对改善 |
|---|---:|---:|---:|
| Final / best val loss | 0.024397 | **0.022643** | **7.19%** |
| Final val continuous loss | 0.019114 | **0.017756** | **7.10%** |
| Final val keyframe loss | 0.032670 | **0.030506** | **6.63%** |
| Final / best val sample action MSE | 0.006680 | **0.005561** | **16.75%** |
| Final train loss | 0.024300 | **0.020849** | **14.20%** |

## 收敛判断

最后一轮相对 epoch 3 的变化：

| 指标 | Frozen | Fine-tune 0.1× |
|---|---:|---:|
| val loss | 改善 5.10% | 改善 4.89% |
| val continuous loss | 改善 7.86% | 改善 8.23% |
| val keyframe loss | 改善 0.78% | **变差 0.43%** |
| val sample action MSE | 改善 2.12% | 改善 2.02% |

aggregate/continuous 指标仍下降，但 sample MSE 已接近平台，0.1× keyframe 指标在
最后一轮轻微反弹。0.1× 的 final train-to-validation gap 为 0.001794（相对 train
约 8.61%）；validation 仍下降，因此目前不能据此断言明显过拟合。

## 判断与下一步

**当前 offline baseline 选择 Fine-tune 0.1×，但最终模型选择必须等待同协议 rollout。**

- Frozen 在全部最终 offline 指标上落后，不继续训练，仅保留作 rollout 对照。
- 不直接续训 0.1×：当前 cosine 已衰减到 0，原 schedule 已完成。
- 先对两组 best checkpoint 使用同一固定初始条件各做至少 20 次 rollout。
- 若成功率差距小于约 10 个百分点，扩大到 50 次再判断。
- 如果 0.1× rollout 方向正确但绝对成功率不足，建立独立 stage 2：从 best 模型
  权重开始，建议先试 2 epochs / 10,178 steps、约 509 warmup steps、head peak LR
  `1e-5` 至 `2e-5`、backbone 0.1×，使用新的完整 schedule。
- stage 2 不得直接恢复已经归零的 scheduler；训练前需明确 model-only initialization
  与 optimizer/scheduler 的新状态。

## W&B 与产物

- Frozen：[W&B run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/monbffdt)
  - run：`cfm_1090ep_vit_frozen_5ep_seed42_20260721_142310`
  - best checkpoint SHA-256：`783d7346331d70cf6d35651a150ed96997cf28e2180f62daed4e096e11b58d3a`
  - resolved config SHA-256：`30c4f992244a495cac79bf3d5aa51a7dba3716c36180a7aea36cc582a37a46fd`
- Fine-tune 0.1×：[W&B run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/fmo6wobe)
  - run：`cfm_1090ep_vit_finetune_01x_5ep_seed42_20260721_142310`
  - best checkpoint SHA-256：`2e3552469842f56c85804d4c044d6c522120b7b91ae176e14e462afca21e5ea9`
  - resolved config SHA-256：`2f948287d7dbae6384186d26f4fa6afcca1820c78b38f8cfe1e4c57fdf50db3c`

远端产物根目录：

```text
/share_data/zhangyurui/flow-matching-test-a2d-v2/runs/
```

## 可复现性说明

训练时远端工作树包含已同步但未提交的 dataset LRU、watchdog、W&B 降级、配置与测试
修改。因此记录的 Git HEAD 不能单独恢复训练源码。归档或提交前必须同时保存 `git diff`、
untracked 文件、resolved config 和上述 SHA；后续正式实验应使用 clean commit。
