# CFM ViT 冻结 / 微调消融报告

- 日期：2026-07-20
- 数据：A2D 双 RGB，66 episodes（固定 60 train / 6 val）
- 策略：Conditional Flow Matching
- 控制变量：seed 42、batch 32、30 epochs、7380 steps、相同数据增强/调度器/动作空间
- 代码：f5deeef59011b7a28ce38392ed6a8b31b2464150
- W&B group：cfm-vit-freeze-ablation-20260720

## 实验定义

| 组别 | ViT 参数 | Backbone LR | 其余网络 LR |
|---|---:|---:|---:|
| Frozen | requires_grad=False，不进入 AdamW | 0 | 1e-4 |
| Fine-tune 0.1× | 参与反向传播与 AdamW | 1e-5 | 1e-4 |

视觉 adapter、proprio 路径和动作模型在两组中均参与训练。三个 encoder 指标只用于监控，不加入 CFM loss。

## 核心结果

| 指标（越低越好） | Frozen ViT | Fine-tune 0.1× | 结论 |
|---|---:|---:|---|
| Best val flow loss | **0.041467** @ epoch 22 | 0.061160 @ epoch 18 | Frozen 低 **32.20%** |
| Final val flow loss | **0.046043** | 0.062731 | Frozen 低 **26.60%** |
| Last-5 val loss 均值 | **0.049277** | 0.065624 | Frozen 低 **24.91%** |
| Best-epoch val continuous loss | **0.030125** | 0.046865 | Frozen 低 **35.72%** |
| Best-epoch val keyframe loss | **0.049980** | 0.071463 | Frozen 低 **30.06%** |
| Best val sample action MSE | 0.012682 @ epoch 25 | **0.012586** @ epoch 19 | Fine-tune 仅低 **0.76%** |
| Final train flow loss | 0.030339 | **0.023905** | Fine-tune 低 **21.21%** |
| Final train→val gap | **0.015704** | 0.038827 | Fine-tune 泛化间隙更大 |

## Encoder 监控

| 指标 | Frozen ViT | Fine-tune 0.1× |
|---|---:|---:|
| Backbone epoch 最大更新比例 | **0** | 0.001726 |
| Backbone epoch 平均梯度范数 | **0** | 0.311034 |
| Feature std：首 epoch → 末 epoch | 1.7565 → 1.7660（+0.54%） | 1.4776 → 1.1284（−23.63%） |

冻结组 backbone 梯度和参数更新严格为 0，说明冻结实现有效。微调组确实在学习，但 validation flow loss 更差、train→val gap 更大；feature std 明显下降表示表征分布发生较大变化，单凭该指标不能断言坍缩。

## 判断

**当前 66-episode 数据规模下，冻结预训练 ViT 是更优主基线。**

- 微调 ViT 能把训练 loss 压得更低，但没有转化为更好的 validation flow loss，表现出更明显的过拟合。
- Fine-tune 在最佳 sample action MSE 上仅领先 0.76%，不足以抵消 flow loss、连续段和关键帧 validation 指标的全面劣势。
- Frozen 耗时 778.1 秒；Fine-tune 耗时 1146.7 秒，后者多约 47.36%。

建议正式仿真验证优先使用 Frozen 的 best checkpoint；Fine-tune checkpoint 保留作对照。下一轮若继续探索视觉适配，优先尝试只解冻最后若干 block 或更小 backbone LR。

## W&B 与产物

- Frozen：[W&B run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/essqf82i)
  - best checkpoint SHA256：d09e73ef14e94b600b3537d0844b9a385ef14c0bf8111b2369e5cf39965ba2b4
- Fine-tune：[W&B run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/y57y2zjq)
  - best checkpoint SHA256：c95af39c14190ac91b2167048143c793cf1ed42f310743e0561c45be6181729e

验收：两组本地 metrics.jsonl 均为 30 epochs；W&B 服务端 history 均为 30 epochs；最终 step 均为 7380。

> Temporal-contract notice (2026-07-20): both checkpoints in this report were trained with the legacy offset=0 window, where chunk[0] reconstructs the current executed qpos. They remain valid evidence for the freeze ablation but are not next-action checkpoints. Use the new offset=1 config and retrain before deploying a server whose chunk[0] must mean the next frame.
