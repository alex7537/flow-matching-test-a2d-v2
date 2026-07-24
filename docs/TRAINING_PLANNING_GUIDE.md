# 训练预算与学习率规划指南

这份文档用于回答一个核心问题：当数据量、batch size 或模型可训练参数发生变化时，应该如何重新设置 epochs、steps、warmup 和学习率，而不是机械沿用旧配置。

训练技巧、消融顺序和 rollout 决策规则见
[TRAINING_TRICKS_GUIDE.md](TRAINING_TRICKS_GUIDE.md)。

## 1. 先统一计量单位

训练预算优先使用 optimizer steps 和样本曝光量衡量，epoch 只表示“完整遍历一次当前训练集”。数据集变大以后，同样一个 epoch 会包含更多 steps，因此不同数据集之间不能直接比较 epoch 数。

定义：

- `N`：经过 transition/keyframe oversampling 后的有效训练样本数。
- `B`：batch size。
- `S`：每个 epoch 的 optimizer steps。
- `E`：epoch 数。
- `T`：总 optimizer steps。
- `r`：warmup 占总 steps 的比例。

当前训练器不丢弃最后一个不足 batch 的训练 batch，因此：

```text
S = ceil(N / B)
T = S × E
warmup_steps = round(T × r)
样本曝光量 = N × E
```

如果其他训练器设置了 `drop_last=true`，则 `S=floor(N/B)`。

## 2. 新旧数据如何换算

如果目标是保持和旧实验相近的 optimizer-step 预算：

```text
旧总 steps = 旧 steps/epoch × 旧 epochs
新等价 epochs = 旧总 steps / 新 steps/epoch
```

以旧 66 episodes、30 epochs 和当前 1,090 episodes 为例，在单条 episode 长度与采样规则近似时，新数据约是旧数据的 16.5 倍：

```text
30 / 16.5 ≈ 1.8 epochs
```

这说明“新数据 30 epochs”不是旧实验的等价迁移，而是大约 16.5 倍的额外计算量。

## 3. 当前 1,090-episode 数据计算实例

实际训练日志给出：

```text
基础完整窗口：146,324
transition 窗口：16,520
oversample factor：2
有效训练样本 N：162,844
batch size B：32
```

因此：

```text
steps/epoch = ceil(162844 / 32) = 5,089
5 epochs 总 steps = 5,089 × 5 = 25,445
5% warmup = round(25,445 × 0.05) = 1,272
样本曝光量 = 162,844 × 5 = 814,220
```

可用设备级 skill 中的确定性脚本复算：

```bash
python ~/.codex/skills/plan-training-run/scripts/plan_training.py \
  --samples 162844 \
  --batch-size 32 \
  --epochs 5 \
  --warmup-ratio 0.05
```

## 4. Warmup 与 cosine decay

本项目在未显式配置 `warmup_steps` 时使用总 steps 的 5% 做线性 warmup，然后在剩余 steps 上执行 cosine decay。

因此，直接把一个 30-epoch run 在第 5 个 epoch 手工停止，不等价于从一开始配置 5 epochs：

- 30-epoch 计划在 epoch 5 时仍处于长 cosine 曲线前段，学习率接近峰值。
- 5-epoch 计划会在 step 25,445 附近衰减到接近 0。

如果实验问题是“完整的 5-epoch schedule 效果如何”，应从头使用 5 epochs。人工截断只适合快速获取早期 checkpoint。

## 5. Frozen 与 0.1× backbone

参数组学习率为：

```text
head_lr = base_lr
backbone_lr = base_lr × backbone_lr_multiplier
```

- Frozen：`freeze_encoder_backbone=true`，`backbone_lr_multiplier=0.0`。
- 0.1×：`freeze_encoder_backbone=false`，`backbone_lr_multiplier=0.1`。

0.1× 表示 backbone 学习率是 head 的十分之一，不表示只训练 10% 的 backbone。

做公平对比时，两组必须共享：

- dataset/split/stats；
- seed、batch size、总 steps、warmup 和 cosine 终点；
- augmentation、CFM policy、action horizon 与 offset；
- checkpoint 和评估协议。

## 6. Batch size 如何调整

增大 batch size 通常能提高 GPU 吞吐，但会同时减少每个 epoch 的 optimizer steps，并改变梯度噪声。它不是纯粹的性能参数。

调整 batch size 后必须重新计算：

- steps/epoch；
- total steps；
- warmup steps；
- 样本曝光量；
- 是否需要调整 base LR。

线性放大学习率可以作为待验证假设，但不是对所有模型都成立的规则。做 frozen/0.1× 消融时，优先保持 batch size 不变。

## 7. 何时停止或续训

至少同时检查：

- `val_loss`；
- `val_continuous_loss` 与 `val_keyframe_loss`；
- `val_sample_action_mse`；
- train/val gap；
- rollout 或任务成功率；
- 当前学习率在 schedule 中的位置。

可作为起点的经验规则：达到计划的最小预算后，如果验证指标连续两次改善不足 1–2%，且 rollout 不再改善，可以停止。该阈值是决策辅助，不是数学定律。

如果 cosine 已经衰减到 0，而指标仍在明显改善，不应直接给原 run 增加 epochs；应定义带新 LR schedule 的第二阶段，并明确记录它是 continuation。

## 8. 已完成的五轮对比基线

Frozen 与 0.1× 已于 2026-07-21 各自从头完成：

```text
epochs：5
steps/epoch：5,089
total steps：25,445
warmup steps：约 1,272
batch size：32
base LR：1e-4
schedule：linear warmup + cosine decay
```

每个 epoch 保存 `latest.ckpt`，按最低 `val_loss` 保存 `best.ckpt`。两组最低
`val_loss` 与最低 `val_sample_action_mse` 均出现在最终 epoch 4。最终结果：

| 指标 | Frozen | 0.1× ViT | 0.1× 相对改善 |
|---|---:|---:|---:|
| `val_loss` | 0.024397 | **0.022643** | **7.19%** |
| `val_continuous_loss` | 0.019114 | **0.017756** | **7.10%** |
| `val_keyframe_loss` | 0.032670 | **0.030506** | **6.63%** |
| `val_sample_action_mse` | 0.006680 | **0.005561** | **16.75%** |

当前 offline baseline 为 0.1× ViT，但最终选择等待固定协议 rollout。两组 cosine
均已衰减到 0，不直接增加 epochs。若 rollout 证明仍需训练，只为胜出候选建立新的
stage 2 LR schedule。完整证据与 checkpoint SHA 见
[实验报告](../reports/cfm_a2d_450gb_vit_freeze_ablation_20260721.md)。

## 9. 训练运行的最低监控要求

- batch 级 heartbeat 或定期 step 指标；
- epoch 级 train/val 分项 loss；
- 原子写入 latest/best checkpoint；
- 固定 dataset/split/stats 和 resolved config provenance；
- 卡死与未捕获异常告警；
- 失败时返回非零退出码，防止队列继续启动下一实验。
