# CFM ViT 0.1× Stage-2 十组串行实验计划

## 目标

针对 1,090-episode CFM ViT 0.1× 模型在真实测试中的动作抖动，区分以下可能原因：

1. 原 5-epoch cosine schedule 已归零，模型仍存在少量欠训练；
2. Stage-2 学习率过高或过低；
3. ViT backbone 更新幅度不合适；
4. AdamW 正则化或动量造成参数更新不稳定；
5. 5-step Euler 推理过粗。

## 共同训练协议

- 初始化：原 5-epoch best checkpoint，仅加载模型权重；
- source checkpoint SHA-256：
  `2e3552469842f56c85804d4c044d6c522120b7b91ae176e14e462afca21e5ea9`；
- optimizer、scheduler 和 global step 全部重置；
- 数据：1,090 episodes，981 train / 109 validation；
- 有效训练样本：162,844；
- batch size：32；
- 每版：3 epochs / 15,267 optimizer steps；
- warmup：500 steps；
- linear warmup + cosine decay；
- seed：42；
- `action_offset_steps=1`，action horizon 16；
- 单 A800 严格串行，任何一版失败即停止队列；
- checkpoint 仍按最低 `val_loss` 选择，同时单独比较
  `val_sample_action_mse` 和 keyframe loss。

## 十组变量

| 版本 | 唯一主要变化 | Head LR | ViT multiplier | Weight decay | Betas | Clip | CFM steps |
|---|---|---:|---:|---:|---|---:|---:|
| V01 | 保守 LR | 1e-5 | 0.1 | 1e-4 | 0.9/0.95 | 1.0 | 10 |
| V02 | 推荐基线 | 2e-5 | 0.1 | 1e-4 | 0.9/0.95 | 1.0 | 10 |
| V03 | 较高 LR | 5e-5 | 0.1 | 1e-4 | 0.9/0.95 | 1.0 | 10 |
| V04 | Stage-2 冻结 ViT | 2e-5 | 0/frozen | 1e-4 | 0.9/0.95 | 1.0 | 10 |
| V05 | 更保守 ViT 更新 | 2e-5 | 0.05 | 1e-4 | 0.9/0.95 | 1.0 | 10 |
| V06 | 更积极 ViT 更新 | 2e-5 | 0.2 | 1e-4 | 0.9/0.95 | 1.0 | 10 |
| V07 | 无 weight decay | 2e-5 | 0.1 | 0 | 0.9/0.95 | 1.0 | 10 |
| V08 | 更强 weight decay | 2e-5 | 0.1 | 1e-3 | 0.9/0.95 | 1.0 | 10 |
| V09 | 更平滑二阶动量 | 2e-5 | 0.1 | 1e-4 | 0.9/0.99 | 1.0 | 10 |
| V10 | 更强梯度裁剪 + 20-step | 2e-5 | 0.1 | 1e-4 | 0.9/0.95 | 0.5 | 20 |

V10 同时改变了训练稳定性和推理积分步数，因此它是组合候选，不用于单变量归因；
若 V10 最优，后续需要用 V02 checkpoint 做 10/20-step 交叉推理确认收益来源。

## 明日验收顺序

1. 先要求队列为 10/10 `COMPLETED`，且不存在 `failure.json`；
2. 每版必须有 3 行 epoch metrics，最终 `global_step=15267`；
3. 每版必须有 `config_resolved.yaml`、`init_events.jsonl`、`best.ckpt`、
   `latest.ckpt` 和 `summary.json`；
4. 检查 source checkpoint SHA 与 model-only reset 标记；
5. 离线初排以 `val_sample_action_mse` 为主、`val_loss` 为次，keyframe loss 为保护指标；
6. 对排名靠前的版本使用相同初始状态、相同 seed 和相同场景进行 rollout；
7. 真实选择不能只看离线 loss，至少同时记录成功率、chunk 内动作一阶差分、二阶差分和
   chunk 边界跳变量；
8. V02 必须分别以 5/10/20 inference steps 交叉测试，用于隔离训练收益与积分收益。

## 自动产物

训练队列：

`/share_data/zhangyurui/flow-matching-test-a2d-v2/runs/cfm_stage2_10way_20260722_194500/`

完成后自动生成：

- `ACCEPTANCE_REPORT.md` 与 `acceptance_report.json`；
- 全部十版的 `test_bundles/`；
- 全部十版的固定输入参考推理 `reference_inference/`；
- `TEST_ARTIFACTS.json`；
- `FINALIZATION_COMPLETE`。
