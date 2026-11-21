# Training Budget and Learning-Rate Planning

**English** | [简体中文](TRAINING_PLANNING_GUIDE.zh-CN.md)

Use this guide when dataset size, sampling, batch size, or trainable backbone settings change. The numerical five-epoch experiment below is a **historical baseline**, not the current V3 training budget.

## Count steps before epochs

Let N be the effective training sample count after oversampling, B the batch size, E the epochs, and r the warmup fraction. With no dropped final batch:

```text
steps_per_epoch = ceil(N / B)
total_steps = steps_per_epoch * E
warmup_steps = round(total_steps * r)
sample_presentations = N * E
```

If another trainer uses `drop_last=true`, use floor rather than ceil. These formulas describe the documented single-device, one-update-per-batch setup.

To preserve an older step budget:

```text
new_equivalent_epochs = old_total_steps / new_steps_per_epoch
```

If episode lengths and sampling are similar, moving from 66 to 1,090 episodes increases examples by about 16.5×. An old 30-epoch budget then corresponds to roughly 1.8 epochs on the larger dataset, not 30.

## Historical 1,090-episode example

| Quantity | Value |
|---|---:|
| Base complete windows | 146,324 |
| Transition windows | 16,520 |
| Transition factor | 2 |
| Effective train samples | 162,844 |
| Batch size | 32 |
| Steps per epoch | 5,089 |
| Epochs | 5 |
| Total steps | 25,445 |
| 5% warmup | 1,272 |
| Sample presentations | 814,220 |

Tail padding and lift oversampling later changed the effective count; see the [data guide](DATA_PIPELINE.md). Recalculate from the actual dataset/index rather than reusing this table.

## Schedule and continuation

The documented default is linear warmup for 5% of planned steps when not explicitly configured, followed by cosine decay.

Stopping a 30-epoch run at epoch 5 is not equivalent to training a complete five-epoch schedule. The former is still near the early part of a longer schedule; the latter reaches its cosine endpoint.

If cosine has reached zero, extending the epoch count changes the experiment. Define a new continuation stage with an explicit schedule instead.

## Frozen versus 0.1× backbone

```text
head_lr = base_lr
backbone_lr = base_lr * backbone_lr_multiplier
```

| Setting | freeze_encoder_backbone | backbone_lr_multiplier |
|---|---|---|
| Frozen | true | 0.0 |
| 0.1× fine-tuning | false | 0.1 |

0.1× means one tenth of the head learning rate, not training only 10% of the backbone. Keep data/split/statistics, seed, batch, steps, warmup, augmentation, action horizon/offset, and evaluation protocol fixed.

Changing batch size changes update count and gradient noise as well as throughput. Recompute the schedule and sample exposure; linear LR scaling is a hypothesis to test, not a universal rule.

## Recorded five-epoch comparison

The source record reports both runs completed from scratch on 2026-07-21, with batch 32, 25,445 steps, base LR `1e-4`, and linear warmup plus cosine. The best recorded validation results occurred at zero-based epoch 4.

| Offline metric | Frozen | 0.1× ViT | Relative improvement |
|---|---:|---:|---:|
| val_loss | 0.024397 | 0.022643 | 7.19% |
| val_continuous_loss | 0.019114 | 0.017756 | 7.10% |
| val_keyframe_loss | 0.032670 | 0.030506 | 6.63% |
| val_sample_action_mse | 0.006680 | 0.005561 | 16.75% |

These are reported **offline** measurements; they do not establish a grasp-success improvement. See the [original experiment report](../reports/cfm_a2d_450gb_vit_freeze_ablation_20260721.md) for evidence and checkpoint hashes.

## Stop or continue?

Inspect validation loss, continuous/keyframe errors, sampled-action MSE, train/validation gap, rollout results, and current LR together. After the minimum budget, less than 1–2% improvement at two successive checks with no rollout gain can be an initial stopping heuristic.

Record step heartbeats, per-epoch metrics, atomically saved checkpoints, resolved configuration, and dataset/split/statistics provenance. A failed run should exit nonzero and preserve its failure record.

Next: [Training practices](TRAINING_TRICKS_GUIDE.md) · [Fair comparison](POLICY_COMPARISON_PROTOCOL.md)
