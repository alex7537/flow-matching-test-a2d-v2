# Training Practices and Ablation Guide

**English** | [简体中文](TRAINING_TRICKS_GUIDE.zh-CN.md)

This is an English guide to the training practices and experimental sequence recorded in this repository. The [Chinese record](TRAINING_TRICKS_GUIDE.zh-CN.md) retains the full operational discussion.

**Version note:** that record includes an early executed-arm/executed-hand contract and a five-epoch budget. Current V3 uses actual arm positions plus commanded hand targets as action labels; observation state remains actual joints. Use the [data contract](DATA_PIPELINE.md) and the checkpoint's resolved configuration for the dataset being evaluated.

## Core principles

1. Establish data, time alignment, and inference correctness before tuning.
2. Change one variable per experiment and connect each technique to an observed problem.
3. Compare optimizer steps and sample exposure across dataset sizes.
4. Use offline metrics for diagnosis and matched closed-loop rollouts for task conclusions.
5. Distinguish implemented features, completed offline experiments, proposed ablations, and unimplemented work.

## Before training or deployment

- Bind normalization to training-episode membership; reuse the same statistics for validation, export, and rollout. Check normalization round trips to approximately `1e-5`.
- Use `obs[t] -> action[t+1:t+1+H]` for the next-action contract. Legacy offset-0 checkpoints require explicit handling.
- Keep camera names/order, crop, resize, normalization, action semantics, and horizons consistent.
- Run inference with `model.eval()` and no training augmentation.
- Read policy sampling steps and raw/EMA selection from the bundle rather than silently overriding them.
- Version evaluation object poses, orientations, seeds, and success criteria. The source recommends an initial 20 rollouts per checkpoint, expanding to 50 or more for close candidates.
- Report successful and total valid trials, uncertainty, and failures such as approach, close, lift, timeout, and simulator error.

## Techniques and recorded status

| Technique | Purpose | Recorded status / baseline |
|---|---|---|
| Per-dimension min/max | Different arm/hand numerical ranges | Implemented; normalized range `[-1,1]` |
| Next-action offset | Avoid repeating the current joint state | Implemented; offset 1 |
| Split/statistics digests | Detect data drift and leakage | Implemented; manifests and hashes |
| Transition oversampling | Increase exposure to sparse key actions | Implemented; factor 2, ablation still required |
| RGB augmentation | Lighting and limited viewpoint variation | Implemented; effectiveness requires ablation |
| Frozen ViT | Reduce compute and adaptation | Implemented; zero backbone LR |
| 0.1× backbone LR | Adapt pretrained visual features | Historical five-epoch offline comparison completed; rollout decides task benefit |
| Warmup / cosine | Early stability and late refinement | Implemented; historical 1,272-step warmup and zero cosine endpoint |
| Gradient clipping | Limit extreme updates | Implemented; global norm 1.0 |
| Watchdog | Detect stalls/failures | Implemented; recorded timeout 300 seconds |
| EMA | Maintain smoothed alternative weights | Raw/EMA saving and validation implemented; decay 0.999 |
| Separate checkpoint criteria | Loss and action MSE may disagree | Implemented |
| Per-dimension error diagnostics | Locate arm/hand weaknesses | Listed as pending in the source guide |

This table records the guide's status; it is not a fresh test of every experimental branch.

## Learning rate, EMA, and batch size

The recorded starting point is AdamW with head LR `1e-4`, backbone LR `1e-5` for 0.1× fine-tuning, betas `(0.9,0.95)`, and weight decay `1e-4`. After establishing a baseline, a controlled head-LR scan can use `{3e-5,1e-4,3e-4}`.

EMA updates an independent shadow copy:

```text
ema = decay * ema + (1 - decay) * weight
```

It does not alter the main optimizer trajectory. Checkpoints retain raw/EMA variants; export selects one explicitly. Report their rollout behavior separately.

Before increasing batch size, benchmark 100–200 steps at candidate sizes such as 32/64/128. Record samples/second, seconds/step, memory, utilization, gradients, and stability. More available memory alone is not a reason to choose a much larger batch.

## Sampling and augmentation

The historical transition example duplicates 16,520 key-action windows once:

```text
original transition share: 16,520 / 146,324 = 11.29%
factor-2 share:            33,040 / 162,844 = 20.29%
```

This changes the training index, not the HDF5 data or physical duration. Validate marked windows manually, then compare factors 1/2/3 with other settings fixed. A lower keyframe loss without a rollout gain is not enough to justify stronger sampling.

Use augmentations that plausibly occur at deployment: mild brightness, contrast, saturation, hue, and small crops. Large translations/crops can disrupt image-to-action correspondence. The source does not recommend flips or rotations that change arm or gravity semantics.

## Policy-specific experiments

| Policy | Controlled inference experiment |
|---|---|
| CFM | Keep uniform-time sampling as baseline; compare Euler/ODE step counts such as 5/10/20 and measure latency plus success |
| Diffusion | Compare DDIM counts such as 8/16/32; evaluate raw/EMA separately |
| IMLE | Use sampled-action diagnostics and rollouts; do not compare its loss numerically against CFM/DP |

These are proposed experiment grids, not claims that all variants were completed.

## Checkpoints, monitoring, and decision order

Retain latest, best validation-loss, best action-MSE, and best EMA action-MSE artifacts as supported by the run. Historical `best.ckpt` is a compatibility alias for validation-loss selection, not a universal deployment recommendation.

Log train/validation components, sampled-action MSE, both LRs, gradient clipping statistics, fixed-sample prediction curves, heartbeat/failure records, code/configuration/data hashes, and rollout counts. Per-dimension error logging remains a target where not implemented. W&B transport failure must not lose local metrics or checkpoints.

Recommended order:

1. Audit data, normalization, time offset, cameras, bundle, and rollout protocol.
2. Compare frozen and 0.1× visual backbones at the same budget.
3. Freeze artifact provenance and evaluate candidates in matched rollouts.
4. Test head LR, EMA, oversampling, then inference-step/latency tradeoffs.
5. Compare policy families only after the shared contract is stable.

When candidates are close relative to sampling uncertainty, increase rollout trials before introducing more training variables.

## Failure triage

| Symptom | First checks |
|---|---|
| Training improves, validation does not | Overfitting, diversity, augmentation, backbone unfreezing |
| Offline improves, rollout fails | Time offset, statistics, cameras, action/execution contract |
| Jitter | Chunk execution, continuity handling, EMA and sampling ablations |
| Reaches object but does not close | Hand labels/errors and key-action coverage |
| Occasional extreme motion | Corrupt demonstrations, gradient spikes, output range |
| Smooth phases worsen as key phases improve | Excessive oversampling |
| Epoch-boundary sawtooth | Sampler, shuffle, LR, metric aggregation |
| Empty dashboard with live process | Local metrics, logging frequency, synchronization |

Next: [Budget calculations](TRAINING_PLANNING_GUIDE.md) · [Comparison protocol](POLICY_COMPARISON_PROTOCOL.md) · [Deployment](DEPLOYMENT_OVERVIEW.md)
