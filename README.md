# Generative Robot Policies for A2D Manipulation

[中文技术文档](README.zh-CN.md) · [Data pipeline](docs/DATA_PIPELINE.md) · [Vision encoder](docs/VISION_ENCODER.md) · [Change log](CHANGE.md)

This repository studies **continuous robot action generation from visual observations and robot state**. It provides a shared training and deployment contract for comparing Conditional Flow Matching, Diffusion Policy, IMLE, video-assisted learning, Joint World Action Models, and an RDT-170M transfer route.

The goal is not to present every method as equally mature. Offline training, bundle loading, and closed-loop robot performance are reported separately so that implementation progress is not mistaken for demonstrated manipulation success.

## Research question

How should a robot generate a multimodal sequence of continuous arm and hand actions from multi-view RGB observations—and how can different generative policies be compared fairly under the same data, architecture, and evaluation protocol?

## System overview

```text
head RGB + right-hand RGB + optional proprioception
                         ↓
              shared Hybrid ViT encoder
                         ↓
             observation-condition tokens
                         ↓
   CFM / Diffusion / IMLE / Video-Aux / Joint WAM / RDT
                         ↓
             16-step × 13-D action chunk
                         ↓
            Isaac-based closed-loop rollout
```

### Shared action contract

```text
observation at t:
  head camera RGB
  right-hand camera RGB
  arm + hand actual joint state (optional)

training target:
  action[t+1 : t+17]
  arm actual joints (7) + hand commanded targets (6)

model output:
  [batch, 16, 13]
```

## Policy routes

| Route | Learning objective | Inference | Verified status |
|---|---|---|---|
| Conditional Flow Matching | Velocity from Gaussian noise to action | 5-step Euler / ODE | Main baseline; historical training complete, new task training tracked separately |
| Diffusion Policy | Predict Gaussian noise at discrete noise levels | 15-step DDIM | Formal offline training complete on the historical V3 dataset |
| IMLE | Select and regress the closest action among latent candidates | One-shot latent generation | Formal offline training complete on the historical V3 dataset |
| Video-Aux | CFM action loss + frozen Wan2.2-VAE future-latent loss | Action policy only at deployment | Implementation and smoke tests complete; long training pending |
| Joint WAM | Joint flow objective over future-video latent and action | Joint video/action ODE | Experimental branch |
| RDT-170M | Masked clean-action regression | 5-step DPM-Solver | Transfer and deployment-adapter branch |

Loss values across these routes are **not directly comparable**. Fair comparison requires the same dataset version, episode split, action contract, visual encoder, optimizer-step budget, random seed, rollout initialization, and success criterion.

## Model backbone

The default observation encoder uses a shared Hybrid ViT for both RGB views:

```text
RGB [B, 1, 3, 224, 224]
        ↓
ResNetV2-26 feature map [B, 2048, 7, 7]
        ↓ 1×1 projection
49 spatial tokens per camera, dimension 384
        ↓
98 visual tokens + optional proprioception token
        ↓
4-layer Action Transformer
```

The `7×7` grid is a spatial image-feature grid, not a segmentation output or a robot XYZ coordinate system. See [Vision encoder documentation](docs/VISION_ENCODER.md) for tensor shapes and the spatial-token/CLS ablation boundary.

## Current experiment

The current main experiment trains the unchanged CFM baseline from scratch on the `b23v2` task distribution:

- 654 episodes
- 108,950 frames after exact duplicate removal
- episode-level train/validation split
- dual RGB + current proprioception
- transition/lift oversampling
- no enhanced-proprioception features and no explicit action-smoothing loss

Configuration: [`configs/a2d_b23v2_v3_hybrid_hand_target_cfm_a800_100ep_scratch.yaml`](configs/a2d_b23v2_v3_hybrid_hand_target_cfm_a800_100ep_scratch.yaml)

The README records that the run was launched; final claims require the completed summary, checkpoint provenance, and closed-loop rollout results.

## Experimental branches

| Branch | Purpose |
|---|---|
| [`feat/v3-wan-video-aux-v1`](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/feat/v3-wan-video-aux-v1) | Frozen Wan2.2-VAE future-video auxiliary objective |
| [`feat/a2d-v3-joint-latent-wam-scratch-v1`](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/feat/a2d-v3-joint-latent-wam-scratch-v1) | Joint video-latent/action Flow Matching |
| [`feat/a2d-rdt170m-adapter`](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/feat/a2d-rdt170m-adapter) | RDT-170M data mapping, training patch, and deployment bundle |
| [`feat/task-level-grasp-retry`](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/feat/task-level-grasp-retry) | Attempt → verify → recover → retry execution state machine |
| [`docs/grasp-success-gallery`](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/docs/grasp-success-gallery) | Rollout videos, GIFs, and grasp-result gallery |

## Training and deployment

Start the current CFM experiment with:

```bash
python3 -u -m flow_matching_test.train \
  --config configs/a2d_b23v2_v3_hybrid_hand_target_cfm_a800_100ep_scratch.yaml
```

Training artifacts include resolved configuration, step metrics, latest/best checkpoints, a final summary, and an explicit failure record when a run terminates unsuccessfully.

Deployment is maintained in [`alex7537/fk-issac-logistics`](https://github.com/alex7537/fk-issac-logistics). A policy that loads successfully is not automatically described as grasp-capable; closed-loop rollout is the final evaluation boundary.

## Documentation

- [Chinese technical record](README.zh-CN.md): full experiment history and implementation details
- [Data pipeline](docs/DATA_PIPELINE.md): preprocessing, action labels, deduplication, padding, masks, and sampling
- [Training planning](docs/TRAINING_PLANNING_GUIDE.md): epochs, optimizer steps, warmup, and learning rates
- [Training practices](docs/TRAINING_TRICKS_GUIDE.md): monitoring, stability checks, and stopping decisions
- [Vision encoder](docs/VISION_ENCODER.md): architecture, tensor shapes, spatial tokens, and CLS ablations
- [Artifact index](artifacts_index.md): external checkpoints, bundles, and experiment outputs

Large datasets, checkpoints, local experiment logs, and deployment bundles are intentionally excluded from Git.
