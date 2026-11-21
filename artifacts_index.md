# Model and Deployment Artifact Index

**English** | [简体中文](artifacts_index.zh-CN.md)

Large checkpoints, deployment bundles, datasets, and experiment logs are stored outside Git. This page explains the artifact roles and provenance chain; the [original registry](artifacts_index.zh-CN.md) retains the exact historical locations, full SHA-256 values, run IDs, and delivery notes.

## Provenance chain

```text
source revision + resolved training config
  -> dataset / split / normalization digests
  -> checkpoint + explicit selection criterion
  -> raw or EMA deployment bundle
  -> fixed reference input/output checks
  -> closed-loop rollout evidence
```

## Recorded artifact groups

| Date | Artifact group | What it establishes |
|---|---|---|
| 2026-07-22 | 1,090-episode CFM, 0.1× ViT, Level 0 preparation package | Historical bundle, environment lock, fixed I/O checks, and file manifests |
| 2026-07-15 | 66-episode CFM best bundle and Level 0 handoff | A selected checkpoint packaged with configuration, statistics, reference samples, and environment metadata |
| 2026-07-15 | Fixed reference input/output and A800 environment lock | Numerical and environment checks for that particular bundle |
| 2026-07-15 | Validation keypoint visualizations and CFM W&B record | Offline diagnostics and training curves; unresolved weak poses require rollout evaluation |
| 2026-07-16 | 66-episode RS-IMLE and diffusion Level 0 handoffs | Method-specific bundles with checkpoint identity, fixed test inputs, and hashes |

“Level 0 preparation” is a packaging/reproduction stage, not demonstrated grasp success. References are bundle-specific and must not be silently reused across models.

## What belongs in a bundle?

The current [exporter](flow_matching_test/export_bundle.py) writes:

| File | Purpose |
|---|---|
| `ckpt.pt` | Selected raw or EMA model weights |
| `config.yaml` | Observation, action, policy, and execution configuration |
| `norm_stats.json` | Training-bound normalization statistics |
| `manifest.json` | Source checkpoint identity, data provenance, environment, variant, and file hashes |
| `data_split.json` | Optional episode split, when present in the source checkpoint |
| `README.md` | Short source/selection description |

An optional `.tgz` packages these files. It is an inference handoff, not a full optimizer-state training-resume checkpoint.

## Registration and verification

- Record SHA-256 for files; directory deliveries need a checksum manifest or an explicit verification entry point.
- Recheck hashes after copies or moves. A stale filesystem path is not evidence that the artifact is available.
- Record the revision that actually produced the artifact, not the latest repository HEAD.
- Keep exact bytes in artifact storage; Git tracks their identity and production procedure.
- Update this registry and the change log when registering an important artifact.
- Training losses are method-specific diagnostics. Report matched rollout outcomes for cross-policy task conclusions.

The historical machine paths in the Chinese registry are not public download links. The repository alone does not provide every checkpoint needed to reproduce those runs.

Next: [Deployment overview](docs/DEPLOYMENT_OVERVIEW.md) · [Fair comparison](docs/POLICY_COMPARISON_PROTOCOL.md)
