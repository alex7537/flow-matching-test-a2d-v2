# A800 CFM noise-floor report

## Protocol

- Models: V03 (`lr=5e-5`, backbone `0.1x`) and V02 (`lr=2e-5`, backbone `0.1x`)
- Weights: raw historical `best.ckpt`, selected by validation loss
- Validation subset: the first 1,024 of 16,323 validation windows
- Sampling: 10 Euler inference steps, one sample draw per window
- Protocol noise: 10 deterministic seeds; each seed changes validation noise, `t`, and sampling noise
- GPU floating noise: the same seed and inputs repeated five times
- Hardware: NVIDIA A800-SXM4-80GB, PyTorch 2.4.1+cu121

## Results

| Metric | V03 mean ± std | V02 mean ± std | V03 relative to V02 | V03 seed wins |
|---|---:|---:|---:|---:|
| loss | 0.01833709 ± 0.00136822 | 0.01852519 ± 0.00136370 | -1.02% | 10/10 |
| continuous loss | 0.01561670 ± 0.00170532 | 0.01594080 ± 0.00172229 | -2.03% | 10/10 |
| keyframe loss | 0.02222813 ± 0.00262492 | 0.02226792 ± 0.00263917 | -0.18% | 6/10 |
| sampled action MSE | 0.00428893 ± 0.00016964 | 0.00445497 ± 0.00016297 | -3.73% | 10/10 |

For both models, all fixed-seed repeated aggregate metrics were exactly equal:

- maximum observed absolute delta: `0.0`
- measurement tolerance: `1e-9`
- classification: `below_measurement_resolution`

## Decision

V03 is the preferred first rollout candidate. It beats V02 on total loss and sampled-action
MSE for every matched protocol seed. The keyframe-loss difference is small and not resolved by
this 10-seed measurement, so V02 remains the useful lower-learning-rate comparison candidate.

These measurements characterize stochastic validation/inference variance on a fixed offline
subset. They do not replace Isaac rollout success-rate testing.

## Provenance

- V03 source checkpoint SHA-256:
  `cf06bd874498c028713f58e780b82beaf3d2aa174e840cb9c3416a1740197288`
- V02 source checkpoint SHA-256:
  `d93c0307fa705e4feb3551753552891bcbe78d8f4f162080ba073146f7fc4812`
- Corrected bundles record `weights_variant=raw` and
  `source_checkpoint_selection=historical_val_loss`.
- The raw noise-floor JSON retains `legacy_unspecified` from the historical checkpoint and old
  manifest; the corrected bundle manifest resolves this using the acceptance report and exact
  source checkpoint SHA.
