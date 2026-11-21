# Fair Comparison of CFM, RS-IMLE, and Diffusion Policies

**English** | [简体中文](POLICY_COMPARISON_PROTOCOL.zh-CN.md)

This protocol compares `flow_matching`, `imle`, and `diffusion`. It is a **recorded experiment protocol**, not a table of achieved success rates.

## Historical entry conditions

The source describes six episodes as implementation/overfitting probes only. Formal conclusions were gated on versioned ingestion of approximately 640 episodes, a fixed split, rollout Level 3 acceptance, and acceptance of the hand `6→11` mapping. These numbers and level names belong to that experiment stage; they are not the current dataset size or evidence that the gates passed.

## Freeze shared factors

Use the same:

- Dataset version, dataset-manifest SHA-256, statistics digest, and episode split.
- Head/right-hand RGB inputs, proprioception/action semantics, history, and prediction horizon.
- Visual backbone, Transformer width/layers/heads, augmentation, optimizer, LR schedule, batch, total optimizer steps, and seed.
- Evaluation grid, execution horizon, seeds, success criterion, and declared checkpoint-selection rule, with explicit provenance for each method's checkpoint and bundle.

Terminate at the same optimizer-step budget, not merely the same epoch count or wall-clock time.

Algorithm-specific settings may differ but must be recorded: CFM Euler steps; RS-IMLE `n_samples_per_condition` and `epsilon`; diffusion noise schedule and DDIM steps.

## Method-specific inference

A shared backbone means the same Transformer skeleton and observation pipeline. CFM/diffusion time conditioning and IMLE latent injection are intrinsic method differences.

- CFM: the recorded baseline uses five Euler steps. The source notes an earlier 5-versus-16 replay check without observed degradation; that observation is not a universal guarantee.
- Diffusion: use its DDIM inference route.
- RS-IMLE: the protocol specifies bidirectional continuity selection among 32 candidates during replanning. For setpoint semantics, default continuity weights are `arm_weight=0`, `hand_weight=1`. Record both weights and candidate count.
- Continuity selection applies when `execute_horizon < chunk_size`; the source reserves separate Level 4 comparisons at horizons 8 and 4.

The IMLE policy can generate a candidate in one pass; candidate generation and continuity selection at deployment are separate costs.

## Evaluate and report

| Measure | Required interpretation |
|---|---|
| Primary task measure | Successes / valid trials on the same recorded 182-grid evaluation, with approach/close/lift failures |
| Offline diagnostic | `val_sample_action_mse`; not the sole criterion for multimodal actions |
| Practical deployment | Warmed-up batch-1 end-to-end policy latency, median/P95, on matched A800/4090 conditions |
| Resource cost | State the actual compute/memory/cost basis alongside task performance |

Do not rank methods by flow loss, IMLE minimum-distance loss, or diffusion epsilon loss: their objective spaces differ.

The recorded W&B convention uses the team entity and groups by `policy_type`. Independently of dashboard availability, retain Git SHA, data/split/statistics digests, step budget, method-specific settings, and checkpoint/bundle provenance.

If a run differs in data, shared architecture, or step budget, mark it non-comparable and exclude it from the controlled main table. A selected-success gallery does not replace this evaluation.

Related: [Training practices](TRAINING_TRICKS_GUIDE.md) · [Deployment contract](DEPLOYMENT_OVERVIEW.md)
