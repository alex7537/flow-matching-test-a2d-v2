# First Robot-ML Loop Instance

## Scope

This repository is the first evidence-producing instance for `robot-ml-lifecycle`.
It starts at autonomy level L1: inspect, plan, and maintain local state only. It does
not authorize a training launch, TI-ONE mutation, Isaac Sim run, holdout opening,
Git push, or artifact publication.

Objective: iteratively improve the next-action (`action_offset_steps=1`) A2D policy
using a reproducible evaluation funnel, while treating Isaac Sim and future world
models as different evidence levels rather than interchangeable scores.

Source baseline:

- repository: `https://github.com/alex7537/flow-matching-test-a2d-v2`
- branch: `main`
- commit: `fd2b7512affaff00ef03165953978c8e85ab20b2`
- local checkout was clean before this instance was created

## Current evidence

- The policy consumes `rgb_head`, `rgb_right_hand`, and 13-D proprioception and
  predicts a 16-step, 13-D absolute joint-position chunk.
- CFM, RS-IMLE, and Diffusion share a policy interface, trainer, dataset contract,
  checkpoint path, and rollout bundle format.
- Existing offline evaluation reports validation loss, sampled action MSE, and
  qualitative GT/prediction curves. These are diagnostics, not task success.
- The 66-episode frozen-versus-finetuned ViT ablation used 60 train and 6 val
  episodes, batch 32, 30 epochs, and 7,380 optimizer steps. It supports keeping the
  backbone frozen at this data scale.
- Those ablation checkpoints use the legacy `action_offset_steps=0` contract and
  must not be deployed as next-action policies.
- A next-action frozen-ViT config already exists, but its completed run and artifact
  evidence are not present in this checkout.
- The Isaac rollout harness, bundle validation, 182-trial grid, stage-wise checker,
  and offline tests exist. Scored rollout is currently blocked by unfilled scene,
  prim, camera, articulation, contact-sensor, and physics calibration fields.
- Local Python lacks `pytest`; no dependency installation was authorized. Therefore
  this checkout has not independently executed the repository tests in this run.

### Development-machine snapshot (2026-08-14)

Read-only inspection of SSH target `yurui_dev_flowmatching_issac` found:

- hardware: one NVIDIA A800-SXM4-80GB; the inspected code tree is at source commit
  `fd2b7512affaff00ef03165953978c8e85ab20b2`;
- dataset `a2d_450gb_rgb_v2_exact_dedup_keep_last`: 1,090 episodes, 179,160
  frames, 981 train episodes, 109 validation episodes, no episode-name overlap, and
  complete split coverage;
- manifest SHA-256 values: dataset `9ce8fec7d9f6a286c1ab7841298eb86660cfb1102c935deccc5bf82bc0c61f82`,
  split `0700e981a297930b10a57d8e84f653f99604a3a5cf8a8bebc57b5be0be6a495c`,
  normalization stats `5ca6281497b9b4eef7106f08397e9fbec9a97cd698e7ece8fb9a4e0337608e63`,
  and derivation `7d3fe326f6c29fdeaf8cb1f5458b1ebb38af9f666b5bbf7eb8831ac88e13300b`;
- the derivation removes 927 exact duplicate-action frames while keeping the later
  frame of each pair; this is provenance evidence, not yet a semantic label audit;
- an offset-1, 16-step CFM run is active with W&B run ID `xifmcqbp`, seed 42,
  deterministic three-draw validation, and a 687,600-step/100-epoch plan. Its
  experiment removes proprioception and trains only from the two RGB streams;
- at the observation point it had completed 47 epochs (323,172 steps). Best raw
  validation loss was at epoch 10, best raw sampled-action MSE at epoch 12, and
  best EMA sampled-action MSE at epoch 21. Later training loss continued falling
  while validation loss worsened, so completion of all 100 epochs is not evidence
  of improvement;
- current best EMA sampled-action MSE is `0.0086918`, versus `0.0037390` for the
  immediately preceding 100-epoch run that retained proprioception. This is not a
  fully controlled ablation, but it is strong evidence against promoting the
  RGB-only candidate on offline metrics;
- the step-89,388 evaluation bundle is self-contained and has SHA-256
  `1ea43b38f5fbc87dd1303f7cbc2350973c3c102000243ea46d94e744ed8d4195`.
  At the same snapshot, `best.ckpt` was
  `e4bb7d5a95875dd4a0224e1eee7172e6f377b7bdf77969dc5a21ca9b9a1eb9df`,
  `best_action_mse.ckpt` was
  `4743a9e72d563a9bd5e94f254fcbbfbc08000133e37f6eda3da77fdb181ee1b5`,
  `best_ema_action_mse.ckpt` was
  `db3906619f33ce63d138149d94d525e3bdf923a4ccf675b3973d12c3ac594627`,
  and `latest.ckpt` was
  `438a1521e2dc423f58d38df8ebc5c244f1847526af8e3b471721825d7ef742da`;
  these live-run paths remain mutable identities until the run finishes;
- no project USD/USDA assets, Isaac/Omniverse process, or importable `isaacsim`,
  `omni`, or `isaaclab` package was found. Despite its SSH alias, this node is
  currently a training machine, not a calibrated E3 evaluator.

The remote root lacks a Git executable, so the commit was read from `.git/HEAD` and
its branch ref; remote working-tree cleanliness could not be independently checked.

## Evaluation funnel

Each level answers a different question. Passing a lower level never implies passing
a higher one.

### E0 — Contract and artifact validity

Question: can the exact checkpoint be loaded and executed under its declared data,
normalization, temporal, and action contracts?

Required evidence:

- Git commit, resolved config, dataset/split/stats hashes, seed, step count;
- `action_offset_steps=1`, joint order, camera keys, and bundle SHA-256;
- deterministic fixed-input/fixed-seed inference;
- finite outputs and zero action-range violations;
- bundle round-trip and offline unit tests.

Failure returns to `source`, `understand`, `split`, or `package`; it is not a model
quality result.

### E1 — Offline predictive diagnostics

Question: on held-out episodes, does the policy produce numerically plausible and
temporally useful action distributions?

Use fixed observation IDs and fixed sampling seeds. Report at least:

- mean sampled action MSE, separated by arm/hand, phase, and horizon index;
- best-of-K error and mean-of-K error for stochastic policies, with the same K;
- sample diversity, action-range violation rate, first-action discontinuity from
  current proprioception, velocity/acceleration-limit violations, and inference
  median/P95;
- checkpoint-to-checkpoint deltas on identical samples and seeds.

These metrics may shortlist checkpoints. They cannot establish grasp success, and
loss values from CFM, IMLE, and Diffusion must never be compared directly.

### E2 — Recorded-episode temporal replay

Question: does repeated replanning remain coherent along a real successful episode?

Extend the existing replay path to measure:

- next-action error under the offset-1 contract;
- overlap consistency: predictions for the same future timestep from adjacent
  observations;
- phase-transition timing error, especially approach-to-close and close-to-lift;
- per-joint drift and action smoothness across replans.

This remains open-loop because observations come from recorded trajectories. It is
cheaper and more diagnostic than Isaac, but cannot measure recovery from policy
errors or environment interaction.

### E3 — Calibrated Isaac closed-loop evaluation

Question: does the policy complete the task under a frozen simulator contract?

Promotion gates, in order:

1. camera/observation and joint-order alignment;
2. GT action replay and success-checker calibration;
3. one known training pose with the next-action bundle;
4. diagnostic rollout subset;
5. promotion suite;
6. sealed holdout, opened only after checkpoint and rules are frozen.

Report paired trial outcomes for competing checkpoints using identical seeds,
approach/close/lift failure counts, simulator-error rate, action violations,
latency, and confidence intervals. A grid repeatedly viewed or used for model
decisions is diagnostic, not a final holdout.

Do not assign the existing 182 trials to roles until scene/pose correlation keys are
known. Split by the strongest shared scene/pose group, preserve explicit manifests,
and maintain an exposure ledger.

### E4 — World-model proxy evaluation

Question: after calibration, can a learned dynamics model rank policies similarly to
Isaac or real rollouts at much lower cost?

World-model evaluation is a research track, not the current promotion authority.
Before using it for decisions:

- train or adapt an action-conditioned model to this embodiment, camera layout,
  13-D action semantics, and task distribution;
- reserve policy checkpoints and rollout cases not used to train the world model;
- measure rank correlation, calibration error, and false-positive promotion rate
  against E3/real outcomes;
- keep world-model-only scores labeled `diagnostic-only` until thresholds are frozen.

WorldGym is a useful implementation reference, but its published runners target
OpenVLA, SpatialVLA, Octo, and RT-1-X and its public model is based on other robot
datasets. It is not a drop-in evaluator for this custom policy.

## External benchmark fit

No public benchmark can score the current checkpoint unchanged: its two-camera
observation contract, custom arm/hand embodiment, 13-D absolute joint actions, and
task assets differ from public environments. Public suites are useful for testing the
algorithm after an adapter and benchmark-specific training, not for replacing the
asset-matched E3 evaluation.

| Candidate | Best use here | Limitation |
|---|---|---|
| RoboLab | Preferred future public benchmark because it uses Isaac Lab, provides automated success predicates and server/client policy integration, and supports bringing a custom Isaac-Lab-compatible robot | Still requires a robot/task adapter, assets, controller validation, and benchmark-specific training; it is not lightweight |
| ManiSkill3 | Method-level CFM/IMLE/Diffusion comparison on standardized manipulation tasks, with existing imitation-learning baselines | SAPIEN tasks, observations, actions, and robots differ from A2D; conclusions transfer to the algorithm, not directly to this deployed checkpoint |
| LIBERO | Literature comparability for imitation-learning and multitask transfer | Panda/MuJoCo contract and older dependency stack differ substantially; not the first integration target |
| robomimic | Reuse evaluation/reporting ideas and standardized demonstration-learning baselines | A framework and dataset suite rather than an evaluator for this custom embodiment |
| WorldGym | Long-term learned proxy evaluator and reference implementation | Current public runners/world model do not natively support this policy or embodiment; requires correlation calibration before any promotion use |

Recommended order: finish E0-E3 for the real A2D task, then port the policy interface
to one small RoboLab or ManiSkill task as a separate method benchmark. Do not delay
the asset-matched evaluation while attempting to support several public suites.

## Evaluation-role contract

Use four explicit roles for both episodes and rollout trials:

- `val`: training-time checkpoint and scheduler diagnostics;
- `diagnostic`: visible samples/trials used for debugging and iteration;
- `promotion`: frozen comparison suite used sparingly for candidate promotion;
- `holdout`: sealed one-time final evaluation, never visualized or used for tuning.

Any sample or trial whose images, failures, or outputs influenced a code/model
decision is permanently exposed and cannot return to holdout.

## First model-iteration loop

1. Resolve the exact offset-1 dataset manifest, split, effective sample count,
   steps per epoch, and available completed checkpoints on the training machine.
2. Establish the frozen-ViT offset-1 baseline. Recalculate its optimizer-step and
   warmup budget from the resolved loader; do not assume the legacy 7,380 steps are
   preserved after the temporal-window change.
3. Run E0, E1, and E2 with fixed samples/seeds and produce one comparison table.
4. Use the error decomposition to choose exactly one hypothesis: observation/data,
   temporal contract, model/sampler, or training schedule.
5. Change one controlled factor, train with a comparable step budget, and repeat
   E0-E2. Do not launch a sweep before the baseline is valid.
6. When the calibrated Isaac machine is ready, run E3 gates in order. Only E3 task
   evidence may promote a checkpoint for deployment.
7. Record what this case required but the generic Loop could not express; use those
   observations to refine the runner and later ledger schema.

## Loop feedback log

Capture runner/control-plane gaps as observations rather than immediately expanding
the generic framework. Initial expected gaps are:

- executor-result fields required by a real training/evaluation handoff;
- idempotent tracking of long-running external tasks and resource observations;
- evaluation-role exposure and payload-bound human decisions;
- distinction between scheduler noops and material experiment attempts;
- exact evidence needed for automatic versus human verification per phase.

The first runner implementation should be derived from repeated needs observed here,
not from hypothetical L3 behavior.

## Immediate resume point

Continue read-only evidence collection from the training and actual Isaac machines:

- freeze the active RGB-only run identity after it finishes and retain the already
  identified best-action checkpoints rather than assuming `latest.ckpt` is best;
- complete label/segmentation semantics checks beyond manifest integrity;
- Isaac scene/asset location and version, prim paths, camera calibration, joint map,
  contact sensors, and physics snapshot;
- which of the existing 182 trials have already been viewed or used for decisions.

Until these are available, the lifecycle remains L1 and evaluation beyond E2 is
blocked by external evidence rather than by missing model code.
