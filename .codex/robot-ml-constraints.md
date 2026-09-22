# Robot ML Loop Constraints — flow-matching-test-a2d-v2

## Ownership and autonomy

- Autonomy level: L1.
- One designated control node is the only ledger writer.
- Runner-driven sessions must return structured results and must not edit the
  lifecycle ledger directly.
- Manual ledger changes are allowed only while no runner process is active.

## External write gates

Explicit human authorization tied to the exact payload is required before:

- creating, starting, stopping, or modifying TI-ONE tasks;
- running paid or long-lived cloud compute;
- opening a sealed evaluation holdout;
- changing files on the Isaac evaluation machine;
- pushing Git branches, creating PRs, uploading bundles, or publishing models;
- deleting checkpoints, datasets, results, images, or remote resources.

Authentication and read-only access do not imply write authorization.

## Evaluation invariants

- Legacy `action_offset_steps=0` checkpoints are not next-action deployment
  candidates.
- CFM, IMLE, and Diffusion training losses are not cross-policy ranking metrics.
- Offline action metrics are diagnostic and do not prove closed-loop success.
- A simulator suite used to select checkpoints is diagnostic, not sealed holdout.
- Isaac scoring requires calibrated observation, action, timing, physics, and
  success-checker contracts.
- World-model scores remain diagnostic until calibrated against independent rollout
  outcomes for this embodiment and task distribution.

## Loop limits

- No unattended scheduling at L1.
- No parallel ledger writers.
- One model hypothesis per cycle.
- Do not launch a new training run until the prior candidate has E0 evidence and the
  comparison contract is frozen.
- Scheduler polling/no-change events belong in runner state, not lifecycle attempts.
- Repeated identical failures escalate instead of increasing limits.

## Current stop conditions

- Missing dataset/split/checkpoint provenance.
- Missing Isaac scene, prim, camera, contact-sensor, or physics calibration.
- Required verifier cannot run.
- Proposed action requires an unrecorded external-write approval.
