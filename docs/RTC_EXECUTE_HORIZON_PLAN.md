# RTC-guided execute-horizon improvement plan

Status: design proposal. No RTC runtime or training change is implemented by this document.

## 1. Motivation

The current A2D policy predicts a 16-step absolute-joint action chunk. Deployment chooses an
`execute_horizon <= 16`, executes that prefix, observes again, and samples a new chunk.

The expected benefit of a shorter execute horizon is more frequent feedback. The existing diagnostic
rollouts show that this benefit is not monotonic: `horizon=4` can underperform `horizon=8/16`, with
the earliest visible degradation concentrated in hand-command readiness, target-object contact, and
grasp/lift persistence rather than only arm arrival.

The current CFM rollout path also creates a new Gaussian noise sample for every policy call. CFM does
not condition the new chunk on the previous chunk suffix; the previous-chunk continuity path is
currently specific to RS-IMLE. Therefore a shorter execute horizon also creates more independent
generative boundaries and more opportunities to switch action modes.

This plan evaluates that mechanism before implementing full real-time chunking (RTC), then adds
cross-chunk continuity and asynchronous execution in controlled stages.

## 2. Current diagnostic evidence

The v16/v17 handoff snapshots were paused before the requested 500 episodes per configuration. They
are diagnostic evidence, not a final benchmark. No runtime/RPC errors were present in the captured
records.

For the same `hand_target_best_ema` bundle:

| Snapshot | Execute horizon | Completed | Sustained 5 cm success | Hand command ready | Target contact ready | Median 5 cm grasp/lift streak |
|---|---:|---:|---:|---:|---:|---:|
| v16 | 4 | 58 | 33/58 (56.9%) | 74.1% | 62.1% | 31.5 frames |
| v16 | 8 | 58 | 43/58 (74.1%) | 87.9% | 77.6% | 70.0 frames |
| v16 | 16 | 57 | 39/57 (68.4%) | 91.2% | 78.9% | 53.0 frames |
| v17 | 4 | 25 | 13/25 (52.0%) | 80.0% | 60.0% | 42.0 frames |
| v17 | 8 | 24 | 18/24 (75.0%) | 95.8% | 79.2% | 47.5 frames |
| v17 | 16 | 24 | 18/24 (75.0%) | 91.7% | 79.2% | 85.5 frames |

Matched-seed diagnostics point in the same direction for best EMA:

- v16: 8 seeds succeed only at H4, while 15 succeed only at H16;
- v17: 5 seeds succeed only at H4, while 10 succeed only at H16;
- insufficient target-contact patterns are more frequent at H4 in both snapshots.

The effect is checkpoint-dependent: one v16 raw checkpoint performs slightly better at H4. The
working conclusion is not “longer is always better”; it is that fixed short horizons do not reliably
convert additional observations into useful corrections.

## 3. Boundary quantities to measure

Let the previous predicted chunk be

```text
A_old = [a_0, a_1, ..., a_15]
```

and let `h` actions have been executed. After replanning, the new chunk is

```text
A_new = [ã_0, ã_1, ..., ã_15].
```

Record two distinct boundary errors:

```text
command jump: J_command = ||ã_0 - a_(h-1)||
plan disagreement: J_plan = ||ã_0 - a_h||, when h < 16
```

Also compare the new first command with the measured joint state at the boundary:

```text
state jump: J_state = ||ã_0 - q_actual||.
```

Report arm 7D and hand 6D values separately. For each boundary, retain:

- episode, seed, checkpoint/bundle SHA-256, chunk index, execute horizon, and policy call index;
- old chunk, executed prefix, unexecuted suffix, new chunk, and actual 13D qpos;
- CFM noise seed or latent identity;
- action velocity, acceleration, jerk, range/saturation, and per-joint maximum jump;
- inference latency, observation timestamp, action timestamp, and achieved control cadence;
- grasp stage, contact pattern, object height, lift streak, and terminal failure stage.

The current summary JSON/CSV files do not contain full chunks or boundary metrics, so they cannot
directly prove that chunk discontinuity is the cause. Phase 0 closes that evidence gap.

## 4. Design principles

1. **Preserve feedback.** New observations must be able to change future actions.
2. **Preserve local intent.** A new chunk must not arbitrarily switch the strategy already in progress.
3. **Separate latency from continuity.** First validate cross-chunk consistency synchronously; add
   asynchronous execution only after the boundary mechanism works.
4. **Do not average incompatible modes blindly.** Linear blending can reduce numeric jumps while
   producing an invalid path between two valid strategies.
5. **Keep training and inference contracts explicit.** A CFM interior start with pure Gaussian noise is
   not a valid warm start; any suffix-conditioned warm start or guidance must match the trained path.
6. **Treat horizon as phase-dependent.** A long coherent transit and a contact transition need not use
   the same replanning interval.

## 5. Staged implementation

### Phase 0 — Instrument the existing rollout

Do not change generated actions. Add append-only boundary records and summary statistics for
`J_command`, `J_plan`, `J_state`, arm/hand jerk, inference latency, and failure stage.

Acceptance:

- current H4/H8/H16 success and funnel metrics reproduce under matched seeds;
- every replan boundary has complete old/new/action/state identities;
- instrumentation does not change sampled chunks for a frozen seed.

### Phase 1 — Diagnose stochastic mode switching

Compare the same frozen bundle under:

1. current call-dependent noise (`base_seed + policy_call`);
2. one fixed latent/noise identity per episode across replans;
3. matched deterministic latent schedules shared across horizons.

This is a diagnostic ablation, not the final deployment mode. If fixed latent materially reduces
boundary disagreement and recovers H4 contact/lift, frequent mode switching is a supported cause.

### Phase 2 — Add continuity baselines

Evaluate in increasing order of complexity:

1. post-hoc command clamp or short blend, used only as a numeric smoothness baseline;
2. hard committed prefix using actions that are already guaranteed to execute;
3. previous-suffix conditioning or compatible warm start;
4. soft overlap guidance during CFM integration.

Do not promote a method for lower jerk alone. It must preserve target contact, sustained lift, final
retention, action range, and responsiveness to changed observations.

### Phase 3 — RTC-style guided inpainting

Retain the unexecuted suffix of the previous chunk as target `Y`. During CFM integration, guide the
estimated clean chunk toward `Y` over an overlap mask `W`:

```text
v_guided(x, o, t) = v_model(x, o, t) + lambda(t) * guidance(x, Y, W).
```

Use a soft mask:

- weight 1 for actions committed during inference delay;
- decaying weights through the editable overlap region;
- weight 0 for genuinely new future actions.

Tune arm and hand guidance separately because arm motion and contact-sensitive hand targets have
different continuity requirements. Record the guidance schedule and weight in run metadata.

### Phase 4 — Asynchronous remote inference

After synchronous continuity passes:

- keep a real-time action queue owned by the controller thread;
- run observation preprocessing and CFM inference in a background worker;
- measure inference delay `d` in achieved control steps, not nominal milliseconds only;
- freeze actions that will execute before inference finishes;
- reject stale chunks whose observation/action identity no longer matches the queue;
- swap chunks atomically and preserve an emergency hold/stop path.

Asynchronous execution must not be used to bypass simulator/server ownership or safety checks.

### Phase 5 — Phase-aware execute horizon

Once boundaries are continuous, compare fixed horizons with an adaptive rule. A first training-free
candidate is to compute the predicted arm/hand speed profile and replan near stable low-speed valleys,
while keeping longer horizons through coherent transit and lift segments.

RTC-style continuity and phase-aware horizon selection solve complementary problems:

- RTC controls how a new chunk connects to the previous one;
- phase-aware execution controls when the connection should occur.

## 6. Proposed controlled matrix

Freeze the exact bundle, weights variant, model revision, dataset/action contract, CFM 5-step Euler
solver, observation preprocessing, simulator, seeds, 30 Hz requested rate, maximum executed actions,
and `target-contact-lift-5f-v1` success profile.

| ID | Execute rule | Noise/latent | Continuity | Purpose |
|---|---|---|---|---|
| A | H16 fixed | current | none | established long-horizon baseline |
| B | H4 fixed | current | none | reproduce short-horizon regression |
| C | H4 fixed | episode-fixed | none | isolate stochastic mode switching |
| D | H4 fixed | matched | hard prefix | test minimal continuity constraint |
| E | H4 fixed | matched | RTC soft guidance | test feedback plus continuity |
| F | H8 fixed | matched | RTC soft guidance | test moderate feedback frequency |
| G | adaptive H4–H16 | matched | RTC soft guidance | phase-aware candidate |

Run a small matched-seed smoke set first. Advance only configurations with valid actions, populated
boundary logs, plausible videos, and no evaluator/RPC contamination. The qualification run must use
equal matched counts and alternate configurations episode by episode.

## 7. Promotion gates

A continuity candidate is promoted only when all applicable gates pass:

1. no schema, non-finite, action-range, RPC, or concurrent-controller invalidity;
2. boundary arm/hand `J_command` and jerk improve over vanilla H4;
3. target-contact readiness and sustained 5 cm lift do not regress against H16;
4. final retention and maximum lift streak do not hide late drops;
5. observation-to-action latency and achieved cadence satisfy the frozen runtime target;
6. responsiveness is preserved in intervention tests where the observation changes materially;
7. gains repeat across matched seeds and are not limited to one checkpoint role.

Use `promote`, `retry-eval`, `new-experiment`, or `stop` as the final decision. An incomplete or
diagnostic-only run cannot qualify a production execution mode.

## 8. Repository touchpoints

Expected implementation areas:

- `rollout/policy_wrapper.py`: previous chunk state, latent schedule, continuity config, boundary records;
- `flow_matching_test/policies/flow_matching.py`: sampling from supplied noise and guided CFM steps;
- `rollout/run_rollout.py`: queue semantics, synchronous baseline, later asynchronous worker;
- `rollout/report.py`: boundary distributions, requested/achieved cadence, funnel comparison;
- `configs/rollout_eval_grid.yaml` or a versioned RTC eval config: frozen matrix and metadata;
- `tests/test_policy_interface.py`: deterministic noise, prefix/suffix alignment, guidance boundaries;
- `tests/test_rollout_offline.py`: queue handoff, stale-result rejection, resume and failure behavior.

Keep the first change instrumentation-only. Do not combine logging, RTC guidance, asynchronous
execution, and adaptive horizon in one patch.

## 9. Open decisions

- Should continuity target the previous commanded suffix, predicted suffix, measured actual qpos, or
  a hybrid of them under absolute-setpoint semantics?
- Should arm and hand use different overlap weights near contact?
- How should observation, inference, and action timestamps be synchronized across remote RPC?
- What maximum inference delay is permitted before a chunk is rejected as stale?
- Should adaptive replanning use only kinematics, or also contact/lift stage signals?
- Does the enhanced-proprio policy reduce the need for guidance, or complement it?

Resolve these with frozen ablations rather than assumptions.

## 10. References

- Kevin Black, Manuel Y. Galliker, Sergey Levine, “Real-Time Execution of Action Chunking Flow
  Policies,” NeurIPS 2025: <https://arxiv.org/abs/2506.07339>
- Junnan Nie et al., “PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking,”
  2026: <https://arxiv.org/abs/2606.00537>
- `docs/POLICY_COMPARISON_PROTOCOL.md`
- `train+deploy/ROLLOUT_DEPLOYMENT_PLAN.md`
- `scripts/audit_action_continuity.py`
