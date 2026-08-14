# CFM Stage-2 10-Way Acceptance (2026-07-23)

## Outcome

- Queue: `cfm_stage2_10way_20260722_194500`
- Status: 10/10 runs completed strictly serially; audit violations: 0.
- Budget per run: 3 epochs, 15,267 optimizer steps, batch size 32, warmup 500.
- Common initialization: the same 5-epoch ViT 0.1× checkpoint, loaded model-only with optimizer/scheduler reset.
- Primary rollout candidate: `v03_lr5e5_bb01`.
- Secondary rollout candidate: `v06_lr2e5_bb02`.
- Control: `source_baseline_5step`.
- Do not prioritize `v10_lr2e5_clip05_20s`: 20-step sampling was slower and did not improve sample MSE or the fixed-input smoothness diagnostic.

## Offline ranking

This ranking is limited to the checkpoints that were actually saved by the historical
runs. Those checkpoints were selected by the old stochastic val-loss criterion, so a
new deterministic evaluation can correct the measurement and rollout order, but cannot
recover an unsaved action-MSE-optimal epoch from any historical run.

| Rank | Variant | CFM steps | val sample action MSE | val loss |
|---:|---|---:|---:|---:|
| 1 | v03_lr5e5_bb01 | 10 | 0.005790767 | 0.021774535 |
| 2 | v06_lr2e5_bb02 | 10 | 0.005828712 | 0.021850563 |
| 3 | v09_lr2e5_beta2_099 | 10 | 0.005872315 | 0.021886513 |
| 4 | v07_lr2e5_no_wd | 10 | 0.005876356 | 0.021897449 |
| 5 | v08_lr2e5_wd1e3 | 10 | 0.005881211 | 0.021903859 |
| 6 | v02_lr2e5_bb01 | 10 | 0.005889976 | 0.021908811 |
| 7 | v05_lr2e5_bb005 | 10 | 0.005898267 | 0.021895402 |
| 8 | v01_lr1e5_bb01 | 10 | 0.005927427 | 0.022067835 |
| 9 | v04_lr2e5_frozen | 10 | 0.005952090 | 0.021902024 |
| 10 | v10_lr2e5_clip05_20s | 20 | 0.006172337 | 0.021885944 |

The historical source checkpoint metric used 5 CFM inference steps. Its historical
`val_sample_action_mse=0.005561013` is therefore not an inference-step-matched
training-only comparison with the 10-step candidates. Use the exported 5-step and
10-step source bundles as rollout controls.

## Fixed-input smoothness diagnostic

On the same validation observation and seed, V03 had the lowest mean action-step
delta and acceleration among candidates: `0.163756 / 0.304740`. V10 measured
`0.174373 / 0.326926`. The source checkpoint measured `0.164972 / 0.314973` at
5 steps and `0.206319 / 0.400651` at 10 steps. This is a one-observation diagnostic,
not a replacement for simulator or robot rollout success rates.

## Remote artifacts

Root:

```text
/share_data/zhangyurui/flow-matching-test-a2d-v2/runs/cfm_stage2_10way_20260722_194500
```

Important files:

- `ACCEPTANCE_REPORT.md` and `acceptance_report.json`
- `TEST_ARTIFACTS.json`
- `test_bundles/` (10 candidates plus source 5-step and 10-step controls)
- `reference_inference/`
- `FINAL_SOURCE_SNAPSHOT.tgz` and `FINAL_SOURCE_PROVENANCE.json`

## Suggested rollout order

Run the same calibrated grid in this order: source 5-step control, V03, V06, then
V10 only if the first three finish. Example for V03:

```bash
QUEUE=/share_data/zhangyurui/flow-matching-test-a2d-v2/runs/cfm_stage2_10way_20260722_194500
CODE=/share_data/zhangyurui/flow-matching-test-a2d-v2/code

cd "$CODE"
/isaac-sim/python.sh rollout/run_rollout.py \
  --bundle "$QUEUE/test_bundles/v03_lr5e5_bb01" \
  --grid rollout/eval_grid.yaml \
  --out "$QUEUE/rollout_results/v03_lr5e5_bb01"

/isaac-sim/python.sh rollout/report.py \
  --in "$QUEUE/rollout_results/v03_lr5e5_bb01"
```

Before using the command, confirm that `rollout/eval_grid.yaml` contains the
calibrated scene USD, prim paths, DOF names, cameras, and contact sensors for the
target test machine.
