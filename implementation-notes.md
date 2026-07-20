# Implementation Notes | 实施记录

This file records implementation decisions, deviations from the agreed plan, and verification results.

本文件用于记录实施决策、相对既定计划的偏差，以及验证结果。

## Working Rule | 执行规则

- Follow the agreed implementation plan by default.
- If an edge case makes the plan unsafe, ambiguous, or infeasible, choose the most conservative option that preserves existing behavior and data.
- Record every such deviation under **Deviations | 偏差** before continuing.
- Do not silently expand scope, weaken validation, replace missing assets with guessed values, or overwrite existing artifacts.
- Keep entries concise, dated, and linked to the relevant commit, PR, issue, run, artifact, or report when available.

- 默认遵循已确认的实施计划。
- 如果极端情况导致原计划不安全、有歧义或不可执行，选择最保守、能够保留现有行为和数据的方案。
- 继续执行前，将此类偏差记录在 **Deviations | 偏差** 下。
- 不得静默扩大范围、降低验收标准、用猜测值替代缺失资产，或覆盖现有产物。
- 记录应简洁、带日期；如有对应的 commit、PR、issue、run、产物或报告，应一并注明。

## Current Plan | 当前计划

- Source of truth: `README.md`, `目标架构.md`, and the applicable task/runbook under `train+deploy/`.
- Keep the processed A2D HDF5 data contract, 13-dimensional action semantics, normalization, split, and provenance checks unchanged unless an explicit plan revision is approved.
- Preserve the shared policy interface for CFM, RS-IMLE, and Diffusion; compare policies using the frozen rollout protocol rather than cross-policy training-loss values.
- Execute deployment validation conservatively in order: Level 0 → Level 0.5 → Level 1 → Level 2 → Level 3. Do not proceed past a failed gate.
- Treat Level 4 as non-blocking and run it only after Level 3 acceptance.

## Decisions | 决策

| Date | Decision | Reason | Reference |
|---|---|---|---|
| 2026-07-20 | Add this persistent implementation log to the repository. | Make plan deviations and verification evidence explicit and reviewable. | User request |
| 2026-07-20 | Add three encoder diagnostics without adding an auxiliary encoder loss. | Observe encoder learning and collapse risk while preserving the frozen CFM/RS-IMLE/Diffusion objectives and comparison protocol. | `encoder_update_ratio`, `encoder_grad_param_ratio_mean`, `encoder_feature_std` |

## Deviations | 偏差

> Record only actual deviations from the agreed plan. If there are none, keep this section empty.
>
> 仅记录相对既定计划的实际偏差；如无偏差，保持本节为空。

| Date | Planned approach | Edge case / trigger | Conservative action taken | Impact | Follow-up / reference |
|---|---|---|---|---|---|
| 2026-07-20 | Run the repository test suite with the project virtual environment's `pytest` executable before generating the HTML change report. | `.venv/bin/pytest` is not installed, so `pytest -q` failed with “command not found” before any tests ran. | Do not install or mutate the environment automatically; first try the environment's existing interpreter with `python -m pytest`, and report the exact result. | No repository code or environment was changed. | HTML change report verification |
| 2026-07-20 | Run `python -m pytest -q` using the existing environment. | Pytest auto-loaded an incompatible user-site `anyio` plugin and failed on missing `_pytest.scope` before collection. | Keep the environment unchanged and disable third-party plugin auto-loading for this repository test run via `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. | Test semantics remain repository-local; unrelated user-site plugins are excluded. | HTML change report verification |
| 2026-07-20 | Reuse a fixed `/tmp/flow_encoder_metrics_smoke` directory for the encoder-monitoring integration smoke. | The safety layer rejected the command because it included recursive deletion of the old temporary directory. | Do not delete anything; allocate a new unique temporary output directory with `mktemp -d` and run the smoke there. | Only the output path changes; training configuration and verification semantics remain unchanged. | Encoder monitoring integration smoke |

## Verification Log | 验证记录

| Date | Scope | Command / procedure | Result | Evidence / reference |
|---|---|---|---|---|
| 2026-07-20 | Documentation | Confirmed `implementation-notes.md` did not previously exist, then created it without modifying existing tracked files. | PASS | Local working tree |
| 2026-07-20 | Repository tests for HTML change report baseline | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` | PASS — 16 passed, 1 external W&B/Sentry deprecation warning | Local test run; 2.39s |
| 2026-07-20 | `repository-change-report.html` | Parsed with Python `HTMLParser`; checked required sections, unique IDs, eight quiz questions, full-score gate, and relative link targets; extracted inline JavaScript and ran `node --check`. | PASS — HTML parse, report invariants, JavaScript syntax, and source-link targets all valid | Local verification |
| 2026-07-20 | Chinese annotation layer in `repository-change-report.html` | Added inline Chinese explanations, hover definitions, and an 18-item Chinese glossary; rechecked HTML parsing, glossary/quiz counts, and JavaScript syntax. | PASS — 18 glossary entries, 8 quiz questions, JavaScript syntax valid | Local verification |
| 2026-07-20 | Flow Matching encoder-loss audit | Traced `FlowMatchingPolicy.compute_loss()` through `_encode_obs()`, optimizer parameter groups, and epoch metrics; ran a fresh CNN gradient probe after `loss.backward()`. | PASS — no separate encoder loss exists; the encoder is trained end-to-end by `flow_loss`; all 8 backbone parameter tensors received non-zero gradients (probe L2 norm 2.1526). Existing metrics expose backbone LR and gradient norms, not encoder convergence loss. | `flow_matching_test/policies/flow_matching.py`, `flow_matching_test/policies/base.py`, `flow_matching_test/train.py` |
| 2026-07-20 | Encoder diagnostics implementation | Ran `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`, then a two-epoch CPU integration smoke on `data-rgb-complete-processed`. | PASS — 18 tests passed; both smoke epochs emitted finite `encoder_update_ratio`, `encoder_grad_param_ratio_mean`, and `encoder_feature_std`. | Smoke metrics: `/tmp/flow_encoder_metrics_smoke.EIn3o3/metrics.jsonl` |

## Open Items | 待处理事项

- [ ] Add a deviation entry whenever implementation must depart from the current plan.
- [ ] Attach verification commands and report/artifact paths after each completed acceptance level.
- [ ] Update this file in the same PR as any approved plan-changing implementation.

## Entry Template | 记录模板

```markdown
### YYYY-MM-DD — Short title

- Planned approach:
- Edge case / trigger:
- Conservative action:
- Impact:
- Verification:
- Follow-up:
- References: commit / PR / issue / run / artifact / report
```

### 2026-07-20 — CFM ViT freeze ablation

- Planned approach: Compare the same CFM training setup with a parameter-frozen pretrained ViT versus a trainable ViT using the existing 0.1× backbone learning-rate multiplier.
- Conservative definition: “Frozen ViT” means all parameters returned by backbone_parameters() have requires_grad=False and are excluded from AdamW; visual adapters, proprio path, and action policy head remain trainable.
- Controlled variables: Same A2D dataset/split, seed 42, augmentation, batch size, 30 epochs, scheduler, policy architecture, and W&B group.
- Verification gates: Repository tests → one-step frozen/fine-tuned GPU smoke → authenticated W&B online smoke → sequential full runs.
- Configs: configs/a2d_parallel_1507_cfm_a800_vit_frozen.yaml and configs/a2d_parallel_1507_cfm_a800_vit_finetune_01x.yaml.

### 2026-07-20 — A800 patch application identity

- Planned approach: Transfer the tested local commit to the clean A800 repository with git format-patch and git am so the training provenance has an exact commit SHA.
- Edge case / trigger: The A800 repository had no committer identity configured, so git am stopped before applying the patch.
- Conservative action: Abort the incomplete am operation and transfer the tested branch as a Git bundle, preserving the exact commit objects without changing local or global Git identity.
- Impact: No code or training output was produced by the failed attempt; remote branch creation is retained.
- Verification: Require remote HEAD to match the local tested commit and remote worktree to be clean before any GPU smoke.
