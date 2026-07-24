# CFM Stage-2 10-Way Acceptance Report

- Ready: **True**
- Completed: **10/10**
- Audited: **10/10**
- Violations: **0**
- Source comparison: historical 5-step metrics; **not inference-step matched**

| Rank | Variant | Checkpoint selection | Best epoch | val sample MSE | vs source | val loss | vs source | keyframe loss | W&B |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | v03_lr5e5_bb01 | historical_val_loss | 2 | 0.00579077 | -4.13% | 0.02177454 | +3.84% | 0.02842618 | [run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/akvb08if) |
| 2 | v06_lr2e5_bb02 | historical_val_loss | 2 | 0.00582871 | -4.81% | 0.02185056 | +3.50% | 0.02851064 | [run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/fqvv3ht3) |
| 3 | v09_lr2e5_beta2_099 | historical_val_loss | 2 | 0.00587231 | -5.60% | 0.02188651 | +3.34% | 0.02858804 | [run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/klxiaif6) |
| 4 | v07_lr2e5_no_wd | historical_val_loss | 2 | 0.00587636 | -5.67% | 0.02189745 | +3.29% | 0.02858842 | [run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/uwmhhwy5) |
| 5 | v08_lr2e5_wd1e3 | historical_val_loss | 2 | 0.00588121 | -5.76% | 0.02190386 | +3.26% | 0.02860336 | [run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/59eg98ev) |
| 6 | v02_lr2e5_bb01 | historical_val_loss | 2 | 0.00588998 | -5.92% | 0.02190881 | +3.24% | 0.02861741 | [run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/6kl794j6) |
| 7 | v05_lr2e5_bb005 | historical_val_loss | 2 | 0.00589827 | -6.06% | 0.02189540 | +3.30% | 0.02857736 | [run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/cnigw5q9) |
| 8 | v01_lr1e5_bb01 | historical_val_loss | 2 | 0.00592743 | -6.59% | 0.02206783 | +2.54% | 0.02883487 | [run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/9js768tk) |
| 9 | v04_lr2e5_frozen | historical_val_loss | 2 | 0.00595209 | -7.03% | 0.02190202 | +3.27% | 0.02855849 | [run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/01x5tk6y) |
| 10 | v10_lr2e5_clip05_20s | historical_val_loss | 2 | 0.00617234 | -10.99% | 0.02188594 | +3.34% | 0.02858399 | [run](https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/gpkw2j82) |
