# A2D V3 → RDT-170M initial adapter

This integration is an isolated transfer probe against the official
`thu-ml/RoboticsDiffusionTransformer` source at commit
`cd79363a1387e8f81c7724d070ef7e45fd23150f`.

It does not replace the verified A2D CFM/DP/IMLE baselines and does not yet
authorize a formal RDT fine-tuning run.

## Contract

```text
A2D state:  arm actual(7) + hand actual(6)
A2D action: arm actual future(7) + hand commanded target future(6)
alignment:  obs[t] -> action[t+1:t+65]
RDT state/action vector: 128
RDT image history: 2
RDT action chunk: 64
```

Mapping:

```text
A2D arm[0:7]  -> RDT right_arm_joint_pos[0:7]  -> indices 0..6
A2D hand[0:5] -> RDT right_gripper_joint[0:5]  -> indices 10..14
A2D hand[5]   -> project-reserved slot          -> index 45
```

Index 45 has no official pretrained semantic. Even the five named gripper
slots are not guaranteed to transfer to A2D dexterous-hand kinematics. The
adapter preserves all 13 values exactly; scientific transfer remains an
experiment.

Camera mapping:

```text
rgb_head       -> cam_high
rgb_right_hand -> cam_right_wrist
missing camera -> cam_left_wrist with false mask
```

The first historical frame repeats frame zero and is marked invalid at episode
start. JPEGs are decoded BGR→RGB before the SigLIP processor.

## Data-only smoke

```bash
export A2D_RDT_DATA_DIR=/path/to/a2d_v3_multitask_box300_bottle300_seed42
python integrations/rdt/smoke_a2d_adapter.py \
  --data-dir "$A2D_RDT_DATA_DIR" --split train --samples 8
python integrations/rdt/smoke_a2d_adapter.py \
  --data-dir "$A2D_RDT_DATA_DIR" --split val --samples 8
```

## RDT-170M A800 smoke

The bounded transfer probe passed against immutable official artifacts:

```text
RDT-170M revision: 8aa386cac3bbfd9540676c75b3d767cc7f88a10a
SigLIP revision:   9fdffc58afc957d1a03a25b10dba0329ab15c2a3
RDT parameters:    166,229,888
image tokens:      [1,4374,1152] (2 histories × 3 cameras × 729 patches)
state/action:      [1,1,128] / [1,64,128]
active A2D dims:   13
```

Two gates passed on one real A2D bottle sample: first with shape-correct zero
condition embeddings, then with frozen SigLIP features from the two real A2D
camera histories plus a masked background camera. Both produced a finite loss
and finite backward gradients. The second gate used the official empty language
embedding, so it validates mechanical compatibility—not task-language semantics
or rollout quality.

```bash
python integrations/rdt/smoke_rdt170m_core.py \
  --data-dir /path/to/a2d_v3_multitask_box300_bottle300_seed42 \
  --model-dir /path/to/rdt-170m/snapshot \
  --siglip-dir /path/to/siglip-so400m-patch14-384/snapshot \
  --empty-lang-embed /path/to/official-rdt/data/empty_lang_embed.pt
```

## Directly trainable data derivative

Do not rewrite the RGB payloads. Build an immutable RDT derivative with verified
hard links, train-only 128D statistics, task metadata and a 64-step time mask:

```bash
python integrations/rdt/prepare_a2d_rdt_dataset.py \
  --source /path/to/a2d_v3_multitask_box300_bottle300_seed42 \
  --output /path/to/a2d_v3_multitask_box300_bottle300_seed42__rdt_v1_emptylang \
  --empty-lang-embed /path/to/official-rdt/data/empty_lang_embed.pt
```

The verified A800 derivative contains 600 hard-linked episodes (no RGB payload
copy), 540 train / 60 val, and balanced 270/270 plus 30/30 box/bottle roles. It
has 88,687 train windows. Of the 5,675,968 possible 64-step action slots,
1,088,640 (19.2%) are tail padding and must be excluded by `action_time_mask`.

`emptylang` is deliberate: box versus bottle is inferred visually and no task
language is claimed. To study language-conditioned use genuine T5-XXL embeddings in a
new immutable data version; do not relabel the empty embedding as task text.

Apply [official_rdt_a2d_training.patch](official_rdt_a2d_training.patch) only to
the pinned upstream commit. It gives train and val separate HDF5 split roles,
propagates the time mask through collation, masks loss to 13 active A2D action
dimensions and valid time steps, and avoids importing augmentation dependencies
when augmentation is disabled.

Before official RDT training, copy `a2d_hdf5_vla_dataset.py` over the isolated
RDT checkout's `data/hdf5_vla_dataset.py`, add `a2d_v3_multitask` to its
fine-tune dataset/control-frequency/stat JSON files, and patch upstream train
construction so train and sample datasets receive distinct train/val splits.
The unmodified official HDF5 path constructs both from the same no-argument
class and therefore is not a held-out validation contract.

## Remaining gates

1. ~~Exact state/action round-trip and offset check on both split roles.~~ Passed.
2. ~~Dataset-stat generation for the 128D representation.~~ Passed from train only.
3. ~~Frozen SigLIP + empty-language smoke preparation.~~ Passed; encode the
   fixed generic task instruction before formal fine-tuning.
4. ~~RDT-170M checkpoint load without architecture overrides.~~ Passed.
5. ~~One forward/loss/backward step with finite loss and nonzero hand gradients.~~ Passed.
6. ~~Patch official HDF5 construction for separate train/val roles.~~ Passed.
7. ~~Mask loss to the 13 active A2D action dimensions.~~ Passed.
8. ~~Mask 64-step tail padding while retaining terminal lift samples.~~ Passed.
9. ~~Run a bounded optimizer smoke (for example 100 steps) before formal training.~~ Passed.
10. Pretrained versus action-randomized controlled probe before any claim that
   action pretraining helps A2D.

## First formal RDT-170M fine-tune

The first formal action-policy run uses the `emptylang` derivative and starts
from the immutable official RDT-170M checkpoint, not from the smoke checkpoint:

```text
optimizer steps:       100,000
batch size:            8
window-equivalent pass: about 9.0
peak LR:               1e-4
warmup:                fixed 5,000 steps
schedule after warmup: constant (resume-friendly)
validation:            every 1,000 steps, fixed episodes/steps/noise
checkpoint:            every 5,000 steps, keep latest 10
selection:             raw overall_avg_sample_mse
artifacts:             best_raw + best_ema_at_raw_best + periodic resume state
```

Use [launch_a2d_rdt170m_100k.sh](launch_a2d_rdt170m_100k.sh) to reproduce the
run and [watch_a2d_rdt_run.sh](watch_a2d_rdt_run.sh) for stale-log/failure
detection. Upstream prints 1,471 "epochs" because its HDF5 DataLoader length is
540 episodes; the scientifically meaningful budget is 100,000 optimizer steps
or about nine passes over the 88,687 possible train windows.
