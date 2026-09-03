# Wan video auxiliary V1

This branch adds a training-only future-video latent objective to the existing
A2D Flow Matching action policy.

## Contract

```text
action observation: rgb_head[t] + rgb_right_hand[t] + proprio[t]
video condition:    rgb_head[t-8:t]       (9 real frames)
video target:       rgb_head[t+1:t+16]   (16 real frames when available)
action target:      action[t+1:t+16]
```

The frozen Wan2.2 VAE encodes the 25-frame video to seven latent time steps.
The first three are condition latents and the final four are future targets.
Only a small future-latent head and the shared action policy receive video-loss
gradients. The VAE is loaded from an external cache, remains under `no_grad`,
and is excluded from the optimizer and checkpoint.

This deliberately differs from the current `WanOfficialObsEncoder` in
`origin/Wam_Pre_Train@7bf3519`, which keeps nine observation frames in the
dataset but sends only the final condition RGB plus sixteen future frames to
the official VAE path.

## Tail rule

- Action padding remains repeat-last plus a per-step `action_mask`.
- Video tensors remain fixed at sixteen future frames.
- If fewer than sixteen real future frames exist, `video_valid_mask=false` and
  the sample contributes zero video loss while retaining valid action loss.
- V1 does not use a partial latent-time mask.

## Loss

```text
total_loss = action_flow_loss + video_loss_weight * future_latent_mse
```

V1 uses teacher-forced clean future actions for the auxiliary context. It is an
auxiliary predictive policy, not the full joint video/action denoising WAM.

## Runtime boundary

`wan_runtime_site_packages` is appended only when the codec is first used. Do
not prepend the WAM environment to `PYTHONPATH`: that can replace the training
environment's NumPy/OpenCV before checkpoints and datasets are loaded.

On-the-fly VAE encoding is intended only for smoke tests. Formal training uses
`scripts/precompute_wan_video_latents.py` to materialize frozen video latents
offline; otherwise every epoch repeatedly decodes JPEG clips and runs the 2.8
GB VAE. The cache is bound to the dataset-manifest SHA, episode content hashes,
the VAE SHA, camera key, resize rule, and the 9+16 temporal contract. Incomplete
tail windows keep their action loss but receive zero video loss.

The cache deliberately uses deterministic resize-only video targets. RGB policy
observations keep their normal train augmentation, while the auxiliary target
does not move randomly between epochs.

## Mixed-task post-training

The prepared route upgrades a finished base CFM checkpoint; it is not scratch
training and not optimizer/scheduler resume:

```text
finished mixed-task CFM checkpoint
  -> model-only init (shared action policy weights)
  -> randomly initialized video auxiliary head
  -> new optimizer, warmup, and cosine schedule
  -> 10 epochs on the same frozen train/val split
```

Build the cache once on an idle A800:

```bash
bash scripts/precompute_multitask_wan_latents.sh
```

After the mixed-task checkpoint is available, only its path is required:

```bash
bash scripts/launch_multitask_wan_video_aux_posttrain.sh \
  /absolute/path/to/best_action_mse.ckpt
```

`init_from` validates action semantics, action offset, statistics digest, and
dataset/split manifest hashes before loading. It resets optimizer and scheduler
state by design. The exported deployment bundle contains the trained action
policy and auxiliary head state, but inference still consumes only RGB and
proprio; neither future frames nor the external Wan VAE are runtime inputs.
