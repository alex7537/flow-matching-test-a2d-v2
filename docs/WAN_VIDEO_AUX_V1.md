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

On-the-fly VAE encoding is intended only for smoke tests. Formal training must
materialize frozen video latents offline; otherwise every epoch repeatedly
decodes JPEG clips and runs the 2.8 GB VAE.
