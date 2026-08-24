from __future__ import annotations

import math

import torch

from flow_matching_test.policies.base import SamplingResult, masked_action_mse_per_sample
from flow_matching_test.policies.flow_matching import FlowMatchingPolicy


def _cosine_beta_schedule(num_steps: int, offset: float = 0.008) -> torch.Tensor:
    grid = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((grid / num_steps + offset) / (1.0 + offset)) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1.0e-4, 0.999).float()


def _scaled_linear_beta_schedule(
    num_steps: int,
    *,
    beta_start: float = 1.0e-4,
    beta_end: float = 2.0e-2,
) -> torch.Tensor:
    scale = 1000.0 / float(num_steps)
    return torch.linspace(
        beta_start * scale,
        beta_end * scale,
        num_steps,
        dtype=torch.float64,
    ).clamp(1.0e-6, 0.999).float()


class DiffusionPolicy(FlowMatchingPolicy):
    """Noise-prediction diffusion policy with deterministic DDIM sampling."""

    def __init__(
        self,
        *,
        diffusion_train_steps: int = 100,
        diffusion_inference_steps: int = 15,
        diffusion_beta_schedule: str = "cosine",
        diffusion_beta_start: float = 1.0e-4,
        diffusion_beta_end: float = 2.0e-2,
        **kwargs,
    ) -> None:
        kwargs["num_inference_steps"] = int(diffusion_inference_steps)
        super().__init__(**kwargs)
        self.diffusion_train_steps = int(diffusion_train_steps)
        self.diffusion_inference_steps = int(diffusion_inference_steps)
        self.diffusion_beta_schedule = str(diffusion_beta_schedule)
        self.diffusion_beta_start = float(diffusion_beta_start)
        self.diffusion_beta_end = float(diffusion_beta_end)
        if self.diffusion_train_steps <= 1:
            raise ValueError("diffusion_train_steps must be > 1")
        if not 1 <= self.diffusion_inference_steps <= self.diffusion_train_steps:
            raise ValueError("diffusion_inference_steps must be in [1, diffusion_train_steps]")
        if self.diffusion_beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.diffusion_train_steps)
        elif self.diffusion_beta_schedule == "scaled_linear":
            if not 0.0 < self.diffusion_beta_start < self.diffusion_beta_end:
                raise ValueError("diffusion beta bounds must satisfy 0 < start < end")
            betas = _scaled_linear_beta_schedule(
                self.diffusion_train_steps,
                beta_start=self.diffusion_beta_start,
                beta_end=self.diffusion_beta_end,
            )
        else:
            raise ValueError(
                "diffusion_beta_schedule must be 'cosine' or 'scaled_linear'"
            )
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def _normalized_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / float(self.diffusion_train_steps - 1)

    def _compute_loss_with_randomness(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        *,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        clean_action = batch["action"]
        obs = batch["obs"]
        if not isinstance(clean_action, torch.Tensor) or not isinstance(obs, dict):
            raise TypeError("batch must contain tensor action and dict obs")
        alpha_bar = self.alpha_bars[timesteps].to(clean_action.dtype)[:, None, None]
        noisy_action = alpha_bar.sqrt() * clean_action + (1.0 - alpha_bar).sqrt() * noise
        pred_noise = self(
            noisy_action=noisy_action,
            obs=obs,
            timesteps=self._normalized_time(timesteps),
        )
        action_mask = batch.get("action_mask")
        if action_mask is not None and not isinstance(action_mask, torch.Tensor):
            raise TypeError("batch['action_mask'] must be a tensor")
        per_sample_loss = masked_action_mse_per_sample(pred_noise, noise, action_mask)
        pred_clean = (
            noisy_action - (1.0 - alpha_bar).sqrt() * pred_noise
        ) / alpha_bar.sqrt()
        per_sample_x0_mse = masked_action_mse_per_sample(
            pred_clean,
            clean_action,
            action_mask,
        )
        outside = (pred_clean.abs() > 1.0).to(clean_action.dtype)
        if action_mask is None:
            clamp_fraction = outside.mean()
        else:
            mask = action_mask.to(clean_action.dtype).unsqueeze(-1)
            clamp_fraction = (outside * mask).sum() / (
                mask.sum() * clean_action.shape[-1]
            ).clamp_min(1)
        loss = per_sample_loss.mean()
        segment_losses = {
            "static_loss": None,
            "continuous_loss": None,
            "keyframe_loss": None,
            "lift_loss": None,
        }
        segment_type = batch.get("segment_type")
        if isinstance(segment_type, torch.Tensor):
            for segment_id, name in enumerate(("static_loss", "continuous_loss", "keyframe_loss")):
                mask = segment_type == segment_id
                if mask.any():
                    segment_losses[name] = float(per_sample_loss[mask].mean().detach().item())
        is_lift = batch.get("is_lift")
        if isinstance(is_lift, torch.Tensor):
            lift_mask = is_lift.bool()
            if lift_mask.any():
                segment_losses["lift_loss"] = float(
                    per_sample_loss[lift_mask].mean().detach().item()
                )
        timestep_metrics: dict[str, float | None] = {}
        boundaries = (self.diffusion_train_steps / 3.0, 2.0 * self.diffusion_train_steps / 3.0)
        bucket_masks = (
            timesteps < boundaries[0],
            (timesteps >= boundaries[0]) & (timesteps < boundaries[1]),
            timesteps >= boundaries[1],
        )
        for name, mask in zip(("low", "mid", "high"), bucket_masks):
            timestep_metrics[f"epsilon_loss_{name}_t"] = (
                float(per_sample_loss[mask].mean().detach().item())
                if mask.any()
                else None
            )
        return loss, {
            "epsilon_loss": float(loss.detach().item()),
            "diffusion_t_mean": float(timesteps.float().mean().detach().item()),
            "pred_x0_mse": float(per_sample_x0_mse.mean().detach().item()),
            "pred_x0_clamp_fraction": float(clamp_fraction.detach().item()),
            **timestep_metrics,
            **segment_losses,
        }

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        clean_action = batch["action"]
        if not isinstance(clean_action, torch.Tensor):
            raise TypeError("batch['action'] must be a tensor")
        timesteps = torch.randint(
            self.diffusion_train_steps,
            (clean_action.shape[0],),
            device=clean_action.device,
        )
        return self._compute_loss_with_randomness(
            batch,
            timesteps=timesteps,
            noise=torch.randn_like(clean_action),
        )

    def compute_loss_seeded(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        seeds: list[int],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        clean_action = batch["action"]
        if not isinstance(clean_action, torch.Tensor):
            raise TypeError("batch['action'] must be a tensor")
        if len(seeds) != int(clean_action.shape[0]):
            raise ValueError("validation seed count must match batch size")
        timestep_rows = []
        noise_rows = []
        for seed in seeds:
            generator = torch.Generator(device=clean_action.device)
            generator.manual_seed(int(seed))
            timestep_rows.append(
                torch.randint(
                    self.diffusion_train_steps,
                    (1,),
                    device=clean_action.device,
                    generator=generator,
                )
            )
            noise_rows.append(
                torch.randn(
                    (1, *clean_action.shape[1:]),
                    device=clean_action.device,
                    dtype=clean_action.dtype,
                    generator=generator,
                )
            )
        return self._compute_loss_with_randomness(
            batch,
            timesteps=torch.cat(timestep_rows),
            noise=torch.cat(noise_rows),
        )

    def _sample_actions_from_noise(
        self,
        *,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> SamplingResult:
        anchor = obs[self.image_keys[0]]
        cond_tokens = self._encode_obs(obs)
        timesteps = torch.linspace(
            self.diffusion_train_steps - 1,
            0,
            self.diffusion_inference_steps,
            device=anchor.device,
        ).round().long()
        for index, timestep in enumerate(timesteps):
            batch_t = timestep.expand(anchor.shape[0])
            pred_noise = self._predict_from_cond_tokens(
                noisy_action=action,
                cond_tokens=cond_tokens,
                timesteps=self._normalized_time(batch_t),
            )
            alpha_bar = self.alpha_bars[timestep].to(action.dtype)
            pred_clean = (action - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            # Actions are min-max normalized to [-1, 1]. DDIM must clip each x0 estimate
            # because the near-zero terminal alpha_bar amplifies small epsilon errors.
            pred_clean = pred_clean.clamp(-1.0, 1.0)
            if index + 1 == len(timesteps):
                action = pred_clean
            else:
                previous_alpha_bar = self.alpha_bars[timesteps[index + 1]].to(action.dtype)
                action = previous_alpha_bar.sqrt() * pred_clean + (
                    1.0 - previous_alpha_bar
                ).sqrt() * pred_noise
        return SamplingResult(
            action_normalized=action,
            action=self.denormalize_action(action),
        )
