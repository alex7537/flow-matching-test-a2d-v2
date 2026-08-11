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


class DiffusionPolicy(FlowMatchingPolicy):
    """Noise-prediction diffusion policy with deterministic DDIM sampling."""

    def __init__(
        self,
        *,
        diffusion_train_steps: int = 100,
        diffusion_inference_steps: int = 15,
        **kwargs,
    ) -> None:
        kwargs["num_inference_steps"] = int(diffusion_inference_steps)
        super().__init__(**kwargs)
        self.diffusion_train_steps = int(diffusion_train_steps)
        self.diffusion_inference_steps = int(diffusion_inference_steps)
        if self.diffusion_train_steps <= 1:
            raise ValueError("diffusion_train_steps must be > 1")
        if not 1 <= self.diffusion_inference_steps <= self.diffusion_train_steps:
            raise ValueError("diffusion_inference_steps must be in [1, diffusion_train_steps]")
        betas = _cosine_beta_schedule(self.diffusion_train_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def _normalized_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / float(self.diffusion_train_steps - 1)

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        clean_action = batch["action"]
        obs = batch["obs"]
        if not isinstance(clean_action, torch.Tensor) or not isinstance(obs, dict):
            raise TypeError("batch must contain tensor action and dict obs")
        batch_size = clean_action.shape[0]
        timesteps = torch.randint(
            self.diffusion_train_steps,
            (batch_size,),
            device=clean_action.device,
        )
        noise = torch.randn_like(clean_action)
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
        return loss, {
            "epsilon_loss": float(loss.detach().item()),
            "diffusion_t_mean": float(timesteps.float().mean().detach().item()),
            **segment_losses,
        }

    @torch.no_grad()
    def sample_actions(self, obs: dict[str, torch.Tensor]) -> SamplingResult:
        anchor = obs[self.image_keys[0]]
        action = torch.randn(
            anchor.shape[0],
            self.action_horizon,
            self.action_dim,
            device=anchor.device,
            dtype=anchor.dtype,
        )
        timesteps = torch.linspace(
            self.diffusion_train_steps - 1,
            0,
            self.diffusion_inference_steps,
            device=anchor.device,
        ).round().long()
        for index, timestep in enumerate(timesteps):
            batch_t = timestep.expand(anchor.shape[0])
            pred_noise = self(
                noisy_action=action,
                obs=obs,
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
