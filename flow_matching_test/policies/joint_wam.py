from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching_test.policies.base import SamplingResult, masked_action_mse_per_sample
from flow_matching_test.policies.flow_matching import FlowMatchingPolicy


@dataclass(frozen=True)
class JointWamSamplingResult(SamplingResult):
    video_latent: torch.Tensor


class JointLatentWamPolicy(FlowMatchingPolicy):
    """Compact joint Flow Matching model over Wan video latents and A2D actions."""

    def __init__(
        self,
        *,
        video_latent_channels: int = 48,
        video_condition_latent_steps: int = 3,
        video_future_latent_steps: int = 4,
        video_latent_spatial_size: int = 14,
        video_patch_size: int = 2,
        action_loss_weight: float = 1.0,
        video_loss_weight: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.video_latent_channels = int(video_latent_channels)
        self.video_condition_latent_steps = int(video_condition_latent_steps)
        self.video_future_latent_steps = int(video_future_latent_steps)
        self.video_latent_spatial_size = int(video_latent_spatial_size)
        self.video_patch_size = int(video_patch_size)
        self.action_loss_weight = float(action_loss_weight)
        self.video_loss_weight = float(video_loss_weight)
        if min(
            self.video_latent_channels,
            self.video_condition_latent_steps,
            self.video_future_latent_steps,
            self.video_latent_spatial_size,
            self.video_patch_size,
        ) <= 0:
            raise ValueError("joint WAM video dimensions must be positive")
        if self.video_latent_spatial_size % self.video_patch_size != 0:
            raise ValueError("video latent spatial size must be divisible by patch size")
        if self.action_loss_weight < 0.0 or self.video_loss_weight < 0.0:
            raise ValueError("joint WAM loss weights must be non-negative")

        self.video_patch_embed = nn.Conv2d(
            self.video_latent_channels,
            self.d_model,
            kernel_size=self.video_patch_size,
            stride=self.video_patch_size,
        )
        self.video_velocity_head = nn.Linear(
            self.d_model,
            self.video_latent_channels * self.video_patch_size**2,
        )
        patches_per_step = (self.video_latent_spatial_size // self.video_patch_size) ** 2
        condition_tokens = self.video_condition_latent_steps * patches_per_step
        future_tokens = self.video_future_latent_steps * patches_per_step
        self.video_condition_pos = nn.Parameter(
            torch.randn(1, condition_tokens, self.d_model) * 0.02
        )
        self.video_future_pos = nn.Parameter(
            torch.randn(1, future_tokens, self.d_model) * 0.02
        )
        self.video_modality = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.action_modality = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

    @property
    def video_tokens_per_step(self) -> int:
        return (self.video_latent_spatial_size // self.video_patch_size) ** 2

    def _validate_latents(
        self,
        condition: torch.Tensor,
        future: torch.Tensor,
        future_mask: torch.Tensor | None = None,
    ) -> None:
        expected_condition = (
            self.video_latent_channels,
            self.video_condition_latent_steps,
            self.video_latent_spatial_size,
            self.video_latent_spatial_size,
        )
        expected_future = (
            self.video_latent_channels,
            self.video_future_latent_steps,
            self.video_latent_spatial_size,
            self.video_latent_spatial_size,
        )
        if condition.ndim != 5 or tuple(condition.shape[1:]) != expected_condition:
            raise ValueError(
                f"video_condition_latent must be [B,{','.join(map(str, expected_condition))}]"
            )
        if future.ndim != 5 or tuple(future.shape[1:]) != expected_future:
            raise ValueError(
                f"video_future_latent must be [B,{','.join(map(str, expected_future))}]"
            )
        if condition.shape[0] != future.shape[0]:
            raise ValueError("condition/future video latent batch sizes must match")
        if future_mask is not None and future_mask.shape != future.shape[:1] + future.shape[2:3]:
            raise ValueError("video_future_mask must have shape [B,T_latent]")

    def _patchify_video(self, latent: torch.Tensor) -> torch.Tensor:
        batch, channels, steps, height, width = latent.shape
        frames = latent.permute(0, 2, 1, 3, 4).reshape(
            batch * steps, channels, height, width
        )
        tokens = self.video_patch_embed(frames)
        return tokens.flatten(2).transpose(1, 2).reshape(batch, -1, self.d_model)

    def _unpatchify_video(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        grid = self.video_latent_spatial_size // self.video_patch_size
        expected_tokens = self.video_future_latent_steps * grid * grid
        if tokens.shape[1] != expected_tokens:
            raise ValueError("future video token count does not match configured latent geometry")
        frames = self.video_velocity_head(tokens).reshape(
            batch * self.video_future_latent_steps,
            grid,
            grid,
            self.video_latent_channels * self.video_patch_size**2,
        )
        frames = frames.permute(0, 3, 1, 2).contiguous()
        frames = F.pixel_shuffle(frames, self.video_patch_size)
        return frames.reshape(
            batch,
            self.video_future_latent_steps,
            self.video_latent_channels,
            self.video_latent_spatial_size,
            self.video_latent_spatial_size,
        ).permute(0, 2, 1, 3, 4)

    def _predict_joint_velocity(
        self,
        *,
        noisy_action: torch.Tensor,
        noisy_video: torch.Tensor,
        condition_video: torch.Tensor,
        video_future_mask: torch.Tensor,
        obs: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_latents(condition_video, noisy_video, video_future_mask)
        obs_tokens = self._encode_obs(obs)
        condition_tokens = self._patchify_video(condition_video)
        condition_tokens = (
            condition_tokens
            + self.video_condition_pos[:, : condition_tokens.shape[1]]
            + self.video_modality
        )
        cond_tokens = torch.cat([obs_tokens, condition_tokens], dim=1)

        video_tokens = self._patchify_video(noisy_video)
        video_tokens = (
            video_tokens
            + self.video_future_pos[:, : video_tokens.shape[1]]
            + self.video_modality
        )
        action_tokens = (
            self.action_proj(noisy_action)
            + self.action_pos[:, : noisy_action.shape[1]]
            + self.action_modality
        )
        video_token_count = video_tokens.shape[1]
        joint_tokens = torch.cat([video_tokens, action_tokens], dim=1)
        joint_tokens = joint_tokens + self.time_embed(timesteps.float()).unsqueeze(1)
        invalid_video_tokens = (~video_future_mask.bool()).repeat_interleave(
            self.video_tokens_per_step, dim=1
        )
        padding_mask = torch.cat(
            [
                invalid_video_tokens,
                torch.zeros(
                    noisy_action.shape[:2],
                    device=noisy_action.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        for block in self.blocks:
            joint_tokens = block(
                joint_tokens,
                cond_tokens,
                self_key_padding_mask=padding_mask,
            )
        joint_tokens = self.final_norm(joint_tokens)
        video_velocity = self._unpatchify_video(joint_tokens[:, :video_token_count])
        action_velocity = self.head(joint_tokens[:, video_token_count:])
        return video_velocity, action_velocity

    @staticmethod
    def _masked_video_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError("video prediction/target must share [B,C,T,H,W] shape")
        if valid_mask.shape != target.shape[:1] + target.shape[2:3]:
            raise ValueError("video valid mask must have shape [B,T]")
        per_step = (prediction.float() - target.float()).square().mean(dim=(1, 3, 4))
        mask = valid_mask.to(device=prediction.device, dtype=per_step.dtype)
        denominator = mask.sum()
        if not bool(denominator.item() > 0):
            return prediction.sum() * 0.0
        return (per_step * mask).sum() / denominator

    def _compute_joint_loss(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        *,
        action_noise: torch.Tensor,
        video_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        obs = batch.get("obs")
        clean_action = batch.get("action")
        clean_video = batch.get("video_future_latent")
        video_mask = batch.get("video_future_mask")
        if not isinstance(obs, dict):
            raise TypeError("joint WAM batch['obs'] must be a dict")
        condition_video = obs.get("video_condition_latent")
        tensors = (clean_action, clean_video, video_mask, condition_video)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise KeyError(
                "joint WAM requires action, video_future_latent, video_future_mask, "
                "and obs.video_condition_latent"
            )
        assert isinstance(clean_action, torch.Tensor)
        assert isinstance(clean_video, torch.Tensor)
        assert isinstance(video_mask, torch.Tensor)
        assert isinstance(condition_video, torch.Tensor)
        self._validate_latents(condition_video, clean_video, video_mask)
        t_action = timesteps[:, None, None].to(clean_action.dtype)
        t_video = timesteps[:, None, None, None, None].to(clean_video.dtype)
        noisy_action = (1.0 - t_action) * action_noise + t_action * clean_action
        noisy_video = (1.0 - t_video) * video_noise + t_video * clean_video
        target_action_velocity = clean_action - action_noise
        target_video_velocity = clean_video - video_noise
        pred_video, pred_action = self._predict_joint_velocity(
            noisy_action=noisy_action,
            noisy_video=noisy_video,
            condition_video=condition_video,
            video_future_mask=video_mask,
            obs=obs,
            timesteps=timesteps,
        )
        action_mask = batch.get("action_mask")
        if action_mask is not None and not isinstance(action_mask, torch.Tensor):
            raise TypeError("action_mask must be a tensor")
        action_loss = masked_action_mse_per_sample(
            pred_action, target_action_velocity, action_mask
        ).mean()
        video_loss = self._masked_video_mse(pred_video, target_video_velocity, video_mask)
        weighted_action = self.action_loss_weight * action_loss
        weighted_video = self.video_loss_weight * video_loss
        total = weighted_action + weighted_video
        metrics: dict[str, float | None] = {
            "flow_loss": float(action_loss.detach().item()),
            "action_flow_loss": float(action_loss.detach().item()),
            "video_flow_loss": float(video_loss.detach().item()),
            "weighted_action_flow_loss": float(weighted_action.detach().item()),
            "weighted_video_flow_loss": float(weighted_video.detach().item()),
            "video_latent_valid_fraction": float(video_mask.float().mean().detach().item()),
            "t_mean": float(timesteps.mean().detach().item()),
        }
        return total, metrics

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        clean_action = batch.get("action")
        clean_video = batch.get("video_future_latent")
        if not isinstance(clean_action, torch.Tensor) or not isinstance(clean_video, torch.Tensor):
            raise KeyError("joint WAM requires action and video_future_latent tensors")
        batch_size = clean_action.shape[0]
        timesteps = torch.rand(batch_size, device=clean_action.device)
        timesteps = timesteps * (1.0 - self.time_eps) + self.time_eps
        return self._compute_joint_loss(
            batch,
            action_noise=torch.randn_like(clean_action),
            video_noise=torch.randn_like(clean_video),
            timesteps=timesteps,
        )

    def compute_loss_seeded(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        seeds: list[int],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        clean_action = batch.get("action")
        clean_video = batch.get("video_future_latent")
        if not isinstance(clean_action, torch.Tensor) or not isinstance(clean_video, torch.Tensor):
            raise KeyError("joint WAM requires action and video_future_latent tensors")
        if len(seeds) != clean_action.shape[0]:
            raise ValueError("validation seed count must match batch size")
        action_rows, video_rows, timestep_rows = [], [], []
        for seed in seeds:
            generator = torch.Generator(device=clean_action.device).manual_seed(int(seed))
            action_rows.append(torch.randn(
                (1, *clean_action.shape[1:]),
                device=clean_action.device,
                dtype=clean_action.dtype,
                generator=generator,
            ))
            video_rows.append(torch.randn(
                (1, *clean_video.shape[1:]),
                device=clean_video.device,
                dtype=clean_video.dtype,
                generator=generator,
            ))
            timestep_rows.append(torch.rand((), device=clean_action.device, generator=generator))
        timesteps = torch.stack(timestep_rows) * (1.0 - self.time_eps) + self.time_eps
        return self._compute_joint_loss(
            batch,
            action_noise=torch.cat(action_rows),
            video_noise=torch.cat(video_rows),
            timesteps=timesteps,
        )

    def _sample_joint_from_noise(
        self,
        *,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        video: torch.Tensor,
    ) -> JointWamSamplingResult:
        condition = obs.get("video_condition_latent")
        if not isinstance(condition, torch.Tensor):
            raise KeyError("joint WAM inference requires obs.video_condition_latent")
        batch_size = action.shape[0]
        video_mask = torch.ones(
            batch_size,
            self.video_future_latent_steps,
            device=action.device,
            dtype=torch.bool,
        )
        dt = (1.0 - self.time_eps) / float(self.num_inference_steps)
        for step_index in range(self.num_inference_steps):
            timestep = torch.full(
                (batch_size,),
                self.time_eps + dt * step_index,
                device=action.device,
            )
            video_velocity, action_velocity = self._predict_joint_velocity(
                noisy_action=action,
                noisy_video=video,
                condition_video=condition,
                video_future_mask=video_mask,
                obs=obs,
                timesteps=timestep,
            )
            action = action + action_velocity.to(action.dtype) * dt
            video = video + video_velocity.to(video.dtype) * dt
        action_normalized = action.float()
        return JointWamSamplingResult(
            action_normalized=action_normalized,
            action=self.denormalize_action(action_normalized),
            video_latent=video.float(),
        )

    @torch.no_grad()
    def sample_actions(self, obs: dict[str, torch.Tensor]) -> JointWamSamplingResult:
        anchor = next(iter(obs.values()))
        condition = obs.get("video_condition_latent")
        if not isinstance(condition, torch.Tensor):
            raise KeyError("joint WAM inference requires obs.video_condition_latent")
        action = torch.randn(
            anchor.shape[0], self.action_horizon, self.action_dim,
            device=anchor.device, dtype=anchor.dtype,
        )
        video = torch.randn(
            anchor.shape[0], self.video_latent_channels, self.video_future_latent_steps,
            self.video_latent_spatial_size, self.video_latent_spatial_size,
            device=condition.device, dtype=condition.dtype,
        )
        return self._sample_joint_from_noise(obs=obs, action=action, video=video)

    @torch.no_grad()
    def sample_actions_seeded(
        self,
        obs: dict[str, torch.Tensor],
        seeds: list[int],
    ) -> JointWamSamplingResult:
        anchor = next(iter(obs.values()))
        condition = obs.get("video_condition_latent")
        if not isinstance(condition, torch.Tensor):
            raise KeyError("joint WAM inference requires obs.video_condition_latent")
        if len(seeds) != anchor.shape[0]:
            raise ValueError("validation seed count must match batch size")
        action_rows, video_rows = [], []
        for seed in seeds:
            generator = torch.Generator(device=anchor.device).manual_seed(int(seed))
            action_rows.append(torch.randn(
                (1, self.action_horizon, self.action_dim),
                device=anchor.device, dtype=anchor.dtype, generator=generator,
            ))
            video_rows.append(torch.randn(
                (1, self.video_latent_channels, self.video_future_latent_steps,
                 self.video_latent_spatial_size, self.video_latent_spatial_size),
                device=condition.device, dtype=condition.dtype, generator=generator,
            ))
        return self._sample_joint_from_noise(
            obs=obs,
            action=torch.cat(action_rows),
            video=torch.cat(video_rows),
        )
