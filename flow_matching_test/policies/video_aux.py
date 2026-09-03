from __future__ import annotations

import site
import sys
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn

from flow_matching_test.policies.flow_matching import FlowMatchingPolicy


class VideoLatentCodec(Protocol):
    latent_channels: int
    condition_latent_steps: int
    future_latent_steps: int

    def encode_condition_future(
        self,
        condition: torch.Tensor,
        future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class FrozenWanVaeCodec:
    """Lazy, checkpoint-external Wan VAE used only to build frozen targets."""

    _runtime_cache: dict[tuple[str, str, str], object] = {}

    def __init__(
        self,
        *,
        vae_checkpoint_path: str,
        runtime_repo: str,
        runtime_site_packages: str = "",
        condition_steps: int = 9,
        future_steps: int = 16,
        dtype: str = "bfloat16",
        latent_channels: int = 48,
        encode_batch_size: int = 1,
    ) -> None:
        self.vae_checkpoint_path = str(vae_checkpoint_path)
        self.runtime_repo = str(runtime_repo)
        self.runtime_site_packages = str(runtime_site_packages)
        self.condition_steps = int(condition_steps)
        self.future_steps = int(future_steps)
        self.dtype_name = str(dtype)
        self.latent_channels = int(latent_channels)
        self.encode_batch_size = int(encode_batch_size)
        if self.encode_batch_size <= 0:
            raise ValueError("video codec encode_batch_size must be positive")
        if self.condition_steps <= 0 or self.future_steps <= 0:
            raise ValueError("video condition/future steps must be positive")
        if (self.condition_steps - 1) % 4 != 0:
            raise ValueError("video_condition_steps must satisfy 4n+1")
        if self.future_steps % 4 != 0:
            raise ValueError("video_future_steps must be divisible by 4")
        if (self.condition_steps + self.future_steps - 1) % 4 != 0:
            raise ValueError("condition+future video length must satisfy 4n+1")
        self.condition_latent_steps = 1 + (self.condition_steps - 1) // 4
        self.future_latent_steps = self.future_steps // 4

    def __deepcopy__(self, memo):
        del memo
        return self

    def _dtype(self) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            return mapping[self.dtype_name.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported Wan VAE dtype: {self.dtype_name}") from exc

    def _runtime(self, device: torch.device):
        if device.type != "cuda":
            raise RuntimeError("Wan video auxiliary training requires CUDA")
        checkpoint = Path(self.vae_checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Wan VAE checkpoint not found: {checkpoint}")
        runtime_repo = Path(self.runtime_repo).expanduser().resolve()
        if not runtime_repo.is_dir():
            raise FileNotFoundError(f"Wan runtime repository not found: {runtime_repo}")
        if self.runtime_site_packages:
            runtime_site_packages = Path(self.runtime_site_packages).expanduser().resolve()
            if not runtime_site_packages.is_dir():
                raise FileNotFoundError(
                    f"Wan runtime site-packages not found: {runtime_site_packages}"
                )
            site.addsitedir(str(runtime_site_packages))
        runtime_repo_str = str(runtime_repo)
        if runtime_repo_str not in sys.path:
            sys.path.insert(0, runtime_repo_str)
        key = (str(checkpoint), str(device), str(self._dtype()))
        runtime = self._runtime_cache.get(key)
        if runtime is None:
            try:
                from psi_policy.model.wan.runtime import Wan2_2_VAE
            except ImportError as exc:
                raise ImportError(
                    "Could not import psi_policy Wan runtime. Add the WAM repository and "
                    "its site-packages to PYTHONPATH before training."
                ) from exc
            runtime = Wan2_2_VAE(
                vae_pth=str(checkpoint),
                dtype=self._dtype(),
                device=str(device),
            )
            self._runtime_cache[key] = runtime
        return runtime

    @torch.no_grad()
    def encode_condition_future(
        self,
        condition: torch.Tensor,
        future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if condition.ndim != 5 or future.ndim != 5:
            raise ValueError("video condition/future must be [B,T,3,H,W]")
        if condition.shape[1] != self.condition_steps:
            raise ValueError(
                f"expected {self.condition_steps} condition frames, got {condition.shape[1]}"
            )
        if future.shape[1] != self.future_steps:
            raise ValueError(f"expected {self.future_steps} future frames, got {future.shape[1]}")
        if condition.shape[0] != future.shape[0] or condition.shape[2:] != future.shape[2:]:
            raise ValueError("video condition/future batch and image shapes must match")
        full_video = torch.cat([condition, future], dim=1)
        video_bcthw = full_video.permute(0, 2, 1, 3, 4).contiguous()
        runtime = self._runtime(video_bcthw.device)
        encoded = []
        for start in range(0, video_bcthw.shape[0], self.encode_batch_size):
            chunk = video_bcthw[start : start + self.encode_batch_size]
            encoded.append(runtime.encode(chunk.to(dtype=self._dtype())).float())
        latents = torch.cat(encoded, dim=0)
        expected_steps = self.condition_latent_steps + self.future_latent_steps
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels:
            raise ValueError(f"unexpected Wan latent shape: {tuple(latents.shape)}")
        if latents.shape[2] != expected_steps:
            raise ValueError(
                f"expected {expected_steps} latent steps from video, got {latents.shape[2]}"
            )
        condition_last = latents[:, :, self.condition_latent_steps - 1]
        future_target = latents[:, :, self.condition_latent_steps :]
        return condition_last.detach(), future_target.detach()


class FutureLatentAuxHead(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        latent_channels: int,
        future_steps: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.future_steps = int(future_steps)
        self.condition_proj = nn.Conv2d(self.latent_channels, hidden_dim, kernel_size=3, padding=1)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.step_embedding = nn.Parameter(torch.randn(self.future_steps, hidden_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, self.latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, condition_last: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if condition_last.ndim != 4:
            raise ValueError("condition_last must be [B,C,H,W]")
        if context.ndim != 2 or context.shape[0] != condition_last.shape[0]:
            raise ValueError("video context must be [B,D] with matching batch")
        batch_size, _, height, width = condition_last.shape
        base = self.condition_proj(condition_last)
        bias = self.context_proj(context)[:, None, :, None, None]
        steps = self.step_embedding[None, :, :, None, None]
        hidden = base[:, None] + bias + steps
        decoded = self.decoder(hidden.reshape(batch_size * self.future_steps, -1, height, width))
        return decoded.reshape(
            batch_size,
            self.future_steps,
            self.latent_channels,
            height,
            width,
        ).permute(0, 2, 1, 3, 4)


class VideoAuxFlowMatchingPolicy(FlowMatchingPolicy):
    """Action CFM plus a frozen-Wan future-latent auxiliary objective."""

    def __init__(
        self,
        *,
        video_loss_weight: float = 0.01,
        video_aux_hidden_dim: int = 128,
        video_condition_steps: int = 9,
        video_future_steps: int = 16,
        wan_vae_checkpoint_path: str = "",
        wan_runtime_repo: str = "",
        wan_runtime_site_packages: str = "",
        wan_vae_dtype: str = "bfloat16",
        video_codec_batch_size: int = 1,
        video_codec: VideoLatentCodec | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.video_loss_weight = float(video_loss_weight)
        if self.video_loss_weight < 0.0:
            raise ValueError("video_loss_weight must be non-negative")
        self.video_codec = video_codec or FrozenWanVaeCodec(
            vae_checkpoint_path=wan_vae_checkpoint_path,
            runtime_repo=wan_runtime_repo,
            runtime_site_packages=wan_runtime_site_packages,
            condition_steps=video_condition_steps,
            future_steps=video_future_steps,
            dtype=wan_vae_dtype,
            encode_batch_size=video_codec_batch_size,
        )
        self.video_aux_head = FutureLatentAuxHead(
            context_dim=self.d_model,
            latent_channels=int(self.video_codec.latent_channels),
            future_steps=int(self.video_codec.future_latent_steps),
            hidden_dim=int(video_aux_hidden_dim),
        )

    def _augment_training_loss(
        self,
        *,
        action_loss: torch.Tensor,
        metrics: dict[str, float | None],
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        cond_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        valid_mask = batch.get("video_valid_mask")
        condition = batch.get("video_condition")
        future = batch.get("video_future")
        cached_condition = batch.get("video_condition_latent")
        cached_future = batch.get("video_future_latent")
        clean_action = batch.get("action")
        raw_available = isinstance(condition, torch.Tensor) and isinstance(future, torch.Tensor)
        cache_available = isinstance(cached_condition, torch.Tensor) and isinstance(
            cached_future, torch.Tensor
        )
        if not isinstance(valid_mask, torch.Tensor) or not isinstance(clean_action, torch.Tensor):
            raise KeyError(
                "video auxiliary policy requires video_valid_mask and action tensors"
            )
        if raw_available == cache_available:
            raise KeyError(
                "video auxiliary policy requires exactly one target source: raw "
                "video_condition/video_future or cached video_condition_latent/video_future_latent"
            )
        assert isinstance(valid_mask, torch.Tensor)
        valid = valid_mask.bool().reshape(-1)
        if valid.shape[0] != clean_action.shape[0]:
            raise ValueError("video_valid_mask must have one value per batch sample")
        valid_fraction = float(valid.float().mean().detach().item())
        if not bool(valid.any().item()):
            video_loss = sum(parameter.sum() for parameter in self.video_aux_head.parameters()) * 0.0
            metrics.update(
                {
                    "video_aux_loss": 0.0,
                    "video_aux_weighted_loss": 0.0,
                    "video_aux_valid_fraction": valid_fraction,
                    "video_aux_target_std": None,
                    "video_aux_cached_targets": float(cache_available),
                }
            )
            return action_loss + video_loss, metrics

        if cache_available:
            assert isinstance(cached_condition, torch.Tensor)
            assert isinstance(cached_future, torch.Tensor)
            condition_last = cached_condition[valid].float()
            future_target = cached_future[valid].float()
            if condition_last.ndim != 4 or future_target.ndim != 5:
                raise ValueError("cached video latents must be [B,C,H,W] and [B,C,T,H,W]")
            if condition_last.shape[1] != self.video_aux_head.latent_channels:
                raise ValueError("cached condition latent channels do not match video head")
            if future_target.shape[1] != self.video_aux_head.latent_channels:
                raise ValueError("cached future latent channels do not match video head")
            if future_target.shape[2] != self.video_aux_head.future_steps:
                raise ValueError("cached future latent steps do not match video head")
        else:
            assert isinstance(condition, torch.Tensor)
            assert isinstance(future, torch.Tensor)
            condition_last, future_target = self.video_codec.encode_condition_future(
                condition[valid], future[valid]
            )
        teacher_features = self._features_from_cond_tokens(
            noisy_action=clean_action[valid],
            cond_tokens=cond_tokens[valid],
            timesteps=torch.ones(
                int(valid.sum().item()),
                device=clean_action.device,
                dtype=torch.float32,
            ),
        )
        prediction = self.video_aux_head(condition_last, teacher_features.mean(dim=1))
        if prediction.shape != future_target.shape:
            raise ValueError(
                f"video auxiliary shape mismatch: prediction={tuple(prediction.shape)} "
                f"target={tuple(future_target.shape)}"
            )
        video_loss = (prediction.float() - future_target.float()).square().mean()
        weighted = self.video_loss_weight * video_loss
        metrics.update(
            {
                "video_aux_loss": float(video_loss.detach().item()),
                "video_aux_weighted_loss": float(weighted.detach().item()),
                "video_aux_valid_fraction": valid_fraction,
                "video_aux_target_std": float(future_target.float().std().detach().item()),
                "video_aux_cached_targets": float(cache_available),
            }
        )
        return action_loss + weighted, metrics
