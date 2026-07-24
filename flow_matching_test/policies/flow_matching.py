from __future__ import annotations

import math

import torch
import torch.nn as nn

from flow_matching_test.observation import build_rgb_obs_composer
from flow_matching_test.policies.base import ActionPolicy, SamplingResult


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        angles = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, emb.new_zeros((emb.shape[0], 1))], dim=-1)
        return emb


class FlowTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, action_tokens: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        x = action_tokens
        self_out, _ = self.self_attn(self.self_norm(x), self.self_norm(x), self.self_norm(x), need_weights=False)
        x = x + self_out
        cross_in = self.cross_norm(x)
        cross_out, _ = self.cross_attn(cross_in, cond_tokens, cond_tokens, need_weights=False)
        x = x + cross_out
        x = x + self.mlp(self.mlp_norm(x))
        return x


class FlowMatchingPolicy(ActionPolicy):
    def __init__(
        self,
        *,
        image_keys: tuple[str, ...],
        encoder_type: str = "cnn",
        timm_model_name: str = "vit_small_r26_s32_224",
        timm_pretrained: bool = True,
        timm_tokens_per_frame: int = 1,
        timm_token_mode: str = "spatial",
        use_proprio: bool = True,
        history_steps: int,
        action_dim: int,
        action_horizon: int,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.0,
        time_eps: float = 1.0e-3,
        num_inference_steps: int = 40,
    ) -> None:
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.history_steps = int(history_steps)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.d_model = int(d_model)
        self.time_eps = float(time_eps)
        self.num_inference_steps = int(num_inference_steps)
        self.use_proprio = bool(use_proprio)

        if not self.image_keys:
            raise ValueError("image_keys must not be empty")
        if not (0.0 <= self.time_eps < 1.0):
            raise ValueError("time_eps must be in [0,1)")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")

        self.encoder_type = str(encoder_type)
        self.obs_composer = build_rgb_obs_composer(
            image_keys=self.image_keys,
            encoder_type=self.encoder_type,
            d_model=self.d_model,
            timm_model_name=timm_model_name,
            timm_pretrained=bool(timm_pretrained),
            timm_tokens_per_frame=int(timm_tokens_per_frame),
            timm_token_mode=str(timm_token_mode),
        )
        self.action_proj = nn.Linear(self.action_dim, self.d_model)
        self.proprio_proj = (
            nn.Sequential(
                nn.Linear(self.history_steps * self.action_dim, self.d_model),
                nn.LayerNorm(self.d_model),
            )
            if self.use_proprio else None
        )
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        tokens_per_frame = getattr(self.obs_composer.encoders[0], "tokens_per_frame", 1)
        max_cond_tokens = (
            len(self.image_keys) * self.history_steps * int(tokens_per_frame)
            + int(self.use_proprio)
        )
        self.cond_pos = nn.Parameter(torch.randn(1, max_cond_tokens, self.d_model) * 0.02)
        self.action_pos = nn.Parameter(torch.randn(1, self.action_horizon, self.d_model) * 0.02)
        self.blocks = nn.ModuleList(
            FlowTransformerBlock(self.d_model, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.action_dim)

        self.register_buffer("action_mean", torch.zeros(self.action_dim), persistent=False)
        self.register_buffer("action_std", torch.ones(self.action_dim), persistent=False)

    def backbone_parameters(self) -> list[nn.Parameter]:
        encoder = self.obs_composer.encoders[0]
        backbone = getattr(encoder, "backbone", None)
        return list(backbone.parameters()) if backbone is not None else []

    def set_action_stats(self, *, action_mean: torch.Tensor, action_std: torch.Tensor) -> None:
        if action_mean.shape[-1] != self.action_dim or action_std.shape[-1] != self.action_dim:
            raise ValueError("Action stats dim mismatch")
        self.action_mean.copy_(action_mean.detach().float())
        self.action_std.copy_(action_std.detach().float())

    def denormalize_action(self, action_normalized: torch.Tensor) -> torch.Tensor:
        return action_normalized * self.action_std.view(1, 1, -1) + self.action_mean.view(1, 1, -1)

    def _encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        cond_tokens = self.obs_composer(obs)
        if self.use_proprio:
            proprio = obs.get("proprio")
            if proprio is None:
                raise KeyError("Missing proprio observation")
            if proprio.ndim != 2 or proprio.shape[1] != self.history_steps * self.action_dim:
                raise ValueError(
                    f"proprio must have shape [B,{self.history_steps * self.action_dim}], "
                    f"got {tuple(proprio.shape)}"
                )
            assert self.proprio_proj is not None
            cond_tokens = torch.cat([cond_tokens, self.proprio_proj(proprio).unsqueeze(1)], dim=1)
        return cond_tokens + self.cond_pos[:, : cond_tokens.shape[1]]

    def forward(
        self,
        *,
        noisy_action: torch.Tensor,
        obs: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        cond_tokens = self._encode_obs(obs)
        action_tokens = self.action_proj(noisy_action) + self.action_pos[:, : noisy_action.shape[1]]
        time_tokens = self.time_embed(timesteps.float()).unsqueeze(1)
        x = action_tokens + time_tokens
        for block in self.blocks:
            x = block(x, cond_tokens)
        x = self.final_norm(x)
        return self.head(x)

    def _compute_loss_with_randomness(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        *,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        clean_action = batch["action"]
        if not isinstance(clean_action, torch.Tensor):
            raise TypeError("batch['action'] must be a tensor")
        obs = batch["obs"]
        if not isinstance(obs, dict):
            raise TypeError("batch['obs'] must be a dict of RGB tensors")
        t_expand = timesteps[:, None, None].to(clean_action.dtype)
        noisy_action = (1.0 - t_expand) * noise + t_expand * clean_action
        target_velocity = clean_action - noise
        pred_velocity = self(noisy_action=noisy_action, obs=obs, timesteps=timesteps)
        per_sample_loss = torch.mean((pred_velocity - target_velocity) ** 2, dim=(1, 2))
        loss = per_sample_loss.mean()
        segment_losses = {"static_loss": None, "continuous_loss": None, "keyframe_loss": None}
        segment_type = batch.get("segment_type")
        if isinstance(segment_type, torch.Tensor):
            for segment_id, name in enumerate(("static_loss", "continuous_loss", "keyframe_loss")):
                mask = segment_type == segment_id
                if mask.any():
                    segment_losses[name] = float(per_sample_loss[mask].mean().detach().item())
        return loss, {
            "flow_loss": float(loss.detach().item()),
            "t_mean": float(timesteps.mean().detach().item()),
            **segment_losses,
        }

    def compute_loss(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> tuple[torch.Tensor, dict[str, float | None]]:
        clean_action = batch["action"]
        if not isinstance(clean_action, torch.Tensor):
            raise TypeError("batch['action'] must be a tensor")
        batch_size = int(clean_action.shape[0])
        noise = torch.randn_like(clean_action)
        timesteps = torch.rand(batch_size, device=clean_action.device, dtype=torch.float32)
        timesteps = timesteps * (1.0 - self.time_eps) + self.time_eps
        return self._compute_loss_with_randomness(batch, noise=noise, timesteps=timesteps)

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
        noise_rows = []
        timestep_rows = []
        for seed in seeds:
            generator = torch.Generator(device=clean_action.device)
            generator.manual_seed(int(seed))
            noise_rows.append(
                torch.randn(
                    (1, *clean_action.shape[1:]),
                    device=clean_action.device,
                    dtype=clean_action.dtype,
                    generator=generator,
                )
            )
            timestep_rows.append(
                torch.rand((), device=clean_action.device, dtype=torch.float32, generator=generator)
            )
        noise = torch.cat(noise_rows, dim=0)
        timesteps = torch.stack(timestep_rows)
        timesteps = timesteps * (1.0 - self.time_eps) + self.time_eps
        return self._compute_loss_with_randomness(batch, noise=noise, timesteps=timesteps)

    def _sample_actions_from_noise(
        self,
        *,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> SamplingResult:
        batch_size = int(action.shape[0])
        anchor = next(iter(obs.values()))
        dt = (1.0 - self.time_eps) / float(self.num_inference_steps)
        for step_idx in range(self.num_inference_steps):
            t_value = self.time_eps + dt * float(step_idx)
            timestep = torch.full((batch_size,), t_value, device=anchor.device, dtype=torch.float32)
            pred_velocity = self(noisy_action=action, obs=obs, timesteps=timestep).to(action.dtype)
            action = action + pred_velocity * dt
        action_normalized = action.float()
        action_unnormalized = self.denormalize_action(action_normalized)
        return SamplingResult(action_normalized=action_normalized, action=action_unnormalized)

    @torch.no_grad()
    def sample_actions(self, obs: dict[str, torch.Tensor]) -> SamplingResult:
        anchor = next(iter(obs.values()))
        if anchor.ndim != 5:
            raise ValueError("RGB observations must have shape [B,T,3,H,W]")
        batch_size = int(anchor.shape[0])
        action = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=anchor.device,
            dtype=anchor.dtype,
        )
        return self._sample_actions_from_noise(obs=obs, action=action)

    @torch.no_grad()
    def sample_actions_seeded(
        self,
        obs: dict[str, torch.Tensor],
        seeds: list[int],
    ) -> SamplingResult:
        anchor = next(iter(obs.values()))
        if anchor.ndim != 5:
            raise ValueError("RGB observations must have shape [B,T,3,H,W]")
        if len(seeds) != int(anchor.shape[0]):
            raise ValueError("validation seed count must match batch size")
        rows = []
        for seed in seeds:
            generator = torch.Generator(device=anchor.device)
            generator.manual_seed(int(seed))
            rows.append(
                torch.randn(
                    (1, self.action_horizon, self.action_dim),
                    device=anchor.device,
                    dtype=anchor.dtype,
                    generator=generator,
                )
            )
        return self._sample_actions_from_noise(obs=obs, action=torch.cat(rows, dim=0))
