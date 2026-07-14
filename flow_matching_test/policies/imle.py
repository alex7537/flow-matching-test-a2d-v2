from __future__ import annotations

import torch
import torch.nn as nn

from flow_matching_test.observation import build_rgb_obs_composer
from flow_matching_test.policies.base import ActionPolicy, SamplingResult
from flow_matching_test.policies.flow_matching import FlowTransformerBlock


class ImlePolicy(ActionPolicy):
    """RS-IMLE action-chunk generator using the same Transformer body as CFM."""

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
        n_samples_per_condition: int = 20,
        rs_imle_epsilon: float = 0.03,
    ) -> None:
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.history_steps = int(history_steps)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.d_model = int(d_model)
        self.use_proprio = bool(use_proprio)
        self.n_samples_per_condition = int(n_samples_per_condition)
        self.rs_imle_epsilon = float(rs_imle_epsilon)
        if not self.image_keys:
            raise ValueError("image_keys must not be empty")
        if self.n_samples_per_condition <= 0:
            raise ValueError("n_samples_per_condition must be > 0")
        if self.rs_imle_epsilon < 0.0:
            raise ValueError("rs_imle_epsilon must be >= 0")

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
            if self.use_proprio
            else None
        )
        tokens_per_frame = int(getattr(self.obs_composer.encoders[0], "tokens_per_frame", 1))
        max_cond_tokens = (
            len(self.image_keys) * self.history_steps * tokens_per_frame + int(self.use_proprio)
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
        backbone = getattr(self.obs_composer.encoders[0], "backbone", None)
        return list(backbone.parameters()) if backbone is not None else []

    def set_action_stats(self, *, action_mean: torch.Tensor, action_std: torch.Tensor) -> None:
        if action_mean.shape[-1] != self.action_dim or action_std.shape[-1] != self.action_dim:
            raise ValueError("Action stats dim mismatch")
        self.action_mean.copy_(action_mean.detach().float())
        self.action_std.copy_(action_std.detach().float())

    def denormalize_action(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.action_std.view(1, 1, -1) + self.action_mean.view(1, 1, -1)

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

    def _generate(self, latent: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        x = self.action_proj(latent) + self.action_pos[:, : latent.shape[1]]
        for block in self.blocks:
            x = block(x, cond_tokens)
        return self.head(self.final_norm(x))

    def _nearest_distances(
        self,
        real_actions: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_samples = candidates.shape[:2]
        real_flat = real_actions.reshape(batch_size, 1, -1)
        candidate_flat = candidates.reshape(batch_size, num_samples, -1)
        distances = torch.cdist(real_flat, candidate_flat).squeeze(1)
        valid = distances > self.rs_imle_epsilon
        max_distance = distances.max().detach()
        masked = distances + (~valid).to(distances.dtype) * max_distance
        nearest = masked.min(dim=1).values
        valid_real = nearest < max_distance
        return nearest, valid_real

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        clean_action = batch["action"]
        obs = batch["obs"]
        if not isinstance(clean_action, torch.Tensor) or not isinstance(obs, dict):
            raise TypeError("batch must contain tensor action and dict obs")
        batch_size = clean_action.shape[0]
        cond_tokens = self._encode_obs(obs)
        latent = torch.randn(
            batch_size * self.n_samples_per_condition,
            self.action_horizon,
            self.action_dim,
            device=clean_action.device,
            dtype=clean_action.dtype,
        )
        repeated_cond = cond_tokens.repeat_interleave(self.n_samples_per_condition, dim=0)
        candidates = self._generate(latent, repeated_cond).reshape(
            batch_size,
            self.n_samples_per_condition,
            self.action_horizon,
            self.action_dim,
        )
        nearest, valid_real = self._nearest_distances(clean_action, candidates)
        denominator = valid_real.sum()
        loss = (nearest * valid_real).sum() / denominator.clamp_min(1)
        loss = torch.where(denominator > 0, loss, loss.new_zeros(()))

        segment_losses = {"static_loss": None, "continuous_loss": None, "keyframe_loss": None}
        segment_type = batch.get("segment_type")
        if isinstance(segment_type, torch.Tensor):
            for segment_id, name in enumerate(segment_losses):
                mask = (segment_type == segment_id) & valid_real
                if mask.any():
                    segment_losses[name] = float(nearest[mask].mean().detach().item())
        return loss, {
            "imle_loss": float(loss.detach().item()),
            "valid_match_fraction": float(valid_real.float().mean().detach().item()),
            **segment_losses,
        }

    @torch.no_grad()
    def sample_actions(self, obs: dict[str, torch.Tensor]) -> SamplingResult:
        batch_size = int(obs[self.image_keys[0]].shape[0])
        latent = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=obs[self.image_keys[0]].device,
            dtype=obs[self.image_keys[0]].dtype,
        )
        normalized = self._generate(latent, self._encode_obs(obs))
        return SamplingResult(
            action_normalized=normalized,
            action=self.denormalize_action(normalized),
        )
