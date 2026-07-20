from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SamplingResult:
    action_normalized: torch.Tensor
    action: torch.Tensor


class ActionPolicy(nn.Module, ABC):
    """Training and sampling boundary shared by policy experiments."""

    @abstractmethod
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        """Return a differentiable scalar objective and detached metrics."""

    @abstractmethod
    def sample_actions(self, obs: dict[str, torch.Tensor]) -> SamplingResult:
        """Sample normalized and denormalized action chunks."""

    @abstractmethod
    def set_action_stats(self, *, action_mean: torch.Tensor, action_std: torch.Tensor) -> None:
        """Install action normalization statistics."""

    @abstractmethod
    def denormalize_action(self, normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized actions back to the dataset action space."""

    def backbone_parameters(self) -> list[nn.Parameter]:
        return []

    def optimizer_parameter_groups(
        self,
        *,
        base_lr: float,
        backbone_lr_multiplier: float,
    ) -> tuple[list[dict[str, Any]], list[nn.Parameter], list[nn.Parameter]]:
        backbone = list(self.backbone_parameters())
        backbone_ids = {id(parameter) for parameter in backbone}
        head = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in backbone_ids and parameter.requires_grad
        ]
        trainable_backbone = [parameter for parameter in backbone if parameter.requires_grad]
        groups = [
            {"params": head, "lr": float(base_lr)},
            {
                "params": trainable_backbone,
                "lr": float(base_lr) * float(backbone_lr_multiplier),
            },
        ]
        return groups, head, backbone
