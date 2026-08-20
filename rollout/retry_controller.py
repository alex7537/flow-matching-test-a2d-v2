from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GraspRetryConfig:
    enabled: bool = False
    max_attempts: int = 1
    approach_timeout_steps: int = 96
    close_timeout_steps: int = 64
    lift_timeout_steps: int = 96
    contact_loss_hold_steps: int = 5
    max_total_policy_steps: int = 512
    recovery_open_steps: int = 10
    recovery_retreat_steps: int = 30
    recovery_settle_steps: int = 5
    seed_stride: int = 100_003
    recovery_joint_positions: tuple[float, ...] | None = None

    @classmethod
    def from_execution(cls, execution_cfg: dict[str, Any]) -> "GraspRetryConfig":
        raw = execution_cfg.get("task_grasp_retry", {})
        enabled = bool(raw.get("enabled", False))
        target = raw.get("recovery_joint_positions")
        recovery_joint_positions = None
        if target is not None:
            values = np.asarray(target, dtype=np.float32)
            if values.shape != (13,) or not np.isfinite(values).all():
                raise ValueError(
                    "task_grasp_retry.recovery_joint_positions must contain 13 finite values"
                )
            recovery_joint_positions = tuple(float(value) for value in values)
        config = cls(
            enabled=enabled,
            max_attempts=int(raw.get("max_attempts", 2 if enabled else 1)),
            approach_timeout_steps=int(raw.get("approach_timeout_steps", 96)),
            close_timeout_steps=int(raw.get("close_timeout_steps", 64)),
            lift_timeout_steps=int(raw.get("lift_timeout_steps", 96)),
            contact_loss_hold_steps=int(raw.get("contact_loss_hold_steps", 5)),
            max_total_policy_steps=int(raw.get("max_total_policy_steps", 512)),
            recovery_open_steps=int(raw.get("recovery_open_steps", 10)),
            recovery_retreat_steps=int(raw.get("recovery_retreat_steps", 30)),
            recovery_settle_steps=int(raw.get("recovery_settle_steps", 5)),
            seed_stride=int(raw.get("seed_stride", 100_003)),
            recovery_joint_positions=recovery_joint_positions,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_attempts < (2 if self.enabled else 1):
            raise ValueError("enabled task grasp retry requires max_attempts >= 2")
        if self.approach_timeout_steps < 1:
            raise ValueError("task_grasp_retry.approach_timeout_steps must be positive")
        if self.close_timeout_steps < 1:
            raise ValueError("task_grasp_retry.close_timeout_steps must be positive")
        if self.lift_timeout_steps < 1:
            raise ValueError("task_grasp_retry.lift_timeout_steps must be positive")
        if self.contact_loss_hold_steps < 1:
            raise ValueError("task_grasp_retry.contact_loss_hold_steps must be positive")
        if self.max_total_policy_steps < 1:
            raise ValueError("task_grasp_retry.max_total_policy_steps must be positive")
        if self.recovery_open_steps < 1:
            raise ValueError("task_grasp_retry.recovery_open_steps must be positive")
        if self.recovery_retreat_steps < 1:
            raise ValueError("task_grasp_retry.recovery_retreat_steps must be positive")
        if self.recovery_settle_steps < 0:
            raise ValueError("task_grasp_retry.recovery_settle_steps must be non-negative")
        if self.seed_stride < 1:
            raise ValueError("task_grasp_retry.seed_stride must be positive")

    @property
    def attempt_limit(self) -> int:
        return self.max_attempts if self.enabled else 1


class GraspRetryController:
    """Outer task-level attempt/verify/recover controller around policy replanning."""

    def __init__(self, config: GraspRetryConfig) -> None:
        self.config = config
        self.attempt_index = -1
        self.approached_at_step: int | None = None

    def begin_attempt(self, attempt_index: int) -> None:
        if not 0 <= attempt_index < self.config.attempt_limit:
            raise ValueError("attempt_index is outside configured retry budget")
        self.attempt_index = int(attempt_index)
        self.approached_at_step = None

    def sampling_seed(self, base_seed: int) -> int:
        if self.attempt_index < 0:
            raise RuntimeError("begin_attempt must be called before sampling_seed")
        return int(base_seed) + self.attempt_index * self.config.seed_stride

    def observe(self, checker: Any) -> None:
        if checker.approached and self.approached_at_step is None:
            self.approached_at_step = int(checker.steps)

    def retry_reason(self, checker: Any) -> str | None:
        if not self.config.enabled or checker.done():
            return None
        if checker.closed:
            if int(checker.contact_loss_streak) >= self.config.contact_loss_hold_steps:
                return "contact_lost_after_close"
            if checker.closed_at_step is not None and (
                int(checker.steps) - int(checker.closed_at_step)
                >= self.config.lift_timeout_steps
            ):
                return "lift_timeout"
            return None
        if self.approached_at_step is None:
            if int(checker.steps) >= self.config.approach_timeout_steps:
                return "approach_timeout"
            return None
        if int(checker.steps) - self.approached_at_step >= self.config.close_timeout_steps:
            return "close_timeout"
        return None

    def can_retry(self) -> bool:
        return self.config.enabled and self.attempt_index + 1 < self.config.attempt_limit

    @property
    def recovery_steps(self) -> int:
        return (
            self.config.recovery_open_steps
            + self.config.recovery_retreat_steps
            + self.config.recovery_settle_steps
        )
