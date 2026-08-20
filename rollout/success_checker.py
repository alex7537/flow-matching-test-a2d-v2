from __future__ import annotations

from typing import Any

import numpy as np


class ThreePhaseChecker:
    """Stateful approach -> stable close -> stable lift success checker."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        criteria = cfg.get("success_criteria", cfg)
        self.approach_distance_m = float(criteria.get("approach_distance_m", 0.08))
        self.approach_rotation_deg = float(criteria.get("approach_rotation_deg", 45.0))
        self.close_contact_count = int(criteria.get("close_contact_count", 2))
        self.close_hold_steps = int(criteria.get("close_hold_steps", 3))
        self.lift_height_m = float(criteria.get("lift_height_m", 0.025))
        self.lift_hold_steps = int(criteria.get("lift_hold_steps", 5))
        self.reset()

    def reset(self) -> None:
        self.approached = False
        self.closed = False
        self.lifted = False
        self.close_streak = 0
        self.lift_streak = 0
        self.contact_loss_streak = 0
        self.current_contact_count = 0
        self.closed_at_step: int | None = None
        self.steps = 0
        self.initial_object_height: float | None = None
        self.max_height_gain = 0.0

    def prime(self, state: dict[str, Any]) -> None:
        """Capture reset height without counting it as an executed control step."""
        self.initial_object_height = float(np.asarray(state["object_position"])[2])

    def update(self, state: dict[str, Any]) -> None:
        self.steps += 1
        object_position = np.asarray(state["object_position"], dtype=np.float64)
        eef_position = np.asarray(state["eef_position"], dtype=np.float64)
        if self.initial_object_height is None:
            self.initial_object_height = float(object_position[2])

        distance = float(np.linalg.norm(object_position - eef_position))
        rotation_error = float(state.get("grasp_rotation_error_deg", 0.0))
        if distance <= self.approach_distance_m and rotation_error <= self.approach_rotation_deg:
            self.approached = True

        contacts = int(state.get("contact_count", 0))
        self.current_contact_count = contacts
        if self.approached and contacts >= self.close_contact_count:
            self.close_streak += 1
        else:
            self.close_streak = 0
        if not self.closed and self.close_streak >= self.close_hold_steps:
            self.closed = True
            self.closed_at_step = self.steps
        if self.closed and contacts < self.close_contact_count:
            self.contact_loss_streak += 1
        else:
            self.contact_loss_streak = 0

        height_gain = float(object_position[2]) - self.initial_object_height
        self.max_height_gain = max(self.max_height_gain, height_gain)
        if self.closed and height_gain >= self.lift_height_m and contacts >= self.close_contact_count:
            self.lift_streak += 1
        else:
            self.lift_streak = 0
        self.lifted = self.lifted or self.lift_streak >= self.lift_hold_steps

    def done(self) -> bool:
        return self.lifted

    def summary(self) -> dict[str, Any]:
        if self.lifted:
            failure_stage = None
        elif not self.approached:
            failure_stage = "approach"
        elif not self.closed:
            failure_stage = "close"
        else:
            failure_stage = "lift"
        return {
            "success": self.lifted,
            "approach_success": self.approached,
            "close_success": self.closed,
            "lift_success": self.lifted,
            "failure_stage": failure_stage,
            "steps": self.steps,
            "max_height_gain_m": self.max_height_gain,
            "current_contact_count": self.current_contact_count,
            "contact_loss_streak": self.contact_loss_streak,
        }
