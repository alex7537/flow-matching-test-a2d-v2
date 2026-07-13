from __future__ import annotations

import numpy as np


SEGMENTATION_VERSION = 1
SEGMENT_STATIC = 0
SEGMENT_CONTINUOUS = 1
SEGMENT_KEYFRAME = 2
DEFAULT_MOTION_THRESHOLD = 1.0e-4
DEFAULT_KEYFRAME_THRESHOLD = 0.1


def compute_executed_action_segments(
    action: np.ndarray,
    *,
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    keyframe_threshold: float = DEFAULT_KEYFRAME_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame segment type and arm-keyframe mask from executed action."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 13:
        raise ValueError(f"action must have shape (T,13), got {action.shape}")

    segment_type = np.full(action.shape[0], SEGMENT_STATIC, dtype=np.uint8)
    arm_keyframe = np.zeros(action.shape[0], dtype=np.uint8)
    if action.shape[0] < 2:
        return segment_type, arm_keyframe

    delta = np.diff(action, axis=0)
    arm_speed = np.linalg.norm(delta[:, :7], axis=1)
    hand_speed = np.linalg.norm(delta[:, 7:], axis=1)
    action_speed = np.linalg.norm(delta, axis=1)
    arm_key = arm_speed > keyframe_threshold
    any_key = arm_key | (hand_speed > keyframe_threshold)
    moving = action_speed > motion_threshold

    segment_type[1:][moving] = SEGMENT_CONTINUOUS
    segment_type[1:][any_key] = SEGMENT_KEYFRAME
    arm_keyframe[1:][arm_key] = 1
    return segment_type, arm_keyframe
