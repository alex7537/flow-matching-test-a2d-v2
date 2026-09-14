"""A2D V3 HDF5 adapter for the official RDT HDF5 fine-tuning interface.

Copy this file over ``data/hdf5_vla_dataset.py`` in an isolated checkout of
thu-ml/RoboticsDiffusionTransformer. Runtime settings are environment variables
because the upstream constructor is called without arguments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np


DATASET_NAME = "a2d_v3_multitask"
STATE_DIM = 128
ACTION_CHUNK_SIZE = 64
IMAGE_HISTORY_SIZE = 2
ACTION_OFFSET_STEPS = 1

# RDT has 10 right-arm position slots and 5 right-gripper position slots.
# A2D uses 7 arm joints and 6 dexterous-hand joints. The sixth hand joint is
# intentionally placed in a named project-reserved slot rather than silently
# dropped or aliased onto another pretrained semantic dimension.
A2D_ARM_INDICES = tuple(range(0, 7))
A2D_HAND_INDICES = tuple(range(10, 15)) + (45,)
A2D_STATE_INDICES = A2D_ARM_INDICES + A2D_HAND_INDICES
A2D_ACTION_INDICES = A2D_STATE_INDICES

EXPECTED_ACTION_SEMANTICS = "arm_executed_hand_commanded_joint_position"
EXPECTED_QPOS_LAYOUT = "arm2_pos(7)+hand2_pos(6)"
EXPECTED_ACTION_LAYOUT = "arm2_pos(7)+hand2_pos_target(6)"


def encode_a2d_vector(values: np.ndarray, *, indices: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 13:
        raise ValueError(f"A2D vector must end in 13 dimensions, got {values.shape}")
    output = np.zeros((*values.shape[:-1], STATE_DIM), dtype=np.float32)
    output[..., list(indices)] = values
    return output


def decode_a2d_vector(values: np.ndarray, *, indices: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != STATE_DIM:
        raise ValueError(f"RDT vector must end in {STATE_DIM} dimensions, got {values.shape}")
    return values[..., list(indices)].copy()


def state_indicator() -> np.ndarray:
    indicator = np.zeros(STATE_DIM, dtype=np.float32)
    indicator[list(A2D_STATE_INDICES)] = 1.0
    return indicator


def _decode_rgb(encoded: np.ndarray) -> np.ndarray:
    raw = encoded.tobytes() if isinstance(encoded, np.ndarray) else bytes(encoded)
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode A2D JPEG frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class HDF5VLADataset:
    """RDT-compatible sampler bound to the versioned A2D train/val split."""

    def __init__(self, split: str | None = None) -> None:
        data_dir = os.environ.get("A2D_RDT_DATA_DIR")
        if not data_dir:
            raise RuntimeError("A2D_RDT_DATA_DIR must point to the processed V3 dataset")
        self.data_dir = Path(data_dir).expanduser().resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"A2D dataset directory not found: {self.data_dir}")
        self.split = str(split or os.environ.get("A2D_RDT_SPLIT", "train"))
        if self.split not in {"train", "val"}:
            raise ValueError("A2D_RDT_SPLIT must be 'train' or 'val'")
        self.seed = int(os.environ.get("A2D_RDT_SEED", "42"))
        self.rng = np.random.default_rng(self.seed)
        self.generic_instruction = os.environ.get(
            "A2D_RDT_INSTRUCTION", "Grasp and lift the target object."
        )
        self.task_specific_instruction = os.environ.get(
            "A2D_RDT_TASK_INSTRUCTIONS", "0"
        ) == "1"
        lang_embed_dir = os.environ.get("A2D_RDT_LANG_EMBED_DIR")
        self.lang_embed_dir = (
            Path(lang_embed_dir).expanduser().resolve() if lang_embed_dir else None
        )

        dataset_manifest_path = self.data_dir / "dataset_manifest.json"
        split_manifest_path = self.data_dir / "split_manifest.json"
        self.dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        self.split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if self.dataset_manifest.get("action_semantics") != EXPECTED_ACTION_SEMANTICS:
            raise ValueError("dataset manifest does not use the A2D V3 hybrid action semantics")
        split_key = "train_episodes" if self.split == "train" else "val_episodes"
        selected = list(self.split_manifest.get(split_key, []))
        if not selected:
            raise ValueError(f"split manifest has no {split_key}")
        records = {
            str(item["file_name"]): item
            for item in self.dataset_manifest.get("episodes", [])
        }
        self.episodes = []
        for file_name in selected:
            if file_name not in records:
                raise ValueError(f"split episode missing from dataset manifest: {file_name}")
            record = dict(records[file_name])
            record["path"] = self.data_dir / file_name
            self.episodes.append(record)
        lengths = np.asarray([int(item["length"]) for item in self.episodes], dtype=np.float64)
        self.episode_sample_weights = lengths / lengths.sum()

    def __len__(self) -> int:
        return len(self.episodes)

    def get_dataset_name(self) -> str:
        return DATASET_NAME

    def _validate_file(self, file: h5py.File, record: dict) -> int:
        attrs = file.attrs
        if str(attrs.get("action_semantics", "")) != EXPECTED_ACTION_SEMANTICS:
            raise ValueError(f"{record['file_name']}: unexpected action semantics")
        if str(attrs.get("qpos_layout", "")) != EXPECTED_QPOS_LAYOUT:
            raise ValueError(f"{record['file_name']}: unexpected qpos layout")
        if str(attrs.get("action_layout", "")) != EXPECTED_ACTION_LAYOUT:
            raise ValueError(f"{record['file_name']}: unexpected action layout")
        qpos = file["observations/qpos"]
        action = file["action"]
        if qpos.ndim != 2 or action.ndim != 2 or qpos.shape[1:] != (13,) or action.shape[1:] != (13,):
            raise ValueError(f"{record['file_name']}: qpos/action must have shape [T,13]")
        if qpos.shape[0] != action.shape[0] or qpos.shape[0] < 2:
            raise ValueError(f"{record['file_name']}: invalid qpos/action length")
        return int(qpos.shape[0])

    def _instruction(self, task_id: str) -> str:
        if not self.task_specific_instruction:
            return self.generic_instruction
        return f"Grasp and lift the {task_id}."

    def _instruction_value(self, task_id: str) -> str:
        if self.lang_embed_dir is None:
            return self._instruction(task_id)
        path = self.lang_embed_dir / f"{task_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing precomputed language embedding: {path}")
        return str(path)

    def get_item(self, index: int | None = None, state_only: bool = False) -> dict:
        explicit_index = index is not None
        if index is None:
            index = int(self.rng.choice(len(self.episodes), p=self.episode_sample_weights))
        record = self.episodes[int(index) % len(self.episodes)]
        with h5py.File(record["path"], "r", libver="latest", swmr=True) as file:
            length = self._validate_file(file, record)
            qpos13 = np.asarray(file["observations/qpos"], dtype=np.float32)
            action13 = np.asarray(file["action"], dtype=np.float32)
            if not np.isfinite(qpos13).all() or not np.isfinite(action13).all():
                raise ValueError(f"{record['file_name']}: non-finite state/action")
            if state_only:
                return {
                    "state": encode_a2d_vector(qpos13, indices=A2D_STATE_INDICES),
                    "action": encode_a2d_vector(action13, indices=A2D_ACTION_INDICES),
                }

            step_rng = (
                np.random.default_rng(self.seed + int(index) * 1_000_003)
                if self.split == "val" and explicit_index
                else self.rng
            )
            step_id = int(step_rng.integers(0, length - ACTION_OFFSET_STEPS))
            state = encode_a2d_vector(
                qpos13[step_id : step_id + 1], indices=A2D_STATE_INDICES
            )
            action_start = step_id + ACTION_OFFSET_STEPS
            action_stop = min(action_start + ACTION_CHUNK_SIZE, length)
            action_chunk13 = action13[action_start:action_stop]
            valid_action_steps = int(action_chunk13.shape[0])
            if valid_action_steps < ACTION_CHUNK_SIZE:
                action_chunk13 = np.concatenate(
                    [
                        action_chunk13,
                        np.repeat(
                            action_chunk13[-1:], ACTION_CHUNK_SIZE - valid_action_steps, axis=0
                        ),
                    ],
                    axis=0,
                )
            actions = encode_a2d_vector(action_chunk13, indices=A2D_ACTION_INDICES)
            action_time_mask = np.zeros(ACTION_CHUNK_SIZE, dtype=np.float32)
            action_time_mask[:valid_action_steps] = 1.0

            history_indices = [max(step_id - 1, 0), step_id]
            history_mask = np.asarray([step_id > 0, True], dtype=bool)
            cam_high = np.stack(
                [_decode_rgb(file["observations/rgb_head"][i]) for i in history_indices]
            )
            cam_right = np.stack(
                [_decode_rgb(file["observations/rgb_right_hand"][i]) for i in history_indices]
            )
            cam_left = np.zeros((IMAGE_HISTORY_SIZE, 0, 0, 3), dtype=np.uint8)

        state_all = encode_a2d_vector(qpos13, indices=A2D_STATE_INDICES)
        state_std = state_all.std(axis=0).astype(np.float32)
        state_mean = state_all.mean(axis=0).astype(np.float32)
        state_norm = np.sqrt(np.mean(state_all**2, axis=0)).astype(np.float32)
        task_id = str(record.get("task_id", record["file_name"].split("__", 1)[0]))
        return {
            "meta": {
                "dataset_name": DATASET_NAME,
                "#steps": length,
                "step_id": step_id,
                "instruction": self._instruction_value(task_id),
                "instruction_text": self._instruction(task_id),
                "task_id": task_id,
                "source_file_name": record["file_name"],
                "action_semantics": EXPECTED_ACTION_SEMANTICS,
                "action_offset_steps": ACTION_OFFSET_STEPS,
                "valid_action_steps": valid_action_steps,
            },
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "action_time_mask": action_time_mask,
            "state_indicator": state_indicator(),
            "cam_high": cam_high,
            "cam_high_mask": history_mask,
            "cam_right_wrist": cam_right,
            "cam_right_wrist_mask": history_mask.copy(),
            "cam_left_wrist": cam_left,
            "cam_left_wrist_mask": np.zeros(IMAGE_HISTORY_SIZE, dtype=bool),
        }


if __name__ == "__main__":
    dataset = HDF5VLADataset()
    sample = dataset.get_item(0)
    print({
        "episodes": len(dataset),
        "split": dataset.split,
        "file": sample["meta"]["source_file_name"],
        "step_id": sample["meta"]["step_id"],
        "state": sample["state"].shape,
        "actions": sample["actions"].shape,
        "cam_high": sample["cam_high"].shape,
        "cam_right_wrist": sample["cam_right_wrist"].shape,
        "cam_left_wrist": sample["cam_left_wrist"].shape,
    })
