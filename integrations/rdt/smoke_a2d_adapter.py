#!/usr/bin/env python3
"""Validate A2D/RDT mapping, timing, split and image-history invariants."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os

import h5py
import numpy as np

from a2d_hdf5_vla_dataset import (
    A2D_ACTION_INDICES,
    A2D_STATE_INDICES,
    ACTION_CHUNK_SIZE,
    EXPECTED_ACTION_SEMANTICS,
    HDF5VLADataset,
    decode_a2d_vector,
    encode_a2d_vector,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    os.environ["A2D_RDT_DATA_DIR"] = args.data_dir
    os.environ["A2D_RDT_SPLIT"] = args.split
    dataset = HDF5VLADataset()
    task_counts = Counter(str(item.get("task_id")) for item in dataset.episodes)
    expected_per_task = 270 if args.split == "train" else 30
    assert task_counts == {"box": expected_per_task, "bottle": expected_per_task}
    split_manifest = dataset.split_manifest
    assert not set(split_manifest["train_episodes"]) & set(split_manifest["val_episodes"])

    probe = np.arange(26, dtype=np.float32).reshape(2, 13) / 10
    state_roundtrip = decode_a2d_vector(
        encode_a2d_vector(probe, indices=A2D_STATE_INDICES),
        indices=A2D_STATE_INDICES,
    )
    action_roundtrip = decode_a2d_vector(
        encode_a2d_vector(probe, indices=A2D_ACTION_INDICES),
        indices=A2D_ACTION_INDICES,
    )
    np.testing.assert_array_equal(state_roundtrip, probe)
    np.testing.assert_array_equal(action_roundtrip, probe)

    candidate_indices = []
    for task in ("box", "bottle"):
        task_indices = [
            i for i, item in enumerate(dataset.episodes) if item.get("task_id") == task
        ]
        candidate_indices.extend(task_indices[: max(1, args.samples // 2)])
    checked = []
    for index in candidate_indices[: args.samples]:
        sample = dataset.get_item(index)
        meta = sample["meta"]
        assert meta["action_semantics"] == EXPECTED_ACTION_SEMANTICS
        assert sample["state"].shape == (1, 128)
        assert sample["actions"].shape == (ACTION_CHUNK_SIZE, 128)
        assert sample["action_time_mask"].shape == (ACTION_CHUNK_SIZE,)
        assert int(sample["action_time_mask"].sum()) == meta["valid_action_steps"]
        assert np.all(sample["action_time_mask"][:meta["valid_action_steps"]] == 1)
        assert np.all(sample["action_time_mask"][meta["valid_action_steps"]:] == 0)
        assert sample["state_indicator"].shape == (128,)
        assert int(sample["state_indicator"].sum()) == 13
        assert sample["cam_high"].shape[0] == 2
        assert sample["cam_right_wrist"].shape[0] == 2
        assert sample["cam_left_wrist"].shape == (2, 0, 0, 3)
        assert sample["cam_left_wrist_mask"].tolist() == [False, False]
        file_path = dataset.data_dir / meta["source_file_name"]
        with h5py.File(file_path, "r") as file:
            expected_state = np.asarray(file["observations/qpos"][meta["step_id"]])
            expected_action = np.asarray(
                file["action"][meta["step_id"] + meta["action_offset_steps"]]
            )
        np.testing.assert_array_equal(
            decode_a2d_vector(sample["state"], indices=A2D_STATE_INDICES)[0],
            expected_state,
        )
        np.testing.assert_array_equal(
            decode_a2d_vector(sample["actions"], indices=A2D_ACTION_INDICES)[0],
            expected_action,
        )
        padded = sample["actions"][meta["valid_action_steps"] :]
        if len(padded):
            np.testing.assert_array_equal(
                padded,
                np.repeat(
                    sample["actions"][meta["valid_action_steps"] - 1 : meta["valid_action_steps"]],
                    len(padded),
                    axis=0,
                ),
            )
        checked.append({
            "file": meta["source_file_name"],
            "task": meta["task_id"],
            "step": meta["step_id"],
            "valid_action_steps": meta["valid_action_steps"],
        })

    print(json.dumps({
        "status": "A2D_RDT_ADAPTER_OK",
        "split": args.split,
        "episodes": len(dataset),
        "task_counts": dict(sorted(task_counts.items())),
        "split_overlap": 0,
        "mapped_indices": list(A2D_STATE_INDICES),
        "roundtrip_exact": True,
        "checked": checked,
    }, indent=2))


if __name__ == "__main__":
    main()
