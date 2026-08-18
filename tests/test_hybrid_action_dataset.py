from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

from flow_matching_test.action_contract import (
    HYBRID_ACTION_SEMANTICS,
    action_layout,
)
from scripts.create_hybrid_hand_target_dataset import materialize_episode


def test_materialize_episode_preserves_timeline_and_replaces_only_hand_action(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.hdf5"
    source_path = tmp_path / "source.hdf5"
    destination_path = tmp_path / "destination.hdf5"
    arm = np.arange(28, dtype=np.float32).reshape(4, 7)
    hand = np.arange(24, dtype=np.float32).reshape(4, 6)
    arm[1] = arm[0]
    hand[1] = hand[0]
    hand_target = hand + 100.0
    executed = np.concatenate([arm, hand], axis=1)
    kept = np.asarray([1, 2, 3])

    with h5py.File(raw_path, "w") as raw:
        trajectory = raw.create_group("trajectory")
        trajectory.create_dataset("arm2_pos", data=arm)
        trajectory.create_dataset("hand2_pos", data=hand)
        trajectory.create_dataset("hand2_pos_target", data=hand_target)

    with h5py.File(source_path, "w") as source:
        observations = source.create_group("observations")
        observations.create_dataset("qpos", data=executed[kept])
        observations.create_dataset(
            "rgb_head", data=np.zeros((len(kept), 2, 2, 3), dtype=np.uint8)
        )
        source.create_dataset("action", data=executed[kept])
        source.attrs["action_layout"] = "arm2_pos(7)+hand2_pos(6)"
        source.attrs["action_semantics"] = "executed_joint_position"

    manifest_sha = "a" * 64
    kept_count, raw_count, target_sha = materialize_episode(
        source_path,
        raw_path,
        destination_path,
        source_manifest_sha256=manifest_sha,
    )

    with h5py.File(destination_path, "r") as destination:
        qpos = np.asarray(destination["observations/qpos"][:])
        action = np.asarray(destination["action"][:])
        np.testing.assert_array_equal(qpos, executed[kept])
        np.testing.assert_array_equal(action[:, :7], arm[kept])
        np.testing.assert_array_equal(action[:, 7:], hand_target[kept])
        assert destination.attrs["action_semantics"] == HYBRID_ACTION_SEMANTICS
        assert destination.attrs["action_layout"] == action_layout(HYBRID_ACTION_SEMANTICS)
        assert destination.attrs["derived_from_dataset_manifest_sha256"] == manifest_sha

    assert kept_count == 3
    assert raw_count == 4
    assert target_sha == hashlib.sha256(hand_target.tobytes()).hexdigest()


def test_materialize_episode_rejects_all_zero_hand_target_row(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.hdf5"
    source_path = tmp_path / "source.hdf5"
    destination_path = tmp_path / "destination.hdf5"
    executed = np.ones((2, 13), dtype=np.float32)
    executed[1, 0] = 2.0
    with h5py.File(raw_path, "w") as raw:
        trajectory = raw.create_group("trajectory")
        trajectory.create_dataset("arm2_pos", data=executed[:, :7])
        trajectory.create_dataset("hand2_pos", data=executed[:, 7:])
        trajectory.create_dataset("hand2_pos_target", data=np.zeros((2, 6), dtype=np.float32))
    with h5py.File(source_path, "w") as source:
        observations = source.create_group("observations")
        observations.create_dataset("qpos", data=executed)
        source.create_dataset("action", data=executed)

    try:
        materialize_episode(
            source_path,
            raw_path,
            destination_path,
            source_manifest_sha256="a" * 64,
        )
    except ValueError as error:
        assert "all-zero hand target" in str(error)
    else:
        raise AssertionError("all-zero hand target row was accepted")
