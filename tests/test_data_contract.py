from __future__ import annotations

import copy
import unittest

import numpy as np

from flow_matching_test.a2d_dataset import (
    STATS_SCHEMA_VERSION,
    train_episode_binding,
    validate_stats_binding,
)
from flow_matching_test.segmentation import compute_executed_action_segments
from scripts.ingest_a2d import admission_reason


class DataContractTest(unittest.TestCase):
    def test_ingest_gate_quarantines_truncated_complete_lift_episode(self) -> None:
        report = {
            "errors": [],
            "structural_alignment": True,
            "recording_completeness": {"truncated_recording": True},
        }
        self.assertEqual(admission_reason(report, "complete-lift"), "truncated_lift_recording")
        self.assertIsNone(admission_reason(report, "structural"))

    def test_train_binding_is_path_independent_and_rejects_membership_change(self) -> None:
        episodes = [
            {"path": "/old/a.h5", "content_hash": "a" * 64},
            {"path": "/old/b.h5", "content_hash": "b" * 64},
        ]
        hashes, digest = train_episode_binding(episodes)
        moved = [dict(item, path=item["path"].replace("/old", "/new")) for item in episodes]
        self.assertEqual(train_episode_binding(moved), (hashes, digest))

        stats = {
            "schema_version": STATS_SCHEMA_VERSION,
            "train_episode_hashes": hashes,
            "train_episode_digest": digest,
            "action_semantics": "executed_joint_position",
        }
        validate_stats_binding(stats, episodes)
        with self.assertRaises(ValueError):
            validate_stats_binding(copy.deepcopy(stats), episodes[:1])

    def test_segmentation_uses_executed_action_thresholds(self) -> None:
        action = np.zeros((4, 13), dtype=np.float32)
        action[1, 7] = 0.01
        action[2, 0] = 0.2
        action[2, 7] = 0.01
        action[3] = action[2]
        segment, arm_keyframe = compute_executed_action_segments(action)
        np.testing.assert_array_equal(segment, [0, 1, 2, 0])
        np.testing.assert_array_equal(arm_keyframe, [0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
