from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flow_matching_test.a2d_dataset import (
    A2DConfig,
    STATS_SCHEMA_VERSION,
    split_episodes_from_manifest,
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

    def test_fixed_split_rejects_stale_dataset_manifest(self) -> None:
        episodes = [
            {"file_name": "a.hdf5", "content_hash": "a" * 64},
            {"file_name": "b.hdf5", "content_hash": "b" * 64},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset_manifest.json"
            split_path = root / "split_manifest.json"
            dataset_path.write_text(
                json.dumps({"episodes": episodes}, indent=2) + "\n",
                encoding="utf-8",
            )
            split_path.write_text(
                json.dumps({
                    "seed": 42,
                    "val_ratio": 0.5,
                    "dataset_manifest_sha256": hashlib.sha256(
                        dataset_path.read_bytes()
                    ).hexdigest(),
                    "train_episodes": ["a.hdf5"],
                    "val_episodes": ["b.hdf5"],
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            cfg = A2DConfig(
                data_dir=str(root),
                val_ratio=0.5,
                dataset_manifest=dataset_path.name,
                split_manifest=split_path.name,
            )
            train, val, manifest = split_episodes_from_manifest(cfg, episodes)
            self.assertEqual([item["file_name"] for item in train], ["a.hdf5"])
            self.assertEqual([item["file_name"] for item in val], ["b.hdf5"])
            self.assertIn("split_manifest_sha256", manifest)

            dataset_path.write_text(json.dumps({"episodes": []}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                split_episodes_from_manifest(cfg, episodes)


if __name__ == "__main__":
    unittest.main()
