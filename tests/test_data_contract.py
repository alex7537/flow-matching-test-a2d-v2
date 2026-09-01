from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from flow_matching_test.a2d_dataset import (
    A2DConfig,
    A2DFlowDataset,
    STATS_SCHEMA_VERSION,
    normalize,
    split_episodes_from_manifest,
    train_episode_binding,
    validate_stats_binding,
)
from flow_matching_test.action_contract import HYBRID_ACTION_SEMANTICS
from flow_matching_test.segmentation import compute_executed_action_segments
from scripts.ingest_a2d import admission_reason


class DataContractTest(unittest.TestCase):
    def test_video_aux_uses_nine_history_frames_and_masks_incomplete_future(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            length = 30
            values = np.repeat(np.arange(length, dtype=np.float32)[:, None], 13, axis=1)
            images = np.stack(
                [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(length)]
            )
            with h5py.File(path, "w") as file:
                observations = file.create_group("observations")
                observations.create_dataset("qpos", data=values)
                observations.create_dataset("rgb_head", data=images)
                file.create_dataset("action", data=values)

            dataset = A2DFlowDataset(
                cfg=A2DConfig(
                    image_keys=("rgb_head",),
                    image_size=4,
                    history_steps=1,
                    action_horizon=16,
                    action_offset_steps=1,
                    video_aux_enabled=True,
                    video_key="rgb_head",
                    video_condition_steps=9,
                    video_future_steps=16,
                    video_future_offset_steps=1,
                    aug_random_crop_pad=0,
                ),
                episodes=[{"path": str(path), "length": length}],
                norm_stats={
                    "state": {"min": [0.0] * 13, "span": [29.0] * 13},
                    "action": {"min": [0.0] * 13, "span": [29.0] * 13},
                },
                train=False,
            )

            self.assertEqual(dataset.samples[0], (0, 8))
            complete = dataset[0]
            self.assertEqual(tuple(complete["video_condition"].shape), (9, 3, 4, 4))
            self.assertEqual(tuple(complete["video_future"].shape), (16, 3, 4, 4))
            np.testing.assert_allclose(
                complete["video_condition"][:, 0, 0, 0].numpy(),
                np.arange(9, dtype=np.float32) / 255.0,
            )
            np.testing.assert_allclose(
                complete["video_future"][:, 0, 0, 0].numpy(),
                np.arange(9, 25, dtype=np.float32) / 255.0,
            )
            self.assertTrue(bool(complete["video_valid_mask"].item()))

            tail = dataset[len(dataset) - 1]
            self.assertFalse(bool(tail["video_valid_mask"].item()))
            np.testing.assert_allclose(
                tail["video_future"][:, 0, 0, 0].numpy(),
                np.full(16, 29.0 / 255.0, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                tail["action_mask"].numpy(),
                np.array([1] + [0] * 15, dtype=np.float32),
            )
            dataset.close()

    def test_default_action_chunk_starts_at_the_next_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            length = 5
            qpos = np.repeat(np.arange(length, dtype=np.float32)[:, None], 13, axis=1)
            with h5py.File(path, "w") as file:
                observations = file.create_group("observations")
                observations.create_dataset("qpos", data=qpos)
                observations.create_dataset(
                    "rgb_head",
                    data=np.zeros((length, 4, 4, 3), dtype=np.uint8),
                )
                file.create_dataset("action", data=qpos)

            cfg = A2DConfig(
                image_keys=("rgb_head",),
                image_size=4,
                history_steps=1,
                action_horizon=2,
                aug_random_crop_pad=0,
            )
            stats = {
                "state": {"min": [0.0] * 13, "span": [4.0] * 13},
                "action": {"min": [0.0] * 13, "span": [4.0] * 13},
            }
            dataset = A2DFlowDataset(
                cfg=cfg,
                episodes=[{"path": str(path), "length": length}],
                norm_stats=stats,
                train=False,
            )

            sample = dataset[0]
            np.testing.assert_allclose(sample["state"].numpy(), normalize(qpos[0], stats["state"]))
            np.testing.assert_allclose(
                sample["action"].numpy(),
                normalize(qpos[1:3], stats["action"]),
            )
            self.assertEqual(cfg.action_offset_steps, 1)
            dataset.close()

    def test_tail_action_chunk_repeats_last_action_and_masks_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            length = 5
            qpos = np.repeat(np.arange(length, dtype=np.float32)[:, None], 13, axis=1)
            with h5py.File(path, "w") as file:
                observations = file.create_group("observations")
                observations.create_dataset("qpos", data=qpos)
                observations.create_dataset(
                    "rgb_head",
                    data=np.zeros((length, 4, 4, 3), dtype=np.uint8),
                )
                file.create_dataset("action", data=qpos)

            stats = {
                "state": {"min": [0.0] * 13, "span": [4.0] * 13},
                "action": {"min": [0.0] * 13, "span": [4.0] * 13},
            }
            dataset = A2DFlowDataset(
                cfg=A2DConfig(
                    image_keys=("rgb_head",),
                    image_size=4,
                    history_steps=1,
                    action_horizon=4,
                    action_offset_steps=1,
                    aug_random_crop_pad=0,
                ),
                episodes=[{"path": str(path), "length": length}],
                norm_stats=stats,
                train=False,
            )

            sample = dataset[len(dataset) - 1]
            expected = np.repeat(
                normalize(qpos[-1:], stats["action"]),
                4,
                axis=0,
            )
            np.testing.assert_allclose(sample["action"].numpy(), expected)
            np.testing.assert_array_equal(sample["action_mask"].numpy(), [1, 0, 0, 0])
            dataset.close()

    def test_enhanced_proprio_uses_only_current_and_previous_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            length = 5
            qpos = np.repeat(np.arange(length, dtype=np.float32)[:, None], 13, axis=1)
            actions = qpos.copy()
            actions[:, 7:] += 2.0
            with h5py.File(path, "w") as file:
                observations = file.create_group("observations")
                observations.create_dataset("qpos", data=qpos)
                observations.create_dataset(
                    "rgb_head", data=np.zeros((length, 4, 4, 3), dtype=np.uint8)
                )
                file.create_dataset("action", data=actions)

            stats = {
                "state": {"min": [0.0] * 13, "span": [4.0] * 13},
                "action": {"min": [0.0] * 13, "span": [4.0] * 13},
            }
            dataset = A2DFlowDataset(
                cfg=A2DConfig(
                    image_keys=("rgb_head",),
                    image_size=4,
                    history_steps=1,
                    enhanced_proprio=True,
                    action_horizon=2,
                    action_offset_steps=1,
                    aug_random_crop_pad=0,
                ),
                episodes=[{"path": str(path), "length": length}],
                norm_stats=stats,
                train=False,
            )

            self.assertEqual(dataset.samples[0], (0, 1))
            sample = dataset[0]
            np.testing.assert_allclose(sample["joint_delta"].numpy(), [0.5] * 13)
            np.testing.assert_allclose(
                sample["previous_action"].numpy(), normalize(actions[1], stats["action"])
            )
            np.testing.assert_allclose(sample["hand_tracking_error"].numpy(), [1.0] * 6)
            np.testing.assert_allclose(
                sample["action"].numpy(), normalize(actions[2:4], stats["action"])
            )
            dataset.close()

    def test_hdf5_handle_cache_is_lru_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(3):
                path = Path(directory) / f"episode_{index}.hdf5"
                with h5py.File(path, "w"):
                    pass
                paths.append(path)
            dataset = A2DFlowDataset(
                cfg=A2DConfig(max_open_hdf5_files=2),
                episodes=[{"path": str(path), "length": 1} for path in paths],
                norm_stats={"state": {}, "action": {}},
                train=False,
            )

            first = dataset._file(0)
            second = dataset._file(1)
            self.assertIs(dataset._file(0), first)
            third = dataset._file(2)

            self.assertTrue(first.id.valid)
            self.assertFalse(second.id.valid)
            self.assertTrue(third.id.valid)
            self.assertEqual(list(dataset._handles), [0, 2])
            dataset.close()

    def test_augmentation_is_deterministic_per_seed_epoch_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            length = 3
            image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
            images = np.repeat(image[None], length, axis=0)
            values = np.zeros((length, 13), dtype=np.float32)
            with h5py.File(path, "w") as file:
                observations = file.create_group("observations")
                observations.create_dataset("qpos", data=values)
                observations.create_dataset("rgb_head", data=images)
                file.create_dataset("action", data=values)

            dataset = A2DFlowDataset(
                cfg=A2DConfig(
                    seed=7,
                    image_keys=("rgb_head",),
                    image_size=8,
                    action_horizon=1,
                    aug_random_crop_pad=2,
                ),
                episodes=[{"path": str(path), "length": length}],
                norm_stats={
                    "state": {"min": [0.0] * 13, "span": [1.0] * 13},
                    "action": {"min": [0.0] * 13, "span": [1.0] * 13},
                },
                train=True,
            )

            dataset.set_epoch(0)
            first = dataset[0]["images"].clone()
            repeated = dataset[0]["images"].clone()
            dataset.set_epoch(1)
            next_epoch = dataset[0]["images"].clone()

            np.testing.assert_array_equal(first.numpy(), repeated.numpy())
            self.assertFalse(np.array_equal(first.numpy(), next_epoch.numpy()))
            dataset.close()

    def test_augmentation_is_independent_of_dataloader_worker_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            length = 4
            images = np.arange(length * 8 * 8 * 3, dtype=np.uint8).reshape(
                length, 8, 8, 3
            )
            values = np.zeros((length, 13), dtype=np.float32)
            with h5py.File(path, "w") as file:
                observations = file.create_group("observations")
                observations.create_dataset("qpos", data=values)
                observations.create_dataset("rgb_head", data=images)
                file.create_dataset("action", data=values)

            cfg = A2DConfig(
                seed=11,
                image_keys=("rgb_head",),
                image_size=8,
                action_horizon=1,
                aug_random_crop_pad=2,
            )
            stats = {
                "state": {"min": [0.0] * 13, "span": [1.0] * 13},
                "action": {"min": [0.0] * 13, "span": [1.0] * 13},
            }

            def load(num_workers: int) -> torch.Tensor:
                dataset = A2DFlowDataset(
                    cfg=cfg,
                    episodes=[{"path": str(path), "length": length}],
                    norm_stats=stats,
                    train=True,
                )
                dataset.set_epoch(3)
                batch = next(
                    iter(
                        DataLoader(
                            dataset,
                            batch_size=len(dataset),
                            shuffle=False,
                            num_workers=num_workers,
                        )
                    )
                )
                dataset.close()
                return batch["images"]

            torch.testing.assert_close(load(0), load(2), rtol=0.0, atol=0.0)

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
        hybrid_stats = copy.deepcopy(stats)
        hybrid_stats["action_semantics"] = HYBRID_ACTION_SEMANTICS
        validate_stats_binding(hybrid_stats, episodes, HYBRID_ACTION_SEMANTICS)
        with self.assertRaisesRegex(ValueError, "semantics"):
            validate_stats_binding(hybrid_stats, episodes)
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
