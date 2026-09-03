from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from flow_matching_test.a2d_dataset import A2DConfig, A2DFlowDataset
from scripts import precompute_wan_joint_latents as precompute


class FakeJointCodec:
    calls = 0

    def __init__(self, **kwargs) -> None:
        del kwargs

    def encode_joint_condition_future(self, condition, future):
        type(self).calls += 1
        batch = condition.shape[0]
        return (
            torch.full((batch, 2, 3, 2, 2), 3.0),
            torch.full((batch, 2, 4, 2, 2), 5.0),
        )


def test_joint_cache_covers_tail_with_separate_latent_mask(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    source_path = data_dir / "episode.hdf5"
    length = 30
    values = np.zeros((length, 13), dtype=np.float32)
    images = np.stack(
        [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(length)]
    )
    with h5py.File(source_path, "w") as file:
        observations = file.create_group("observations")
        observations.create_dataset("qpos", data=values)
        observations.create_dataset("rgb_head", data=images)
        file.create_dataset("action", data=values)
    dataset_manifest = {
        "episodes": [{
            "file_name": source_path.name,
            "length": length,
            "content_hash": "episode-content-hash",
        }]
    }
    dataset_path = data_dir / "dataset_manifest.json"
    dataset_path.write_text(json.dumps(dataset_manifest), encoding="utf-8")
    vae_path = tmp_path / "vae.pth"
    vae_path.write_bytes(b"fake-vae")
    args = SimpleNamespace(
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        dataset_manifest=dataset_path.name,
        obs_group="observations",
        video_key="rgb_head",
        image_size=4,
        condition_steps=9,
        future_steps=16,
        future_offset_steps=1,
        vae_checkpoint=str(vae_path),
        wan_runtime_repo=str(tmp_path),
        wan_runtime_site_packages="",
        vae_dtype="float32",
        batch_size=4,
        latent_channels=2,
        latent_height=2,
        latent_width=2,
        device="cpu",
    )
    monkeypatch.setattr(precompute, "parse_args", lambda: args)
    monkeypatch.setattr(precompute, "FrozenWanVaeCodec", FakeJointCodec)
    FakeJointCodec.calls = 0

    precompute.main()

    manifest_path = cache_dir / "joint_video_latent_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["dataset_manifest_sha256"] == hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest()
    assert manifest["total_windows"] == 21
    assert manifest["total_valid_future_latent_steps"] == 48
    record = manifest["episodes"][0]
    cache_path = cache_dir / record["cache_file_name"]
    with h5py.File(cache_path, "r") as file:
        assert file["condition_prefix"].shape == (21, 2, 3, 2, 2)
        assert file["future_target"].shape == (21, 2, 4, 2, 2)
        assert file["future_valid_mask"][0].tolist() == [True] * 4
        assert file["future_valid_mask"][-1].tolist() == [False] * 4

    dataset = A2DFlowDataset(
        cfg=A2DConfig(
            data_dir=str(data_dir),
            image_keys=("rgb_head",),
            image_size=4,
            action_horizon=16,
            video_key="rgb_head",
            joint_wam_enabled=True,
            joint_video_latent_cache_dir=str(cache_dir),
            dataset_manifest=dataset_path.name,
            aug_random_crop_pad=0,
        ),
        episodes=[{
            "path": str(source_path),
            "file_name": source_path.name,
            "length": length,
            "content_hash": "episode-content-hash",
        }],
        norm_stats={
            "state": {"min": [0.0] * 13, "span": [1.0] * 13},
            "action": {"min": [0.0] * 13, "span": [1.0] * 13},
        },
        train=False,
    )
    first = dataset[0]
    tail = dataset[len(dataset) - 1]
    assert first["joint_video_condition_latent"].shape == (2, 3, 2, 2)
    assert first["joint_video_future_mask"].tolist() == [True] * 4
    assert tail["joint_video_future_mask"].tolist() == [False] * 4
    assert tail["action_mask"].tolist() == [1.0] + [0.0] * 15
    dataset.close()

    first_run_calls = FakeJointCodec.calls
    precompute.main()
    assert FakeJointCodec.calls == first_run_calls
