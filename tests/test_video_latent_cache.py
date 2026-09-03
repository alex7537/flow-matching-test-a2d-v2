from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from scripts import precompute_wan_video_latents as precompute


class FakeCodec:
    calls = 0

    def __init__(self, **kwargs) -> None:
        del kwargs

    def encode_condition_future(self, condition, future):
        type(self).calls += 1
        batch = condition.shape[0]
        return (
            torch.full((batch, 2, 2, 2), 3.0),
            torch.full((batch, 2, 4, 2, 2), 5.0),
        )


def test_precompute_builds_and_reuses_dataset_bound_cache(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    source_path = data_dir / "episode.hdf5"
    length = 30
    with h5py.File(source_path, "w") as file:
        observations = file.create_group("observations")
        observations.create_dataset(
            "rgb_head",
            data=np.stack([
                np.full((4, 4, 3), index, dtype=np.uint8) for index in range(length)
            ]),
        )
    dataset_manifest = {
        "episodes": [{
            "file_name": source_path.name,
            "length": length,
            "content_hash": "episode-content-hash",
            "file_sha256": "episode-file-sha",
        }]
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest), encoding="utf-8"
    )
    vae_path = tmp_path / "vae.pth"
    vae_path.write_bytes(b"fake-vae")
    args = SimpleNamespace(
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        dataset_manifest="dataset_manifest.json",
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
        batch_size=2,
        latent_channels=2,
        latent_height=2,
        latent_width=2,
        device="cpu",
    )
    monkeypatch.setattr(precompute, "parse_args", lambda: args)
    monkeypatch.setattr(precompute, "FrozenWanVaeCodec", FakeCodec)
    FakeCodec.calls = 0

    precompute.main()

    manifest = json.loads((cache_dir / "video_latent_manifest.json").read_text())
    assert manifest["total_valid_windows"] == 6
    assert manifest["episodes"][0]["first_anchor_t"] == 8
    assert manifest["episodes"][0]["last_anchor_t"] == 13
    cache_path = cache_dir / manifest["episodes"][0]["cache_file_name"]
    with h5py.File(cache_path, "r") as file:
        assert file["condition_last"].shape == (6, 2, 2, 2)
        assert file["future_target"].shape == (6, 2, 4, 2, 2)
        assert file["condition_last"].dtype == np.dtype("float16")
    first_run_calls = FakeCodec.calls
    assert first_run_calls == 3

    precompute.main()

    assert FakeCodec.calls == first_run_calls
