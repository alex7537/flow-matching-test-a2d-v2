#!/usr/bin/env python3
"""Precompute full Wan prefix/future latents for compact A2D Joint WAM training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from flow_matching_test.policies.video_aux import FrozenWanVaeCodec
from scripts.precompute_wan_video_latents import load_video_window, sha256_file


SCHEMA_VERSION = 2
CACHE_KIND = "a2d_joint_wam_latents"


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cache_file_name(source_file_name: str) -> str:
    return f"{Path(source_file_name).stem}.joint_wam_latents.hdf5"


def cache_is_valid(
    path: Path,
    *,
    source_hash: str,
    contract_sha: str,
    windows: int,
    condition_shape: tuple[int, ...],
    future_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as file:
            return (
                int(file.attrs.get("schema_version", -1)) == SCHEMA_VERSION
                and str(file.attrs.get("cache_kind", "")) == CACHE_KIND
                and str(file.attrs.get("source_content_hash", "")) == source_hash
                and str(file.attrs.get("cache_contract_sha256", "")) == contract_sha
                and file["condition_prefix"].shape == (windows, *condition_shape)
                and file["future_target"].shape == (windows, *future_shape)
                and file["future_valid_mask"].shape == (windows, *mask_shape)
            )
    except (OSError, KeyError, ValueError):
        return False


def write_episode_cache(
    path: Path,
    *,
    condition: np.ndarray,
    future: np.ndarray,
    future_mask: np.ndarray,
    source_name: str,
    source_hash: str,
    contract_sha: str,
    first_anchor: int,
    last_anchor: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as file:
            file.attrs["schema_version"] = SCHEMA_VERSION
            file.attrs["cache_kind"] = CACHE_KIND
            file.attrs["source_file_name"] = source_name
            file.attrs["source_content_hash"] = source_hash
            file.attrs["cache_contract_sha256"] = contract_sha
            file.attrs["first_anchor_t"] = first_anchor
            file.attrs["last_anchor_t"] = last_anchor
            file.create_dataset("condition_prefix", data=condition, compression="lzf")
            file.create_dataset("future_target", data=future, compression="lzf")
            file.create_dataset("future_valid_mask", data=future_mask, compression="lzf")
            file.flush()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset-manifest", default="dataset_manifest.json")
    parser.add_argument("--obs-group", default="observations")
    parser.add_argument("--video-key", default="rgb_head")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--condition-steps", type=int, default=9)
    parser.add_argument("--future-steps", type=int, default=16)
    parser.add_argument("--future-offset-steps", type=int, default=1)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--wan-runtime-repo", required=True)
    parser.add_argument("--wan-runtime-site-packages", default="")
    parser.add_argument("--vae-dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latent-channels", type=int, default=48)
    parser.add_argument("--latent-height", type=int, default=14)
    parser.add_argument("--latent-width", type=int, default=14)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.condition_steps != 9 or args.future_steps != 16:
        raise ValueError("Joint WAM V1 requires the 9+16 temporal contract")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    data_dir = Path(args.data_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest_path = data_dir / args.dataset_manifest
    dataset_bytes = dataset_manifest_path.read_bytes()
    dataset_manifest = json.loads(dataset_bytes)
    episodes = dataset_manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("dataset manifest has no episodes")
    vae_path = Path(args.vae_checkpoint).expanduser().resolve()
    if not vae_path.is_file():
        raise FileNotFoundError(f"Wan VAE checkpoint not found: {vae_path}")
    vae_sha = sha256_file(vae_path)
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    condition_shape = (
        args.latent_channels,
        1 + (args.condition_steps - 1) // 4,
        args.latent_height,
        args.latent_width,
    )
    future_shape = (
        args.latent_channels,
        args.future_steps // 4,
        args.latent_height,
        args.latent_width,
    )
    mask_shape = (args.future_steps // 4,)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "cache_kind": CACHE_KIND,
        "dataset_manifest_sha256": dataset_sha,
        "wan_vae_sha256": vae_sha,
        "wan_runtime": "public Wan-Video/Wan2.2 wan.modules.vae2_2.Wan2_2_VAE",
        "video_key": args.video_key,
        "image_size": args.image_size,
        "condition_steps": args.condition_steps,
        "future_steps": args.future_steps,
        "future_offset_steps": args.future_offset_steps,
        "tail_padding": "repeat-last RGB",
        "latent_mask": "only complete groups of four real future RGB frames",
        "preprocessing": "RGB resize INTER_AREA, uint8/127.5-1, no augmentation",
        "condition_prefix_shape": list(condition_shape),
        "future_latent_shape": list(future_shape),
        "future_latent_mask_shape": list(mask_shape),
    }
    contract_sha = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    codec = FrozenWanVaeCodec(
        vae_checkpoint_path=str(vae_path),
        runtime_repo=args.wan_runtime_repo,
        runtime_site_packages=args.wan_runtime_site_packages,
        condition_steps=args.condition_steps,
        future_steps=args.future_steps,
        dtype=args.vae_dtype,
        latent_channels=args.latent_channels,
        encode_batch_size=args.batch_size,
    )
    device = torch.device(args.device)
    records = []
    total_windows = 0
    total_valid_latent_steps = 0
    for episode_index, episode in enumerate(episodes, start=1):
        source_name = str(episode["file_name"])
        source_hash = str(episode["content_hash"])
        source_path = data_dir / source_name
        length = int(episode["length"])
        first_anchor = args.condition_steps - 1
        last_anchor = length - args.future_offset_steps - 1
        anchors = list(range(first_anchor, last_anchor + 1))
        cache_path = cache_dir / cache_file_name(source_name)
        if cache_is_valid(
            cache_path,
            source_hash=source_hash,
            contract_sha=contract_sha,
            windows=len(anchors),
            condition_shape=condition_shape,
            future_shape=future_shape,
            mask_shape=mask_shape,
        ):
            print(f"[joint-cache] reuse {episode_index}/{len(episodes)} {source_name}")
        else:
            condition_outputs: list[np.ndarray] = []
            future_outputs: list[np.ndarray] = []
            mask_outputs: list[np.ndarray] = []
            with h5py.File(source_path, "r", libver="latest", swmr=True) as source:
                video = source[args.obs_group][args.video_key]
                for batch_start in range(0, len(anchors), args.batch_size):
                    batch_anchors = anchors[batch_start : batch_start + args.batch_size]
                    conditions, futures, masks = [], [], []
                    for anchor in batch_anchors:
                        conditions.append(load_video_window(
                            video,
                            range(anchor - args.condition_steps + 1, anchor + 1),
                            args.image_size,
                        ))
                        future_start = anchor + args.future_offset_steps
                        valid_frames = min(args.future_steps, length - future_start)
                        indices = list(range(future_start, future_start + valid_frames))
                        indices.extend([indices[-1]] * (args.future_steps - valid_frames))
                        futures.append(load_video_window(video, indices, args.image_size))
                        valid_latent_steps = valid_frames // 4
                        masks.append(np.arange(mask_shape[0]) < valid_latent_steps)
                    condition_batch = torch.stack(conditions).to(device)
                    future_batch = torch.stack(futures).to(device)
                    condition_latent, future_latent = codec.encode_joint_condition_future(
                        condition_batch, future_batch
                    )
                    if tuple(condition_latent.shape[1:]) != condition_shape:
                        raise ValueError("unexpected joint condition latent shape")
                    if tuple(future_latent.shape[1:]) != future_shape:
                        raise ValueError("unexpected joint future latent shape")
                    condition_outputs.append(condition_latent.cpu().numpy().astype(np.float16))
                    future_outputs.append(future_latent.cpu().numpy().astype(np.float16))
                    mask_outputs.append(np.asarray(masks, dtype=bool))
            condition_array = (
                np.concatenate(condition_outputs)
                if condition_outputs
                else np.empty((0, *condition_shape), dtype=np.float16)
            )
            future_array = (
                np.concatenate(future_outputs)
                if future_outputs
                else np.empty((0, *future_shape), dtype=np.float16)
            )
            mask_array = (
                np.concatenate(mask_outputs)
                if mask_outputs
                else np.empty((0, *mask_shape), dtype=bool)
            )
            write_episode_cache(
                cache_path,
                condition=condition_array,
                future=future_array,
                future_mask=mask_array,
                source_name=source_name,
                source_hash=source_hash,
                contract_sha=contract_sha,
                first_anchor=first_anchor,
                last_anchor=last_anchor,
            )
            print(
                f"[joint-cache] wrote {episode_index}/{len(episodes)} "
                f"{source_name} windows={len(anchors)}"
            )
        with h5py.File(cache_path, "r") as cached:
            valid_latent_steps = int(np.asarray(cached["future_valid_mask"], dtype=bool).sum())
        records.append({
            "source_file_name": source_name,
            "source_content_hash": source_hash,
            "source_file_sha256": episode.get("file_sha256"),
            "cache_file_name": cache_path.name,
            "cache_file_sha256": sha256_file(cache_path),
            "first_anchor_t": first_anchor,
            "last_anchor_t": last_anchor,
            "num_windows": len(anchors),
            "valid_future_latent_steps": valid_latent_steps,
        })
        total_windows += len(anchors)
        total_valid_latent_steps += valid_latent_steps
    manifest = {
        **contract,
        "cache_contract_sha256": contract_sha,
        "wan_vae_checkpoint": str(vae_path),
        "episode_count": len(records),
        "total_windows": total_windows,
        "total_valid_future_latent_steps": total_valid_latent_steps,
        "episodes": records,
    }
    output = cache_dir / "joint_video_latent_manifest.json"
    write_json_atomic(output, manifest)
    print(f"JOINT_VIDEO_LATENT_CACHE_OK manifest={output} windows={total_windows}")


if __name__ == "__main__":
    main()
