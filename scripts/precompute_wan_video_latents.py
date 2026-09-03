#!/usr/bin/env python3
"""Build a dataset-bound, resumable Wan VAE latent cache for video-aux training."""

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

try:
    from turbojpeg import TurboJPEG

    _JPEG = TurboJPEG()
except Exception:
    _JPEG = None


SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def decode_rgb(item: np.ndarray) -> np.ndarray:
    import cv2

    if getattr(item, "ndim", 0) == 3:
        image_bgr = np.asarray(item)
    else:
        raw = item.tobytes() if isinstance(item, np.ndarray) else bytes(item)
        if _JPEG is not None:
            image_bgr = _JPEG.decode(raw)
        else:
            image_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("failed to decode JPEG frame")
    return image_bgr[..., ::-1]


def load_video_window(
    dataset: h5py.Dataset,
    indices: range,
    image_size: int,
) -> torch.Tensor:
    import cv2

    frames = []
    for index in indices:
        image = decode_rgb(dataset[index])
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frame = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        frames.append(frame.div_(127.5).sub_(1.0))
    return torch.stack(frames)


def cache_file_name(source_file_name: str) -> str:
    return f"{Path(source_file_name).stem}.wan_latents.hdf5"


def valid_existing_cache(
    path: Path,
    *,
    source_content_hash: str,
    vae_sha256: str,
    cache_contract_sha256: str,
    expected_windows: int,
    condition_shape: tuple[int, ...],
    future_shape: tuple[int, ...],
) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as file:
            return (
                int(file.attrs.get("schema_version", -1)) == SCHEMA_VERSION
                and str(file.attrs.get("source_content_hash", "")) == source_content_hash
                and str(file.attrs.get("wan_vae_sha256", "")) == vae_sha256
                and str(file.attrs.get("cache_contract_sha256", ""))
                == cache_contract_sha256
                and file["condition_last"].shape == (expected_windows, *condition_shape)
                and file["future_target"].shape == (expected_windows, *future_shape)
            )
    except (OSError, KeyError, ValueError):
        return False


def write_episode_cache(
    path: Path,
    *,
    condition_latents: np.ndarray,
    future_latents: np.ndarray,
    source_file_name: str,
    source_content_hash: str,
    vae_sha256: str,
    cache_contract_sha256: str,
    first_anchor_t: int,
    last_anchor_t: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as file:
            file.attrs["schema_version"] = SCHEMA_VERSION
            file.attrs["source_file_name"] = source_file_name
            file.attrs["source_content_hash"] = source_content_hash
            file.attrs["wan_vae_sha256"] = vae_sha256
            file.attrs["cache_contract_sha256"] = cache_contract_sha256
            file.attrs["first_anchor_t"] = first_anchor_t
            file.attrs["last_anchor_t"] = last_anchor_t
            file.create_dataset("condition_last", data=condition_latents, compression="lzf")
            file.create_dataset("future_target", data=future_latents, compression="lzf")
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
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    data_dir = Path(args.data_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    manifest_path = data_dir / args.dataset_manifest
    manifest_bytes = manifest_path.read_bytes()
    dataset_manifest = json.loads(manifest_bytes)
    episodes = dataset_manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("dataset manifest has no episodes")
    vae_path = Path(args.vae_checkpoint).expanduser().resolve()
    if not vae_path.is_file():
        raise FileNotFoundError(f"Wan VAE checkpoint not found: {vae_path}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    vae_sha = sha256_file(vae_path)
    dataset_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    cache_contract = {
        "schema_version": SCHEMA_VERSION,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "wan_vae_sha256": vae_sha,
        "video_key": args.video_key,
        "image_size": args.image_size,
        "condition_steps": args.condition_steps,
        "future_steps": args.future_steps,
        "future_offset_steps": args.future_offset_steps,
        "preprocessing": "RGB resize INTER_AREA, uint8/127.5-1, no augmentation",
        "latent_channels": args.latent_channels,
        "latent_height": args.latent_height,
        "latent_width": args.latent_width,
    }
    cache_contract_sha = hashlib.sha256(
        json.dumps(cache_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    condition_shape = (args.latent_channels, args.latent_height, args.latent_width)
    future_shape = (
        args.latent_channels,
        args.future_steps // 4,
        args.latent_height,
        args.latent_width,
    )
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
    for episode_index, episode in enumerate(episodes, start=1):
        source_name = str(episode["file_name"])
        source_hash = str(episode["content_hash"])
        source_path = data_dir / source_name
        cache_path = cache_dir / cache_file_name(source_name)
        length = int(episode["length"])
        first_anchor = args.condition_steps - 1
        last_anchor = length - args.future_offset_steps - args.future_steps
        anchors = list(range(first_anchor, last_anchor + 1))
        if valid_existing_cache(
            cache_path,
            source_content_hash=source_hash,
            vae_sha256=vae_sha,
            cache_contract_sha256=cache_contract_sha,
            expected_windows=len(anchors),
            condition_shape=condition_shape,
            future_shape=future_shape,
        ):
            print(f"[cache] reuse {episode_index}/{len(episodes)} {source_name}")
        else:
            condition_outputs: list[np.ndarray] = []
            future_outputs: list[np.ndarray] = []
            with h5py.File(source_path, "r", libver="latest", swmr=True) as source:
                video = source[args.obs_group][args.video_key]
                if int(video.shape[0]) != length:
                    raise ValueError(f"{source_name}: video length does not match manifest")
                for batch_start in range(0, len(anchors), args.batch_size):
                    batch_anchors = anchors[batch_start : batch_start + args.batch_size]
                    conditions = []
                    futures = []
                    for anchor in batch_anchors:
                        conditions.append(load_video_window(
                            video,
                            range(anchor - args.condition_steps + 1, anchor + 1),
                            args.image_size,
                        ))
                        future_start = anchor + args.future_offset_steps
                        futures.append(load_video_window(
                            video,
                            range(future_start, future_start + args.future_steps),
                            args.image_size,
                        ))
                    condition_batch = torch.stack(conditions).to(device=device)
                    future_batch = torch.stack(futures).to(device=device)
                    condition_latent, future_latent = codec.encode_condition_future(
                        condition_batch, future_batch
                    )
                    if tuple(condition_latent.shape[1:]) != condition_shape:
                        raise ValueError(
                            f"unexpected condition latent shape {tuple(condition_latent.shape[1:])}"
                        )
                    if tuple(future_latent.shape[1:]) != future_shape:
                        raise ValueError(
                            f"unexpected future latent shape {tuple(future_latent.shape[1:])}"
                        )
                    condition_outputs.append(condition_latent.cpu().numpy().astype(np.float16))
                    future_outputs.append(future_latent.cpu().numpy().astype(np.float16))
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
            write_episode_cache(
                cache_path,
                condition_latents=condition_array,
                future_latents=future_array,
                source_file_name=source_name,
                source_content_hash=source_hash,
                vae_sha256=vae_sha,
                cache_contract_sha256=cache_contract_sha,
                first_anchor_t=first_anchor,
                last_anchor_t=last_anchor,
            )
            print(
                f"[cache] wrote {episode_index}/{len(episodes)} {source_name} "
                f"windows={len(anchors)}"
            )
        records.append({
            "source_file_name": source_name,
            "source_content_hash": source_hash,
            "source_file_sha256": episode.get("file_sha256"),
            "cache_file_name": cache_path.name,
            "cache_file_sha256": sha256_file(cache_path),
            "first_anchor_t": first_anchor,
            "last_anchor_t": last_anchor,
            "num_windows": len(anchors),
        })
        total_windows += len(anchors)

    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "cache_contract": cache_contract,
        "cache_contract_sha256": cache_contract_sha,
        "wan_vae_checkpoint": str(vae_path),
        "wan_vae_sha256": vae_sha,
        "video_key": args.video_key,
        "image_size": args.image_size,
        "condition_steps": args.condition_steps,
        "future_steps": args.future_steps,
        "future_offset_steps": args.future_offset_steps,
        "preprocessing": "RGB resize INTER_AREA, uint8/127.5-1, no augmentation",
        "latent_dtype": "float16",
        "condition_latent_shape": list(condition_shape),
        "future_latent_shape": list(future_shape),
        "episode_count": len(records),
        "total_valid_windows": total_windows,
        "episodes": records,
    }
    output_path = cache_dir / "video_latent_manifest.json"
    write_json_atomic(output_path, output_manifest)
    print(f"VIDEO_LATENT_CACHE_OK manifest={output_path} windows={total_windows}")


if __name__ == "__main__":
    main()
