from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flow_matching_test.a2d_dataset import (
    A2DConfig,
    build_index,
    load_norm_stats,
    split_episodes,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> bytes:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(description="Create immutable A2D dataset and split manifests")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--image-keys", nargs="+", default=["rgb_head", "rgb_right_hand"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--norm-stats", default="norm_stats.json")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    cfg = A2DConfig(
        data_dir=str(data_dir),
        image_keys=tuple(args.image_keys),
        seed=args.seed,
        val_ratio=args.val_ratio,
        norm_stats=args.norm_stats,
    )
    episodes = build_index(cfg, force=True)
    train, val = split_episodes(episodes, cfg.val_ratio, cfg.seed)
    stats = load_norm_stats(cfg, train)

    manifest_episodes = []
    for episode in episodes:
        path = Path(episode["path"])
        manifest_episodes.append({
            "file_name": episode["file_name"],
            "bytes": path.stat().st_size,
            "length": int(episode["length"]),
            "file_sha256": _sha256(path),
            "content_hash": episode["content_hash"],
        })
    dataset_manifest = {
        "schema_version": 1,
        "data_version": args.data_version,
        "image_keys": list(cfg.image_keys),
        "episodes": manifest_episodes,
        "norm_stats": {
            "file_name": cfg.norm_stats,
            "sha256": _sha256(data_dir / cfg.norm_stats),
            "train_episode_digest": stats["train_episode_digest"],
        },
    }
    dataset_path = data_dir / "dataset_manifest.json"
    dataset_bytes = _write_json_atomic(dataset_path, dataset_manifest)

    split_manifest = {
        "schema_version": 1,
        "data_version": args.data_version,
        "dataset_manifest_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "seed": cfg.seed,
        "val_ratio": cfg.val_ratio,
        "train_episodes": [episode["file_name"] for episode in train],
        "val_episodes": [episode["file_name"] for episode in val],
    }
    split_path = data_dir / "split_manifest.json"
    _write_json_atomic(split_path, split_manifest)
    print(json.dumps({
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": split_manifest["dataset_manifest_sha256"],
        "split_manifest": str(split_path),
        "train_episode_digest": stats["train_episode_digest"],
        "train_episodes": split_manifest["train_episodes"],
        "val_episodes": split_manifest["val_episodes"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
