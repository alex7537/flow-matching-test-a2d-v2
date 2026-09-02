"""Create a fresh, task-balanced A2D dataset without copying RGB payloads.

Each ``--source`` is ``TASK=PROCESSED_DATASET_DIR``. The script samples the
same number of episodes from every task, creates namespaced hard links, makes a
fresh per-task train/val split, and recomputes normalization on the combined
train set. Source datasets and their historical split roles are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TASK_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
HYBRID_ACTION_SEMANTICS = "arm_executed_hand_commanded_joint_position"
SEGMENTATION_VERSION = 1
STATS_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> bytes:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(encoded)
    return encoded


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be TASK=DATASET_DIR")
    task, raw_path = value.split("=", 1)
    task = task.strip().lower()
    if not TASK_PATTERN.fullmatch(task):
        raise argparse.ArgumentTypeError(f"invalid task name: {task!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"source directory not found: {path}")
    return task, path


def stable_seed(seed: int, task: str) -> int:
    digest = hashlib.sha256(f"{seed}:{task}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def compute_norm_stats(
    data_dir: Path,
    train_episodes: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    range_eps: float,
) -> dict[str, Any]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in train_episodes:
        with h5py.File(data_dir / str(episode["file_name"]), "r", libver="latest") as handle:
            length = int(episode["length"])
            selected = np.linspace(0, length - 1, min(length, 500)).astype(int)
            states.append(np.asarray(handle["observations/qpos"][:], dtype=np.float64)[selected])
            actions.append(np.asarray(handle["action"][:], dtype=np.float64)[selected])

    def minmax(values: np.ndarray) -> dict[str, Any]:
        lo = values.min(axis=0)
        hi = values.max(axis=0)
        raw_range = hi - lo
        span = np.maximum(raw_range, range_eps)
        unique = [
            int(np.unique(np.round(values[:, index], 6)).size)
            for index in range(values.shape[1])
        ]
        return {
            "min": lo.tolist(),
            "max": hi.tolist(),
            "span": span.tolist(),
            "raw_range": raw_range.tolist(),
            "unique_count_1e-6": unique,
            "warnings": [
                {"dim": index, "range": float(raw_range[index]), "unique_count": unique[index]}
                for index in range(values.shape[1])
                if raw_range[index] < 1.0e-3 or unique[index] <= 2
            ],
        }

    train_hashes = sorted(str(episode["content_hash"]) for episode in train_episodes)
    train_digest = hashlib.sha256(
        json.dumps(train_hashes, separators=(",", ":")).encode()
    ).hexdigest()
    stats = {
        "schema_version": STATS_SCHEMA_VERSION,
        "train_episode_hashes": train_hashes,
        "train_episode_digest": train_digest,
        "train_episode_count": len(train_hashes),
        "split_metadata": {"seed": int(seed), "val_ratio": float(val_ratio)},
        "action_semantics": HYBRID_ACTION_SEMANTICS,
        "normalization": "train_minmax",
        "range_eps": float(range_eps),
        "state": minmax(np.concatenate(states, axis=0)),
        "action": minmax(np.concatenate(actions, axis=0)),
    }
    write_json(data_dir / "norm_stats.json", stats)
    return stats


def select_task_episodes(
    episodes: list[dict[str, Any]],
    *,
    task: str,
    total: int,
    val_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if total > len(episodes):
        raise ValueError(f"task {task!r} has {len(episodes)} episodes, requested {total}")
    if not 0 < val_count < total:
        raise ValueError("val_count must be between zero and total")
    candidates = sorted(episodes, key=lambda item: str(item["file_name"]))
    rng = random.Random(stable_seed(seed, task))
    rng.shuffle(candidates)
    selected = candidates[:total]
    val_names = {
        str(item["file_name"])
        for item in selected[:val_count]
    }
    train = [item for item in selected if str(item["file_name"]) not in val_names]
    val = [item for item in selected if str(item["file_name"]) in val_names]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=parse_source)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=300)
    parser.add_argument("--val-per-task", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion-threshold", type=float, default=1.0e-4)
    parser.add_argument("--keyframe-threshold", type=float, default=0.1)
    parser.add_argument("--range-eps", type=float, default=1.0e-4)
    args = parser.parse_args()

    sources = dict(args.source)
    if len(sources) != len(args.source) or len(sources) < 2:
        raise ValueError("provide at least two sources with unique task names")
    if args.episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")

    destination = args.destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()

    source_records: dict[str, dict[str, Any]] = {}
    selected_by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    image_keys: list[str] | None = None
    try:
        for task, source_dir in sorted(sources.items()):
            manifest_path = source_dir / "dataset_manifest.json"
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            current_keys = [str(value) for value in manifest["image_keys"]]
            current_semantics = str(
                manifest.get("action_semantics", HYBRID_ACTION_SEMANTICS)
            )
            if current_semantics != HYBRID_ACTION_SEMANTICS:
                raise ValueError(
                    f"task {task!r} uses incompatible action semantics: {current_semantics}"
                )
            if image_keys is None:
                image_keys = current_keys
            elif image_keys != current_keys:
                raise ValueError(f"task {task!r} uses image keys {current_keys}, expected {image_keys}")
            train, val = select_task_episodes(
                list(manifest["episodes"]),
                task=task,
                total=args.episodes_per_task,
                val_count=args.val_per_task,
                seed=args.seed,
            )
            selected_by_task[task] = {"train": train, "val": val}
            source_records[task] = {
                "data_dir": str(source_dir),
                "data_version": manifest.get("data_version"),
                "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "available_episodes": len(manifest["episodes"]),
            }

        assert image_keys is not None
        manifest_episodes: list[dict[str, Any]] = []
        train_names: list[str] = []
        val_names: list[str] = []
        cache_episodes: list[dict[str, Any]] = []

        for task, source_dir in sorted(sources.items()):
            for role in ("train", "val"):
                for episode in selected_by_task[task][role]:
                    source_name = str(episode["file_name"])
                    target_name = f"{task}__{source_name}"
                    source_path = source_dir / source_name
                    target_path = temporary / target_name
                    os.link(source_path, target_path)
                    if source_path.stat().st_ino != target_path.stat().st_ino:
                        raise RuntimeError(f"hard-link verification failed: {target_path}")
                    record = {
                        "file_name": target_name,
                        "bytes": int(episode["bytes"]),
                        "length": int(episode["length"]),
                        "file_sha256": str(episode["file_sha256"]),
                        "content_hash": str(episode["content_hash"]),
                        "task_id": task,
                        "source_file_name": source_name,
                        "source_data_version": source_records[task]["data_version"],
                    }
                    manifest_episodes.append(record)
                    cache_episodes.append(
                        {
                            "path": str(destination / target_name),
                            "file_name": target_name,
                            "length": int(episode["length"]),
                            "content_hash": str(episode["content_hash"]),
                        }
                    )
                    (train_names if role == "train" else val_names).append(target_name)

        manifest_episodes.sort(key=lambda item: str(item["file_name"]))
        train_names.sort()
        val_names.sort()
        cache_episodes.sort(key=lambda item: str(item["file_name"]))
        inventory = [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in sorted(temporary.glob("*.hdf5"))
        ]
        write_json(
            temporary / "index_cache.json",
            {
                "version": 5,
                "image_keys": image_keys,
                "action_semantics": HYBRID_ACTION_SEMANTICS,
                "file_inventory": inventory,
                "segmentation_version": SEGMENTATION_VERSION,
                "motion_threshold": float(args.motion_threshold),
                "keyframe_threshold": float(args.keyframe_threshold),
                "duplicate_groups": [],
                "episodes": cache_episodes,
            },
        )

        cache_by_name = {item["file_name"]: item for item in cache_episodes}
        train_episodes = []
        for name in train_names:
            item = dict(cache_by_name[name])
            item["path"] = str(temporary / name)
            train_episodes.append(item)
        val_ratio = len(val_names) / len(manifest_episodes)
        stats = compute_norm_stats(
            temporary,
            train_episodes,
            seed=int(args.seed),
            val_ratio=val_ratio,
            range_eps=float(args.range_eps),
        )
        stats_sha256 = sha256_file(temporary / "norm_stats.json")

        dataset_manifest = {
            "schema_version": 1,
            "data_version": args.data_version,
            "action_semantics": HYBRID_ACTION_SEMANTICS,
            "image_keys": image_keys,
            "tasks": sorted(sources),
            "episodes": manifest_episodes,
            "norm_stats": {
                "file_name": "norm_stats.json",
                "sha256": stats_sha256,
                "train_episode_digest": stats["train_episode_digest"],
            },
        }
        dataset_bytes = write_json(temporary / "dataset_manifest.json", dataset_manifest)
        split_manifest = {
            "schema_version": 1,
            "data_version": args.data_version,
            "dataset_manifest_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "seed": int(args.seed),
            "val_ratio": len(val_names) / len(manifest_episodes),
            "train_episodes": train_names,
            "val_episodes": val_names,
            "task_counts": {
                task: {
                    "train": len(selected_by_task[task]["train"]),
                    "val": len(selected_by_task[task]["val"]),
                }
                for task in sorted(sources)
            },
        }
        split_bytes = write_json(temporary / "split_manifest.json", split_manifest)
        write_json(
            temporary / "derivation_manifest.json",
            {
                "schema_version": 1,
                "data_version": args.data_version,
                "created_by": "scripts/create_balanced_multitask_dataset.py",
                "selection": {
                    "algorithm": "task-local deterministic shuffle without replacement",
                    "seed": int(args.seed),
                    "episodes_per_task": int(args.episodes_per_task),
                    "train_per_task": int(args.episodes_per_task - args.val_per_task),
                    "val_per_task": int(args.val_per_task),
                },
                "storage": "hard links; source RGB payloads are not copied",
                "sources": source_records,
                "dataset_manifest_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
                "train_episode_digest": stats["train_episode_digest"],
                "logical_bytes": sum(int(item["bytes"]) for item in manifest_episodes),
            },
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "destination": str(destination),
                "data_version": args.data_version,
                "tasks": sorted(sources),
                "train_episodes": len(train_names),
                "val_episodes": len(val_names),
                "dataset_manifest_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
                "train_episode_digest": stats["train_episode_digest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
