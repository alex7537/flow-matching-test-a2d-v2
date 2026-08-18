"""Create a derived A2D dataset without exact adjacent joint-position duplicates.

For every maximal run of exactly equal 13-D executed joint positions, keep the
last frame. Keeping the last frame preserves the post-transition phase label
(notably the terminal ``lift`` frame in data-rgb-450gb-v1).

The source directory is never modified. The destination is built in a sibling
temporary directory and atomically renamed only after all manifests validate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flow_matching_test.a2d_dataset import (
    A2DConfig,
    build_index,
    compute_norm_stats,
)
from flow_matching_test.action_contract import EXECUTED_ACTION_SEMANTICS
from flow_matching_test.segmentation import (
    SEGMENTATION_VERSION,
    compute_executed_action_segments,
)


FILTER_VERSION = 1
FILTER_POLICY = "exact_13d_equal_keep_last"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> bytes:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return encoded


def copy_attrs(source: h5py.AttributeManager, destination: h5py.AttributeManager) -> None:
    for key, value in source.items():
        destination[key] = value


def exact_keep_last_mask(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if action.ndim != 2 or action.shape[1] != 13:
        raise ValueError(f"action must have shape [T,13], got {action.shape}")
    duplicate_edges = np.all(action[:-1] == action[1:], axis=1)
    keep = np.ones(len(action), dtype=bool)
    keep[:-1] &= ~duplicate_edges
    return keep, duplicate_edges


def copy_vlen_dataset(
    source: h5py.Dataset,
    destination_group: h5py.Group,
    name: str,
    indices: np.ndarray,
) -> h5py.Dataset:
    destination = destination_group.create_dataset(
        name,
        shape=(len(indices),),
        dtype=source.dtype,
    )
    for output_index, source_index in enumerate(indices):
        destination[output_index] = source[int(source_index)]
    copy_attrs(source.attrs, destination.attrs)
    return destination


def filter_episode(
    source_path: Path,
    destination_path: Path,
    *,
    source_data_version: str,
    source_manifest_sha256: str,
    split: str,
) -> list[dict[str, Any]]:
    with h5py.File(source_path, "r") as source:
        action = np.asarray(source["action"][:], dtype=np.float32)
        qpos = np.asarray(source["observations/qpos"][:], dtype=np.float32)
        keep, duplicate_edges = exact_keep_last_mask(qpos)
        indices = np.flatnonzero(keep)
        removed_indices = np.flatnonzero(~keep)
        filtered_action = action[indices]
        filtered_qpos = qpos[indices]
        motion_threshold = float(source.attrs["segmentation_motion_threshold"])
        keyframe_threshold = float(source.attrs["segmentation_keyframe_threshold"])
        segment_type, arm_keyframe = compute_executed_action_segments(
            filtered_qpos,
            motion_threshold=motion_threshold,
            keyframe_threshold=keyframe_threshold,
        )
        phase_values = source["phase"][:]
        decoded_phases = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in phase_values
        ]

        temporary_path = destination_path.with_suffix(".tmp")
        with h5py.File(temporary_path, "w", libver="latest") as destination:
            copy_attrs(source.attrs, destination.attrs)
            destination.attrs["num_steps"] = int(len(indices))
            destination.attrs["derived_from_data_version"] = source_data_version
            destination.attrs["derived_from_processed_file"] = source_path.name
            destination.attrs["derived_from_dataset_manifest_sha256"] = (
                source_manifest_sha256
            )
            destination.attrs["exact_duplicate_filter_version"] = FILTER_VERSION
            destination.attrs["exact_duplicate_filter_policy"] = FILTER_POLICY
            destination.attrs["exact_duplicate_filter_source"] = "observations/qpos"
            destination.attrs["exact_duplicate_frames_removed"] = int(
                len(removed_indices)
            )
            destination.attrs["segmentation_version"] = SEGMENTATION_VERSION

            observations = destination.create_group("observations")
            copy_attrs(source["observations"].attrs, observations.attrs)
            observations.create_dataset("qpos", data=filtered_qpos)
            copy_attrs(
                source["observations/qpos"].attrs,
                observations["qpos"].attrs,
            )
            for name, dataset in source["observations"].items():
                if name == "qpos":
                    continue
                if dataset.shape[0] != len(action):
                    raise ValueError(
                        f"{source_path}: observations/{name} is not frame-aligned"
                    )
                copy_vlen_dataset(dataset, observations, name, indices)

            destination.create_dataset("action", data=filtered_action)
            copy_attrs(source["action"].attrs, destination["action"].attrs)
            destination.create_dataset(
                "phase",
                data=np.asarray(
                    [decoded_phases[index] for index in indices],
                    dtype=object,
                ),
                dtype=h5py.string_dtype(),
            )
            copy_attrs(source["phase"].attrs, destination["phase"].attrs)
            destination.create_dataset("segment_type", data=segment_type)
            destination.create_dataset("arm_keyframe", data=arm_keyframe)

        temporary_path.replace(destination_path)

    old_to_new = np.full(len(action), -1, dtype=np.int64)
    old_to_new[indices] = np.arange(len(indices), dtype=np.int64)
    removed_rows = []
    for removed_index in removed_indices:
        kept_equal_index = int(removed_index + 1)
        removed_rows.append(
            {
                "episode": source_path.name,
                "split": split,
                "removed_old_frame": int(removed_index),
                "kept_equal_old_frame": kept_equal_index,
                "kept_equal_new_frame": int(old_to_new[kept_equal_index]),
                "phase_removed": decoded_phases[int(removed_index)],
                "phase_kept": decoded_phases[kept_equal_index],
                "source_transition_exact_equal": bool(
                    duplicate_edges[int(removed_index)]
                ),
            }
        )
    return removed_rows


def inspect_filtered_episode(
    source_path: Path,
    destination_path: Path,
    *,
    split: str,
) -> list[dict[str, Any]]:
    with h5py.File(source_path, "r") as source:
        action = np.asarray(source["action"][:], dtype=np.float32)
        qpos = np.asarray(source["observations/qpos"][:], dtype=np.float32)
        phases = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["phase"][:]
        ]
    keep, duplicate_edges = exact_keep_last_mask(qpos)
    indices = np.flatnonzero(keep)
    removed_indices = np.flatnonzero(~keep)
    with h5py.File(destination_path, "r") as destination:
        filtered_action = np.asarray(destination["action"][:], dtype=np.float32)
        filtered_qpos = np.asarray(destination["observations/qpos"][:], dtype=np.float32)
        if str(destination.attrs.get("exact_duplicate_filter_policy", "")) != FILTER_POLICY:
            raise ValueError(f"{destination_path}: filter policy mismatch")
        if not np.array_equal(filtered_action, action[indices]):
            raise ValueError(f"{destination_path}: filtered action does not match source")
        if not np.array_equal(filtered_qpos, qpos[indices]):
            raise ValueError(f"{destination_path}: filtered qpos does not match source")
        if np.any(np.all(filtered_qpos[:-1] == filtered_qpos[1:], axis=1)):
            raise ValueError(f"{destination_path}: exact duplicate remains")

    old_to_new = np.full(len(action), -1, dtype=np.int64)
    old_to_new[indices] = np.arange(len(indices), dtype=np.int64)
    return [
        {
            "episode": source_path.name,
            "split": split,
            "removed_old_frame": int(index),
            "kept_equal_old_frame": int(index + 1),
            "kept_equal_new_frame": int(old_to_new[index + 1]),
            "phase_removed": phases[int(index)],
            "phase_kept": phases[int(index + 1)],
            "source_transition_exact_equal": bool(duplicate_edges[int(index)]),
        }
        for index in removed_indices
    ]


def create_manifests_and_stats(
    data_dir: Path,
    *,
    data_version: str,
    image_keys: tuple[str, ...],
    source_split: dict[str, Any],
    action_semantics: str = EXECUTED_ACTION_SEMANTICS,
) -> dict[str, Any]:
    cfg = A2DConfig(
        data_dir=str(data_dir),
        image_keys=image_keys,
        seed=int(source_split["seed"]),
        val_ratio=float(source_split["val_ratio"]),
        norm_stats="norm_stats.json",
        action_semantics=action_semantics,
    )
    episodes = build_index(cfg, force=True)
    episodes_by_name = {str(item["file_name"]): item for item in episodes}
    train_names = [str(name) for name in source_split["train_episodes"]]
    val_names = [str(name) for name in source_split["val_episodes"]]
    if set(episodes_by_name) != set(train_names).union(val_names):
        raise ValueError("derived episodes do not match the source split membership")
    train = [episodes_by_name[name] for name in train_names]
    stats = compute_norm_stats(cfg, train)

    manifest_episodes = []
    for episode in episodes:
        path = Path(episode["path"])
        manifest_episodes.append(
            {
                "file_name": episode["file_name"],
                "bytes": path.stat().st_size,
                "length": int(episode["length"]),
                "file_sha256": sha256_file(path),
                "content_hash": episode["content_hash"],
            }
        )
    dataset_manifest = {
        "schema_version": 1,
        "data_version": data_version,
        "image_keys": list(image_keys),
        "episodes": manifest_episodes,
        "norm_stats": {
            "file_name": cfg.norm_stats,
            "sha256": sha256_file(data_dir / cfg.norm_stats),
            "train_episode_digest": stats["train_episode_digest"],
        },
    }
    dataset_bytes = write_json_atomic(
        data_dir / "dataset_manifest.json", dataset_manifest
    )
    split_manifest = {
        "schema_version": 1,
        "data_version": data_version,
        "dataset_manifest_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "seed": cfg.seed,
        "val_ratio": cfg.val_ratio,
        "train_episodes": train_names,
        "val_episodes": val_names,
    }
    write_json_atomic(data_dir / "split_manifest.json", split_manifest)
    return {
        "episodes": len(episodes),
        "frames": sum(int(item["length"]) for item in episodes),
        "train_episode_digest": stats["train_episode_digest"],
        "dataset_manifest_sha256": split_manifest["dataset_manifest_sha256"],
        "split_manifest_sha256": sha256_file(data_dir / "split_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    destination_dir = args.destination.resolve()
    if source_dir == destination_dir:
        raise ValueError("source and destination must differ")
    if destination_dir.exists():
        raise FileExistsError(f"destination already exists: {destination_dir}")
    building_dir = destination_dir.parent / f".{destination_dir.name}.building"
    if building_dir.exists() and not args.resume:
        raise FileExistsError(f"incomplete building directory exists: {building_dir}")

    source_manifest_path = source_dir / "dataset_manifest.json"
    source_split_path = source_dir / "split_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_split = json.loads(source_split_path.read_text(encoding="utf-8"))
    source_manifest_sha256 = sha256_file(source_manifest_path)
    if source_split["dataset_manifest_sha256"] != source_manifest_sha256:
        raise ValueError("source split manifest is not bound to the source dataset")
    source_files = [str(item["file_name"]) for item in source_manifest["episodes"]]
    with h5py.File(source_dir / source_files[0], "r") as first_source:
        source_action_semantics = str(
            first_source.attrs.get("action_semantics", EXECUTED_ACTION_SEMANTICS)
        )
    split_by_name = {
        **{str(name): "train" for name in source_split["train_episodes"]},
        **{str(name): "val" for name in source_split["val_episodes"]},
    }
    if set(source_files) != set(split_by_name):
        raise ValueError("source dataset and split membership differ")

    building_dir.mkdir(exist_ok=args.resume)
    removed_rows: list[dict[str, Any]] = []
    for episode_number, file_name in enumerate(source_files, start=1):
        source_path = source_dir / file_name
        building_path = building_dir / file_name
        if building_path.exists():
            if not args.resume:
                raise FileExistsError(f"unexpected existing output: {building_path}")
            rows = inspect_filtered_episode(
                source_path,
                building_path,
                split=split_by_name[file_name],
            )
        else:
            rows = filter_episode(
                source_path,
                building_path,
                source_data_version=str(source_manifest["data_version"]),
                source_manifest_sha256=source_manifest_sha256,
                split=split_by_name[file_name],
            )
        removed_rows.extend(rows)
        if episode_number % 100 == 0 or episode_number == len(source_files):
            print(f"filtered {episode_number}/{len(source_files)} episodes")

    manifest_result = create_manifests_and_stats(
        building_dir,
        data_version=args.data_version,
        image_keys=tuple(str(value) for value in source_manifest["image_keys"]),
        source_split=source_split,
        action_semantics=source_action_semantics,
    )
    removed_path = building_dir / "removed_exact_duplicate_frames.csv"
    with removed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode",
                "split",
                "removed_old_frame",
                "kept_equal_old_frame",
                "kept_equal_new_frame",
                "phase_removed",
                "phase_kept",
                "source_transition_exact_equal",
            ],
        )
        writer.writeheader()
        writer.writerows(removed_rows)

    derivation = {
        "schema_version": 1,
        "filter_version": FILTER_VERSION,
        "filter_policy": FILTER_POLICY,
        "source_data_dir": str(source_dir),
        "source_data_version": source_manifest["data_version"],
        "source_dataset_manifest_sha256": source_manifest_sha256,
        "source_split_manifest_sha256": sha256_file(source_split_path),
        "destination_data_version": args.data_version,
        "keep_rule": (
            "For each exact-equality edge action[t] == action[t+1], "
            "remove t and keep t+1."
        ),
        "frames_before": sum(
            int(item["length"]) for item in source_manifest["episodes"]
        ),
        "frames_removed": len(removed_rows),
        "frames_after": manifest_result["frames"],
        "episodes": manifest_result["episodes"],
        "train_removed": sum(row["split"] == "train" for row in removed_rows),
        "val_removed": sum(row["split"] == "val" for row in removed_rows),
        "removed_frames_csv": removed_path.name,
        **manifest_result,
    }
    write_json_atomic(building_dir / "derivation_manifest.json", derivation)
    building_dir.replace(destination_dir)
    print(json.dumps(derivation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
