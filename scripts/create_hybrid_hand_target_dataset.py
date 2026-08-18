"""Derive v3 A2D data with executed arm waypoints and commanded hand targets.

The source processed dataset supplies RGB, actual qpos, phase, segmentation, and
the exact-dedup timeline. Raw episodes supply dense ``hand2_pos_target`` rows.
The source dataset and split are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flow_matching_test.action_contract import (
    HYBRID_ACTION_SEMANTICS,
    action_layout,
)


RAW_NAME_PATTERN = re.compile(
    r"^episode_env_(?P<env>\d{3})_(?P<episode>\d{6})_success\.hdf5$"
)
STATS_SCHEMA_VERSION = 2


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


def raw_episode_path(raw_root: Path, processed_name: str) -> Path:
    match = RAW_NAME_PATTERN.match(processed_name)
    if match is None:
        raise ValueError(f"unsupported processed episode name: {processed_name}")
    return (
        raw_root
        / f"env_{match.group('env')}"
        / "success"
        / f"episode_{match.group('episode')}_success.hdf5"
    )


def exact_keep_last_mask(executed_qpos: np.ndarray) -> np.ndarray:
    if executed_qpos.ndim != 2 or executed_qpos.shape[1] != 13:
        raise ValueError(f"executed_qpos must have shape [T,13], got {executed_qpos.shape}")
    duplicate_edges = np.all(executed_qpos[:-1] == executed_qpos[1:], axis=1)
    keep = np.ones(len(executed_qpos), dtype=bool)
    keep[:-1] &= ~duplicate_edges
    return keep


def load_raw_alignment(raw_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(raw_path, "r", libver="latest") as raw:
        trajectory = raw["trajectory"]
        arm_actual = np.asarray(trajectory["arm2_pos"][:], dtype=np.float32)
        hand_actual = np.asarray(trajectory["hand2_pos"][:], dtype=np.float32)
        hand_target = np.asarray(trajectory["hand2_pos_target"][:], dtype=np.float32)
    if arm_actual.shape[0] != hand_actual.shape[0] or hand_target.shape != hand_actual.shape:
        raise ValueError(f"{raw_path}: arm/hand/hand-target lengths do not align")
    if hand_target.shape[1:] != (6,) or not np.isfinite(hand_target).all():
        raise ValueError(f"{raw_path}: hand2_pos_target must be finite [T,6]")
    if np.any(np.all(hand_target == 0.0, axis=1)):
        raise ValueError(f"{raw_path}: all-zero hand target row requires a semantic decision")
    executed_qpos = np.concatenate([arm_actual, hand_actual], axis=1)
    keep = exact_keep_last_mask(executed_qpos)
    return executed_qpos[keep], hand_target[keep], hand_target


def validate_hybrid_episode(
    destination_path: Path,
    expected_qpos: np.ndarray,
    expected_hand_target: np.ndarray,
) -> None:
    with h5py.File(destination_path, "r", libver="latest") as destination:
        qpos = np.asarray(destination["observations/qpos"][:], dtype=np.float32)
        action = np.asarray(destination["action"][:], dtype=np.float32)
        if not np.array_equal(qpos, expected_qpos):
            raise ValueError(f"{destination_path}: processed qpos does not match raw dedup timeline")
        if not np.array_equal(action[:, :7], qpos[:, :7]):
            raise ValueError(f"{destination_path}: arm action changed")
        if not np.array_equal(action[:, 7:], expected_hand_target):
            raise ValueError(f"{destination_path}: hand action does not match raw hand target")
        if str(destination.attrs.get("action_semantics", "")) != HYBRID_ACTION_SEMANTICS:
            raise ValueError(f"{destination_path}: hybrid action semantics missing")


def materialize_episode(
    source_path: Path,
    raw_path: Path,
    destination_path: Path,
    *,
    source_manifest_sha256: str,
) -> tuple[int, int, str]:
    expected_qpos, expected_hand_target, raw_hand_target = load_raw_alignment(raw_path)
    target_sha256 = hashlib.sha256(raw_hand_target.tobytes()).hexdigest()
    if destination_path.exists():
        validate_hybrid_episode(destination_path, expected_qpos, expected_hand_target)
        return len(expected_qpos), len(raw_hand_target), target_sha256

    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source_path, temporary)
    try:
        with h5py.File(temporary, "r+", libver="latest") as destination:
            qpos = np.asarray(destination["observations/qpos"][:], dtype=np.float32)
            if not np.array_equal(qpos, expected_qpos):
                raise ValueError(f"{source_path}: source qpos does not match raw dedup timeline")
            action = np.concatenate([qpos[:, :7], expected_hand_target], axis=1)
            destination["action"][:] = action
            destination.attrs["action_layout"] = action_layout(HYBRID_ACTION_SEMANTICS)
            destination.attrs["action_semantics"] = HYBRID_ACTION_SEMANTICS
            destination.attrs["action_arm_source"] = "trajectory/arm2_pos"
            destination.attrs["action_hand_source"] = "trajectory/hand2_pos_target"
            destination.attrs["derived_from_dataset_manifest_sha256"] = source_manifest_sha256
            destination.attrs["source_hand_target_sha256"] = target_sha256
        temporary.replace(destination_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    validate_hybrid_episode(destination_path, expected_qpos, expected_hand_target)
    return len(expected_qpos), len(raw_hand_target), target_sha256


def episode_content_hash(path: Path, image_keys: list[str]) -> tuple[int, str]:
    digest = hashlib.sha256()
    with h5py.File(path, "r", libver="latest") as episode:
        qpos = np.asarray(episode["observations/qpos"][:], dtype=np.float32)
        action = np.asarray(episode["action"][:], dtype=np.float32)
        digest.update(qpos.tobytes())
        digest.update(action.tobytes())
        for key in image_keys:
            images = episode["observations"][key]
            for frame_index in range(len(action)):
                digest.update(np.asarray(images[frame_index], dtype=np.uint8).tobytes())
    return len(action), digest.hexdigest()


def minmax(values: np.ndarray, range_eps: float) -> dict[str, Any]:
    lo, hi = values.min(axis=0), values.max(axis=0)
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


def build_metadata(
    data_dir: Path,
    *,
    data_version: str,
    image_keys: list[str],
    source_split: dict[str, Any],
    range_eps: float,
) -> dict[str, Any]:
    train_names = [str(name) for name in source_split["train_episodes"]]
    val_names = [str(name) for name in source_split["val_episodes"]]
    all_names = train_names + val_names
    manifest_episodes = []
    content_by_name: dict[str, str] = {}
    for number, file_name in enumerate(sorted(all_names), start=1):
        path = data_dir / file_name
        length, content_hash = episode_content_hash(path, image_keys)
        content_by_name[file_name] = content_hash
        manifest_episodes.append(
            {
                "file_name": file_name,
                "bytes": path.stat().st_size,
                "length": length,
                "file_sha256": sha256_file(path),
                "content_hash": content_hash,
            }
        )
        if number % 100 == 0 or number == len(all_names):
            print(f"hashed {number}/{len(all_names)} episodes", flush=True)

    states, actions = [], []
    for file_name in train_names:
        with h5py.File(data_dir / file_name, "r", libver="latest") as episode:
            length = int(episode["action"].shape[0])
            selected = np.linspace(0, length - 1, min(length, 500)).astype(int)
            states.append(np.asarray(episode["observations/qpos"][:], dtype=np.float64)[selected])
            actions.append(np.asarray(episode["action"][:], dtype=np.float64)[selected])
    state_values = np.concatenate(states, axis=0)
    action_values = np.concatenate(actions, axis=0)
    train_hashes = sorted(content_by_name[name] for name in train_names)
    train_digest = hashlib.sha256(
        json.dumps(train_hashes, separators=(",", ":")).encode()
    ).hexdigest()
    stats = {
        "schema_version": STATS_SCHEMA_VERSION,
        "train_episode_hashes": train_hashes,
        "train_episode_digest": train_digest,
        "train_episode_count": len(train_names),
        "split_metadata": {
            "seed": int(source_split["seed"]),
            "val_ratio": float(source_split["val_ratio"]),
        },
        "action_semantics": HYBRID_ACTION_SEMANTICS,
        "normalization": "train_minmax",
        "range_eps": range_eps,
        "state": minmax(state_values, range_eps),
        "action": minmax(action_values, range_eps),
    }
    write_json_atomic(data_dir / "norm_stats.json", stats)
    dataset_manifest = {
        "schema_version": 1,
        "data_version": data_version,
        "image_keys": image_keys,
        "action_semantics": HYBRID_ACTION_SEMANTICS,
        "episodes": manifest_episodes,
        "norm_stats": {
            "file_name": "norm_stats.json",
            "sha256": sha256_file(data_dir / "norm_stats.json"),
            "train_episode_digest": train_digest,
        },
    }
    dataset_bytes = write_json_atomic(data_dir / "dataset_manifest.json", dataset_manifest)
    split_manifest = {
        "schema_version": 1,
        "data_version": data_version,
        "dataset_manifest_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "seed": int(source_split["seed"]),
        "val_ratio": float(source_split["val_ratio"]),
        "train_episodes": train_names,
        "val_episodes": val_names,
    }
    write_json_atomic(data_dir / "split_manifest.json", split_manifest)
    return {
        "episodes": len(manifest_episodes),
        "frames": sum(int(item["length"]) for item in manifest_episodes),
        "train_episode_digest": train_digest,
        "dataset_manifest_sha256": split_manifest["dataset_manifest_sha256"],
        "split_manifest_sha256": sha256_file(data_dir / "split_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--range-eps", type=float, default=1.0e-4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    raw_root = args.raw_root.resolve()
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
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    source_split = json.loads(source_split_path.read_text(encoding="utf-8"))
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    if source_split["dataset_manifest_sha256"] != source_manifest_sha256:
        raise ValueError("source split manifest is not bound to source dataset")
    source_names = [str(item["file_name"]) for item in source_manifest["episodes"]]
    split_names = set(source_split["train_episodes"]).union(source_split["val_episodes"])
    if set(source_names) != split_names:
        raise ValueError("source dataset and split membership differ")

    building_dir.mkdir(exist_ok=args.resume)
    raw_target_digest = hashlib.sha256()
    raw_frames = 0
    output_frames = 0
    for number, file_name in enumerate(source_names, start=1):
        raw_path = raw_episode_path(raw_root, file_name)
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        kept, raw_count, target_sha256 = materialize_episode(
            source_dir / file_name,
            raw_path,
            building_dir / file_name,
            source_manifest_sha256=source_manifest_sha256,
        )
        raw_target_digest.update(file_name.encode())
        raw_target_digest.update(bytes.fromhex(target_sha256))
        raw_frames += raw_count
        output_frames += kept
        if number % 100 == 0 or number == len(source_names):
            print(f"materialized {number}/{len(source_names)} episodes", flush=True)

    removed_csv = source_dir / "removed_exact_duplicate_frames.csv"
    if removed_csv.is_file():
        shutil.copy2(removed_csv, building_dir / removed_csv.name)
    metadata = build_metadata(
        building_dir,
        data_version=args.data_version,
        image_keys=[str(value) for value in source_manifest["image_keys"]],
        source_split=source_split,
        range_eps=float(args.range_eps),
    )
    derivation = {
        "schema_version": 1,
        "derivation_version": 1,
        "source_data_dir": str(source_dir),
        "source_data_version": source_manifest["data_version"],
        "source_dataset_manifest_sha256": source_manifest_sha256,
        "source_split_manifest_sha256": sha256_file(source_split_path),
        "raw_root": str(raw_root),
        "raw_hand_target_digest": raw_target_digest.hexdigest(),
        "destination_data_version": args.data_version,
        "action_semantics": HYBRID_ACTION_SEMANTICS,
        "action_layout": action_layout(HYBRID_ACTION_SEMANTICS),
        "state_source": "arm2_pos(7)+hand2_pos(6)",
        "action_source": "arm2_pos(7)+hand2_pos_target(6)",
        "timeline_source": "source v2 exact-dedup keep-last qpos mask",
        "raw_frames": raw_frames,
        "frames": output_frames,
        "episodes": len(source_names),
        "split_membership_preserved": True,
        **metadata,
    }
    write_json_atomic(building_dir / "derivation_manifest.json", derivation)
    building_dir.replace(destination_dir)
    print(json.dumps(derivation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
