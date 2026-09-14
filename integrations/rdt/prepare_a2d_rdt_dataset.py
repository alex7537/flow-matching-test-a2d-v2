#!/usr/bin/env python3
"""Create an immutable, hard-linked A2D V3 dataset contract for RDT.

The RGB HDF5 payloads are not rewritten. This derives RDT's 128-dimensional
state/action statistics and auditable train/val metadata from train episodes
only, then publishes the result with an atomic directory rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

from a2d_hdf5_vla_dataset import (
    A2D_ACTION_INDICES,
    A2D_STATE_INDICES,
    ACTION_CHUNK_SIZE,
    ACTION_OFFSET_STEPS,
    DATASET_NAME,
    EXPECTED_ACTION_SEMANTICS,
    STATE_DIM,
    encode_a2d_vector,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Moments:
    def __init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(STATE_DIM, dtype=np.float64)
        self.sum_sq = np.zeros(STATE_DIM, dtype=np.float64)
        self.minimum = np.full(STATE_DIM, np.inf, dtype=np.float64)
        self.maximum = np.full(STATE_DIM, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        self.count += int(values.shape[0])
        self.sum += values.sum(axis=0)
        self.sum_sq += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def result(self) -> dict:
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        return {
            "count": self.count,
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--empty-lang-embed",
        type=Path,
        help="Official RDT empty embedding; enables language-free action-policy training.",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    empty_lang_embed = (
        args.empty_lang_embed.expanduser().resolve() if args.empty_lang_embed else None
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if source.stat().st_dev != output.parent.stat().st_dev:
        raise RuntimeError("source and output parent must share a filesystem for hard links")

    source_dataset_path = source / "dataset_manifest.json"
    source_split_path = source / "split_manifest.json"
    source_dataset = json.loads(source_dataset_path.read_text(encoding="utf-8"))
    source_split = json.loads(source_split_path.read_text(encoding="utf-8"))
    if source_dataset.get("action_semantics") != EXPECTED_ACTION_SEMANTICS:
        raise ValueError("source is not the expected A2D V3 hybrid-action dataset")
    records = {item["file_name"]: dict(item) for item in source_dataset["episodes"]}
    train_names = list(source_split["train_episodes"])
    val_names = list(source_split["val_episodes"])
    all_names = train_names + val_names
    if set(train_names) & set(val_names):
        raise ValueError("train/val episode overlap")
    if set(all_names) != set(records) or len(all_names) != len(records):
        raise ValueError("split does not conserve the dataset manifest")
    content_hashes = [item["content_hash"] for item in records.values()]
    if len(content_hashes) != len(set(content_hashes)):
        raise ValueError("duplicate episode content hashes")

    temp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temp.exists():
        raise FileExistsError(f"temporary output already exists: {temp}")
    temp.mkdir(parents=False)
    try:
        linked = 0
        for name in sorted(records):
            src = source / name
            dst = temp / name
            os.link(src, dst)
            if src.stat().st_ino != dst.stat().st_ino:
                raise RuntimeError(f"hard-link verification failed: {name}")
            linked += 1

        state_moments = Moments()
        action_moments = Moments()
        task_counts = {"train": {}, "val": {}}
        total_frames = {"train": 0, "val": 0}
        windows = {"train": 0, "val": 0}
        valid_action_steps = {"train": 0, "val": 0}
        padded_action_steps = {"train": 0, "val": 0}
        for role, names in (("train", train_names), ("val", val_names)):
            for name in names:
                record = records[name]
                task = str(record["task_id"])
                task_counts[role][task] = task_counts[role].get(task, 0) + 1
                with h5py.File(source / name, "r", libver="latest", swmr=True) as file:
                    state13 = np.asarray(file["observations/qpos"], dtype=np.float32)
                    action13 = np.asarray(file["action"], dtype=np.float32)
                if state13.shape != action13.shape or state13.ndim != 2 or state13.shape[1] != 13:
                    raise ValueError(f"invalid state/action shape: {name}")
                if not np.isfinite(state13).all() or not np.isfinite(action13).all():
                    raise ValueError(f"non-finite state/action: {name}")
                length = int(state13.shape[0])
                total_frames[role] += length
                episode_windows = length - ACTION_OFFSET_STEPS
                windows[role] += episode_windows
                # Sum the real future steps represented by every possible
                # obs[t] -> action[t+1:t+65] window. Remaining slots are pad.
                episode_valid = sum(min(ACTION_CHUNK_SIZE, length - t - 1) for t in range(episode_windows))
                valid_action_steps[role] += episode_valid
                padded_action_steps[role] += episode_windows * ACTION_CHUNK_SIZE - episode_valid
                if role == "train":
                    state_moments.update(encode_a2d_vector(state13, indices=A2D_STATE_INDICES))
                    action_moments.update(encode_a2d_vector(action13, indices=A2D_ACTION_INDICES))

        language_metadata = {
            "mode": "pending_task_conditioned_t5",
            "texts": {
                "box": "Grasp and lift the box.",
                "bottle": "Grasp and lift the bottle.",
            },
        }
        version_suffix = "rdt_v1"
        if empty_lang_embed is not None:
            if not empty_lang_embed.is_file():
                raise FileNotFoundError(empty_lang_embed)
            lang_dir = temp / "language"
            lang_dir.mkdir()
            for task in ("box", "bottle"):
                os.link(empty_lang_embed, lang_dir / f"{task}.pt")
            language_metadata.update({
                "mode": "official_empty_embedding_language_conditioning_disabled",
                "source": str(empty_lang_embed),
                "sha256": sha256_file(empty_lang_embed),
                "files": {task: f"language/{task}.pt" for task in ("box", "bottle")},
            })
            version_suffix = "rdt_v1_emptylang"

        dataset_manifest = dict(source_dataset)
        dataset_manifest["data_version"] = f"{source_dataset['data_version']}__{version_suffix}"
        dataset_manifest["derived_from"] = {
            "path": str(source),
            "dataset_manifest_sha256": sha256_file(source_dataset_path),
            "split_manifest_sha256": sha256_file(source_split_path),
        }
        write_json(temp / "dataset_manifest.json", dataset_manifest)
        split_manifest = dict(source_split)
        split_manifest["data_version"] = dataset_manifest["data_version"]
        split_manifest["dataset_manifest_sha256"] = sha256_file(temp / "dataset_manifest.json")
        split_manifest["roles"] = {
            "train": "optimization",
            "val": "checkpoint_selection_diagnostic_not_sealed_holdout",
        }
        write_json(temp / "split_manifest.json", split_manifest)

        state_stats = state_moments.result()
        action_stats = action_moments.result()
        official_stats = {
            DATASET_NAME: {
                "dataset_name": DATASET_NAME,
                "state_mean": state_stats["mean"],
                "state_std": state_stats["std"],
                "state_min": state_stats["min"],
                "state_max": state_stats["max"],
            }
        }
        write_json(temp / "rdt_dataset_stat.json", official_stats)
        write_json(temp / "rdt_action_stat.json", action_stats)
        write_json(temp / "rdt_instructions.json", language_metadata)
        rdt_manifest = {
            "schema_version": 1,
            "data_version": dataset_manifest["data_version"],
            "source_dataset_manifest_sha256": sha256_file(source_dataset_path),
            "source_split_manifest_sha256": sha256_file(source_split_path),
            "dataset_manifest_sha256": sha256_file(temp / "dataset_manifest.json"),
            "split_manifest_sha256": sha256_file(temp / "split_manifest.json"),
            "storage": "verified hard links to immutable source HDF5 payloads",
            "episodes": {"total": len(records), "train": len(train_names), "val": len(val_names)},
            "task_episode_counts": task_counts,
            "frames": total_frames,
            "windows_obs_t_to_action_t_plus_1": windows,
            "chunk_size": ACTION_CHUNK_SIZE,
            "valid_action_steps_across_windows": valid_action_steps,
            "padded_action_steps_across_windows": padded_action_steps,
            "state_action_dim": STATE_DIM,
            "active_state_indices": list(A2D_STATE_INDICES),
            "active_action_indices": list(A2D_ACTION_INDICES),
            "active_dimensions": len(A2D_ACTION_INDICES),
            "action_semantics": EXPECTED_ACTION_SEMANTICS,
            "state_stats_source": "train episodes only",
            "action_stats_source": "train episodes only",
            "language_embeddings": language_metadata,
            "hard_links_verified": linked,
            "train_val_overlap": 0,
            "duplicate_content_hashes": 0,
        }
        write_json(temp / "rdt_manifest.json", rdt_manifest)
        language_ready = empty_lang_embed is not None
        write_json(temp / "rdt_gate_report.json", {
            "status": (
                "DATA_CONTRACT_READY_EMPTY_LANGUAGE"
                if language_ready
                else "DATA_CONTRACT_READY_LANGUAGE_EMBEDDINGS_PENDING"
            ),
            "checks": {
                "source_semantics": "pass",
                "split_conservation": "pass",
                "train_val_overlap": "pass",
                "content_duplicates": "pass",
                "hard_link_identity": "pass",
                "train_only_statistics": "pass",
                "language_embeddings": "pass_empty_unconditioned" if language_ready else "pending",
                "trainer_split_and_loss_masks": "pending",
            },
        })
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    print(json.dumps({
        "status": "A2D_RDT_DATA_CONTRACT_READY",
        "output": str(output),
        "episodes": len(records),
        "train": len(train_names),
        "val": len(val_names),
        "hard_links": linked,
        "physical_payload_copy_bytes": 0,
        "rdt_manifest_sha256": sha256_file(output / "rdt_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
