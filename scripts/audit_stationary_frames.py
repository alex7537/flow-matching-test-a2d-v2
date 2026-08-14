"""Audit adjacent frames with unchanged or near-unchanged executed joint positions.

This script is read-only. It does not remove frames or modify the dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_THRESHOLDS = (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_phases(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def threshold_key(threshold: float) -> str:
    return "exact_zero" if threshold == 0.0 else f"le_{threshold:g}"


def stationary_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    return values == 0.0 if threshold == 0.0 else values <= threshold


def affected_windows(
    transition_mask: np.ndarray,
    *,
    length: int,
    horizon: int,
    offset: int,
) -> tuple[int, int]:
    max_t = length - offset - horizon
    total = max(0, max_t + 1)
    affected = np.zeros(total, dtype=bool)
    for transition_index in np.flatnonzero(transition_mask):
        low = max(0, int(transition_index) - offset - (horizon - 2))
        high = min(max_t, int(transition_index) - offset)
        if low <= high:
            affected[low : high + 1] = True
    return total, int(np.sum(affected))


def transition_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current == previous + 1:
            previous = current
            continue
        runs.append((start, previous))
        start = previous = current
    runs.append((start, previous))
    return runs


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / args.dataset_manifest
    split_path = data_dir / args.split_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    file_names = [str(item["file_name"]) for item in manifest["episodes"]]
    split_by_name = {
        **{str(name): "train" for name in split_manifest["train_episodes"]},
        **{str(name): "val" for name in split_manifest["val_episodes"]},
    }
    if set(file_names) != set(split_by_name):
        raise ValueError("dataset and split manifests contain different episode membership")

    thresholds = tuple(sorted(set(float(value) for value in args.thresholds)))
    summary_by_threshold = {
        threshold_key(threshold): {
            "threshold": threshold,
            "transitions": 0,
            "episodes": 0,
            "phase_boundary_transitions": 0,
            "same_phase_transitions": 0,
            "train_transitions": 0,
            "val_transitions": 0,
            "complete_action_windows": 0,
            "affected_action_windows": 0,
            "train_complete_action_windows": 0,
            "train_affected_action_windows": 0,
            "val_complete_action_windows": 0,
            "val_affected_action_windows": 0,
            "eligible_observation_destination_frames": 0,
        }
        for threshold in thresholds
    }
    phase_pairs: dict[str, Counter[tuple[str, str]]] = {
        threshold_key(threshold): Counter() for threshold in thresholds
    }
    exact_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    total_frames = 0
    total_transitions = 0
    action_qpos_max_abs_diff = 0.0
    segmentation_mismatches = 0

    for episode_number, file_name in enumerate(file_names, start=1):
        path = data_dir / file_name
        split = split_by_name[file_name]
        with h5py.File(path, "r") as handle:
            action = np.asarray(handle["action"][:], dtype=np.float64)
            qpos = np.asarray(handle["observations/qpos"][:], dtype=np.float64)
            phases = decode_phases(handle["phase"][:])
            segment_type = np.asarray(handle["segment_type"][:], dtype=np.uint8)
            semantics = str(handle.attrs.get("action_semantics", ""))
            motion_threshold = float(
                handle.attrs.get("segmentation_motion_threshold", np.nan)
            )
        if action.shape != qpos.shape or action.ndim != 2 or action.shape[1] != 13:
            raise ValueError(f"{path}: expected matching action/qpos shape [T,13]")
        if phases.shape != (len(action),) or segment_type.shape != (len(action),):
            raise ValueError(f"{path}: phase/segment length mismatch")
        if semantics != "executed_joint_position":
            raise ValueError(f"{path}: unexpected action semantics {semantics!r}")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{path}: non-finite joint positions")

        action_qpos_max_abs_diff = max(
            action_qpos_max_abs_diff, float(np.max(np.abs(action - qpos)))
        )
        delta = np.diff(action, axis=0)
        all_l2 = np.linalg.norm(delta, axis=1)
        arm_l2 = np.linalg.norm(delta[:, :7], axis=1)
        hand_l2 = np.linalg.norm(delta[:, 7:], axis=1)
        all_linf = np.max(np.abs(delta), axis=1)
        phase_boundary = phases[:-1] != phases[1:]
        total_frames += len(action)
        total_transitions += len(delta)

        expected_static = all_l2 <= motion_threshold
        segmentation_mismatches += int(
            np.sum((segment_type[1:] == 0) != expected_static)
        )

        for threshold in thresholds:
            key = threshold_key(threshold)
            mask = stationary_mask(all_l2, threshold)
            values = summary_by_threshold[key]
            count = int(np.sum(mask))
            values["transitions"] += count
            values["episodes"] += int(np.any(mask))
            values["phase_boundary_transitions"] += int(np.sum(mask & phase_boundary))
            values["same_phase_transitions"] += int(np.sum(mask & ~phase_boundary))
            values[f"{split}_transitions"] += count
            total_windows, affected = affected_windows(
                mask,
                length=len(action),
                horizon=args.horizon,
                offset=args.offset,
            )
            values["complete_action_windows"] += total_windows
            values["affected_action_windows"] += affected
            values[f"{split}_complete_action_windows"] += total_windows
            values[f"{split}_affected_action_windows"] += affected
            max_observation_t = len(action) - args.offset - args.horizon
            values["eligible_observation_destination_frames"] += int(
                np.sum(np.flatnonzero(mask) + 1 <= max_observation_t)
            )
            for index in np.flatnonzero(mask):
                phase_pairs[key][(str(phases[index]), str(phases[index + 1]))] += 1

        candidate_mask = stationary_mask(all_l2, args.candidate_threshold)
        exact_mask = all_l2 == 0.0
        for index in np.flatnonzero(candidate_mask):
            row = {
                "episode": file_name,
                "split": split,
                "from_frame": int(index),
                "to_frame": int(index + 1),
                "phase_before": str(phases[index]),
                "phase_after": str(phases[index + 1]),
                "phase_boundary": bool(phase_boundary[index]),
                "exact_all_13_equal": bool(exact_mask[index]),
                "all_l2": float(all_l2[index]),
                "all_linf": float(all_linf[index]),
                "arm_l2": float(arm_l2[index]),
                "hand_l2": float(hand_l2[index]),
                "destination_is_complete_window_observation": bool(
                    index + 1 <= len(action) - args.offset - args.horizon
                ),
            }
            candidate_rows.append(row)
            if exact_mask[index]:
                exact_rows.append(row)

        for start, end in transition_runs(candidate_mask):
            run_rows.append(
                {
                    "episode": file_name,
                    "split": split,
                    "start_transition": start,
                    "end_transition": end,
                    "start_frame": start,
                    "end_frame": end + 1,
                    "stationary_transitions": end - start + 1,
                    "frames_in_run": end - start + 2,
                    "phase_start": str(phases[start]),
                    "phase_end": str(phases[end + 1]),
                    "all_exact_zero": bool(np.all(exact_mask[start : end + 1])),
                    "max_all_l2": float(np.max(all_l2[start : end + 1])),
                }
            )

        if episode_number % 100 == 0 or episode_number == len(file_names):
            print(f"audited {episode_number}/{len(file_names)} episodes")

    for key, values in summary_by_threshold.items():
        values["transition_fraction"] = values["transitions"] / total_transitions
        values["affected_action_window_fraction"] = (
            values["affected_action_windows"] / values["complete_action_windows"]
        )
        values["train_affected_action_window_fraction"] = (
            values["train_affected_action_windows"]
            / values["train_complete_action_windows"]
        )
        values["val_affected_action_window_fraction"] = (
            values["val_affected_action_windows"]
            / values["val_complete_action_windows"]
        )

    phase_summary = {
        key: [
            {
                "phase_before": before,
                "phase_after": after,
                "transitions": count,
            }
            for (before, after), count in counter.most_common()
        ]
        for key, counter in phase_pairs.items()
    }
    run_lengths = Counter(row["stationary_transitions"] for row in run_rows)
    report = {
        "schema_version": 1,
        "audit_contract": {
            "data_dir": str(data_dir),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "split_manifest_sha256": sha256_file(split_path),
            "action_semantics": "executed_joint_position",
            "stationary_definition": "L2(action[t+1] - action[t]) <= threshold",
            "exact_definition": "all 13 float values are exactly equal",
            "thresholds": list(thresholds),
            "candidate_threshold": args.candidate_threshold,
            "action_window": {
                "offset": args.offset,
                "horizon": args.horizon,
                "complete_windows_only": True,
            },
            "deletion_performed": False,
        },
        "dataset": {
            "data_version": manifest.get("data_version"),
            "episodes": len(file_names),
            "train_episodes": len(split_manifest["train_episodes"]),
            "val_episodes": len(split_manifest["val_episodes"]),
            "frames": total_frames,
            "transitions": total_transitions,
        },
        "integrity": {
            "action_qpos_max_abs_diff": action_qpos_max_abs_diff,
            "segmentation_static_mask_mismatches": segmentation_mismatches,
        },
        "threshold_summary": summary_by_threshold,
        "phase_pair_summary": phase_summary,
        "candidate_runs": {
            "runs": len(run_rows),
            "run_length_transition_counts": {
                str(length): count for length, count in sorted(run_lengths.items())
            },
            "max_stationary_transitions": max(run_lengths, default=0),
        },
        "candidate_transition_count": len(candidate_rows),
        "exact_transition_count": len(exact_rows),
    }
    return report, candidate_rows, run_rows


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    lines = [
        "# Stationary-frame audit",
        "",
        "Read-only audit of adjacent executed joint positions in data-rgb-450gb-v1.",
        "",
        f"- Episodes: {dataset['episodes']} "
        f"(train {dataset['train_episodes']}, val {dataset['val_episodes']})",
        f"- Frames: {dataset['frames']}",
        f"- Adjacent transitions: {dataset['transitions']}",
        "",
        "## Threshold summary",
        "",
        "| criterion | transitions | episodes | fraction | affected H=16 windows | "
        "eligible destination obs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, item in report["threshold_summary"].items():
        lines.append(
            f"| {key} | {item['transitions']} | {item['episodes']} | "
            f"{item['transition_fraction']:.4%} | "
            f"{item['affected_action_windows']} "
            f"({item['affected_action_window_fraction']:.4%}) | "
            f"{item['eligible_observation_destination_frames']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Max `abs(action - qpos)`: "
            f"{report['integrity']['action_qpos_max_abs_diff']:.9g}",
            f"- Stored segmentation/static-mask mismatches: "
            f"{report['integrity']['segmentation_static_mask_mismatches']}",
            "",
            "## Important boundary",
            "",
            "- A stationary joint transition does not prove that RGB or the object is unchanged.",
            "- Removing a destination frame changes implicit time and action-window indexing.",
            "- No frame was removed by this audit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", default="dataset_manifest.json")
    parser.add_argument("--split-manifest", default="split_manifest.json")
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--candidate-threshold", type=float, default=1.0e-4)
    parser.add_argument("--offset", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()
    if args.candidate_threshold < 0 or any(value < 0 for value in args.thresholds):
        parser.error("thresholds must be non-negative")
    if args.offset < 0 or args.horizon < 2:
        parser.error("offset must be >= 0 and horizon must be >= 2")
    return args


def main() -> None:
    args = parse_args()
    report, candidates, runs = audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode",
        "split",
        "from_frame",
        "to_frame",
        "phase_before",
        "phase_after",
        "phase_boundary",
        "exact_all_13_equal",
        "all_l2",
        "all_linf",
        "arm_l2",
        "hand_l2",
        "destination_is_complete_window_observation",
    ]
    exact = [row for row in candidates if row["exact_all_13_equal"]]
    (args.output_dir / "stationary_frame_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "near_stationary_transitions.csv", candidates, fields)
    write_csv(args.output_dir / "exact_stationary_transitions.csv", exact, fields)
    write_csv(
        args.output_dir / "stationary_runs.csv",
        runs,
        [
            "episode",
            "split",
            "start_transition",
            "end_transition",
            "start_frame",
            "end_frame",
            "stationary_transitions",
            "frames_in_run",
            "phase_start",
            "phase_end",
            "all_exact_zero",
            "max_all_l2",
        ],
    )
    write_markdown(args.output_dir / "README.md", report)
    print(f"wrote {args.output_dir} ({len(exact)} exact, {len(candidates)} near-static)")


if __name__ == "__main__":
    main()
