"""Audit arm/hand action continuity in processed A2D episodes.

The audit is read-only. It reports adjacent-frame joint deltas and quantifies
how many complete action windows would cross candidate discontinuities.

Example:
    python scripts/audit_action_continuity.py \
        --data-dir data-rgb-450gb-v1 \
        --output-dir output/audits/action_continuity_full
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_THRESHOLDS = (0.1, 0.25, 0.5, 1.0)
PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.5, 99.9, 100.0)


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


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    quantiles = np.percentile(values, PERCENTILES)
    result: dict[str, float | int] = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
    }
    for percentile, value in zip(PERCENTILES, quantiles, strict=True):
        label = f"p{percentile:g}".replace(".", "_")
        result[label] = float(value)
    return result


def threshold_counts(
    values: np.ndarray,
    boundaries: np.ndarray,
    episode_slices: list[slice],
    thresholds: tuple[float, ...],
) -> dict[str, dict[str, int | float]]:
    result = {}
    for threshold in thresholds:
        selected = values > threshold
        episode_count = sum(bool(np.any(selected[item])) for item in episode_slices)
        result[f"{threshold:g}"] = {
            "transitions": int(np.sum(selected)),
            "transition_fraction": float(np.mean(selected)) if selected.size else 0.0,
            "episodes": int(episode_count),
            "phase_boundary_transitions": int(np.sum(selected & boundaries)),
            "same_phase_transitions": int(np.sum(selected & ~boundaries)),
        }
    return result


def affected_windows(
    transition_mask: np.ndarray,
    *,
    length: int,
    horizon: int,
    offset: int,
) -> tuple[int, int]:
    """Return complete-window count and count crossing a selected transition.

    A sample at observation t has action indices [t+offset, t+offset+horizon).
    Transition i means action[i] -> action[i+1]. The window crosses it iff
    t+offset <= i <= t+offset+horizon-2.
    """
    max_t = length - offset - horizon
    total = max(0, max_t + 1)
    if total == 0:
        return 0, 0
    affected = np.zeros(total, dtype=bool)
    for transition_index in np.flatnonzero(transition_mask):
        low = max(0, int(transition_index) - offset - (horizon - 2))
        high = min(max_t, int(transition_index) - offset)
        if low <= high:
            affected[low : high + 1] = True
    return total, int(np.sum(affected))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / args.dataset_manifest
    split_path = data_dir / args.split_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))

    manifest_files = [item["file_name"] for item in manifest["episodes"]]
    train_names = set(split_manifest["train_episodes"])
    val_names = set(split_manifest["val_episodes"])
    split_by_name = {name: "train" for name in train_names}
    split_by_name.update({name: "val" for name in val_names})
    if set(manifest_files) != set(split_by_name):
        missing = sorted(set(manifest_files) - set(split_by_name))
        extra = sorted(set(split_by_name) - set(manifest_files))
        raise ValueError(f"split/manifest mismatch: missing={missing[:5]}, extra={extra[:5]}")

    all_arm: list[np.ndarray] = []
    all_hand: list[np.ndarray] = []
    all_boundary: list[np.ndarray] = []
    episode_slices: list[slice] = []
    per_joint_abs: list[np.ndarray] = []
    top_events: list[dict[str, Any]] = []
    candidate_events: list[dict[str, Any]] = []
    phase_pair_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"arm": [], "hand": []}
    )
    window_criteria = ["all_phase_boundaries", "thumb_to_pregrasp"]
    for part in ("arm", "hand"):
        for threshold in args.thresholds:
            window_criteria.extend(
                [
                    f"{part}_gt_{threshold:g}",
                    f"{part}_gt_{threshold:g}_phase_boundary",
                    f"{part}_gt_{threshold:g}_same_phase",
                ]
            )
    windows = {
        split: {
            "complete_windows": 0,
            **{criterion: 0 for criterion in window_criteria},
        }
        for split in ("train", "val", "all")
    }
    episode_summary: list[dict[str, Any]] = []
    transition_cursor = 0
    total_frames = 0
    source_episodes_available = 0

    for episode_number, file_name in enumerate(manifest_files, start=1):
        path = data_dir / file_name
        split = split_by_name[file_name]
        with h5py.File(path, "r") as handle:
            action = np.asarray(handle["action"][:], dtype=np.float64)
            phases = decode_phases(handle["phase"][:])
            action_semantics = str(handle.attrs.get("action_semantics", ""))
            source_path = Path(str(handle.attrs.get("source_file", "")))
        if action.ndim != 2 or action.shape[1] != 13:
            raise ValueError(f"{path}: expected action [T,13], got {action.shape}")
        if phases.shape != (action.shape[0],):
            raise ValueError(f"{path}: phase/action length mismatch")
        if action_semantics != "executed_joint_position":
            raise ValueError(f"{path}: unexpected action_semantics={action_semantics!r}")
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{path}: action contains non-finite values")

        target_action = None
        if source_path.is_file():
            with h5py.File(source_path, "r") as source:
                trajectory = source["trajectory"]
                target_action = np.concatenate(
                    [
                        np.asarray(trajectory["arm2_pos_target"][:], dtype=np.float64),
                        np.asarray(trajectory["hand2_pos_target"][:], dtype=np.float64),
                    ],
                    axis=1,
                )
            if target_action.shape != action.shape:
                raise ValueError(
                    f"{source_path}: target/action shape mismatch "
                    f"{target_action.shape} != {action.shape}"
                )
            source_episodes_available += 1

        delta = np.diff(action, axis=0)
        arm = np.linalg.norm(delta[:, :7], axis=1)
        hand = np.linalg.norm(delta[:, 7:], axis=1)
        boundary = phases[:-1] != phases[1:]
        thumb_to_pregrasp = (phases[:-1] == "thumb") & (phases[1:] == "pregrasp")
        all_arm.append(arm)
        all_hand.append(hand)
        all_boundary.append(boundary)
        per_joint_abs.append(np.abs(delta))
        episode_slices.append(slice(transition_cursor, transition_cursor + len(delta)))
        transition_cursor += len(delta)
        total_frames += len(action)

        masks: dict[str, np.ndarray] = {
            "all_phase_boundaries": boundary,
            "thumb_to_pregrasp": thumb_to_pregrasp,
        }
        for part, values in (("arm", arm), ("hand", hand)):
            for threshold in args.thresholds:
                large = values > threshold
                masks[f"{part}_gt_{threshold:g}"] = large
                masks[f"{part}_gt_{threshold:g}_phase_boundary"] = large & boundary
                masks[f"{part}_gt_{threshold:g}_same_phase"] = large & ~boundary

        complete_windows = None
        for criterion, mask in masks.items():
            total, affected = affected_windows(
                mask,
                length=len(action),
                horizon=args.horizon,
                offset=args.offset,
            )
            if complete_windows is None:
                complete_windows = total
            elif complete_windows != total:
                raise AssertionError("window totals changed within an episode")
            windows[split][criterion] += affected
            windows["all"][criterion] += affected
        windows[split]["complete_windows"] += int(complete_windows or 0)
        windows["all"]["complete_windows"] += int(complete_windows or 0)

        episode_summary.append(
            {
                "episode": file_name,
                "split": split,
                "frames": int(len(action)),
                "transitions": int(len(delta)),
                "phase_boundaries": int(np.sum(boundary)),
                "arm_max_l2": float(np.max(arm)) if arm.size else 0.0,
                "hand_max_l2": float(np.max(hand)) if hand.size else 0.0,
                "arm_gt_0_25": int(np.sum(arm > 0.25)),
                "hand_gt_0_25": int(np.sum(hand > 0.25)),
                "hand_gt_0_5": int(np.sum(hand > 0.5)),
                "hand_gt_1": int(np.sum(hand > 1.0)),
            }
        )

        for index in range(len(delta)):
            before_phase = str(phases[index])
            after_phase = str(phases[index + 1])
            phase_pair = phase_pair_values[(before_phase, after_phase)]
            phase_pair["arm"].append(float(arm[index]))
            phase_pair["hand"].append(float(hand[index]))
            event = {
                "episode": file_name,
                "split": split,
                "transition_index": int(index),
                "from_frame": int(index),
                "to_frame": int(index + 1),
                "phase_before": before_phase,
                "phase_after": after_phase,
                "phase_boundary": bool(boundary[index]),
                "arm_l2": float(arm[index]),
                "hand_l2": float(hand[index]),
                "arm_delta": delta[index, :7].tolist(),
                "hand_delta": delta[index, 7:].tolist(),
                "hand_before": action[index, 7:].tolist(),
                "hand_after": action[index + 1, 7:].tolist(),
            }
            if target_action is not None:
                target_delta = target_action[index + 1] - target_action[index]
                event.update(
                    {
                        "target_arm_l2": float(np.linalg.norm(target_delta[:7])),
                        "target_hand_l2": float(np.linalg.norm(target_delta[7:])),
                        "arm_tracking_error_before": float(
                            np.linalg.norm(action[index, :7] - target_action[index, :7])
                        ),
                        "arm_tracking_error_after": float(
                            np.linalg.norm(
                                action[index + 1, :7] - target_action[index + 1, :7]
                            )
                        ),
                        "hand_tracking_error_before": float(
                            np.linalg.norm(action[index, 7:] - target_action[index, 7:])
                        ),
                        "hand_tracking_error_after": float(
                            np.linalg.norm(
                                action[index + 1, 7:] - target_action[index + 1, 7:]
                            )
                        ),
                    }
                )
            top_events.append(event)
            if arm[index] > args.candidate_threshold or hand[index] > args.candidate_threshold:
                candidate_events.append(event)

        if episode_number % 100 == 0 or episode_number == len(manifest_files):
            print(f"audited {episode_number}/{len(manifest_files)} episodes")

    arm_values = np.concatenate(all_arm)
    hand_values = np.concatenate(all_hand)
    boundaries = np.concatenate(all_boundary)
    joint_values = np.concatenate(per_joint_abs, axis=0)

    phase_pairs = []
    for (before, after), values in phase_pair_values.items():
        arm = np.asarray(values["arm"])
        hand = np.asarray(values["hand"])
        phase_pairs.append(
            {
                "phase_before": before,
                "phase_after": after,
                "is_boundary": before != after,
                "transitions": int(len(arm)),
                "arm": distribution(arm),
                "hand": distribution(hand),
                "arm_gt_0_25": int(np.sum(arm > 0.25)),
                "hand_gt_0_25": int(np.sum(hand > 0.25)),
                "hand_gt_0_5": int(np.sum(hand > 0.5)),
                "hand_gt_1": int(np.sum(hand > 1.0)),
            }
        )
    phase_pairs.sort(
        key=lambda item: (
            -float(item["hand"].get("p100", 0.0)),
            -int(item["transitions"]),
            item["phase_before"],
            item["phase_after"],
        )
    )

    for split_values in windows.values():
        total = split_values["complete_windows"]
        split_values["affected_fraction"] = {
            criterion: (float(count / total) if total else 0.0)
            for criterion, count in split_values.items()
            if criterion != "complete_windows"
        }

    top_hand = sorted(top_events, key=lambda item: item["hand_l2"], reverse=True)[
        : args.top_events
    ]
    top_arm = sorted(top_events, key=lambda item: item["arm_l2"], reverse=True)[
        : args.top_events
    ]
    hand_candidates = [
        item for item in candidate_events if item["hand_l2"] > args.candidate_threshold
    ]
    target_diagnostic = {
        "source_episodes_available": source_episodes_available,
        "source_episodes_missing": len(manifest_files) - source_episodes_available,
        "hand_candidate_transitions": len(hand_candidates),
        "hand_candidates_with_target": sum(
            "target_hand_l2" in item for item in hand_candidates
        ),
        "steady_target_le_0_1": sum(
            item.get("target_hand_l2", float("inf")) <= 0.1
            for item in hand_candidates
        ),
        "target_jump_gt_0_25": sum(
            item.get("target_hand_l2", 0.0) > 0.25 for item in hand_candidates
        ),
        "tracking_error_reduced": sum(
            item.get("hand_tracking_error_after", float("inf"))
            < item.get("hand_tracking_error_before", float("-inf"))
            for item in hand_candidates
        ),
        "steady_target_and_tracking_error_reduced": sum(
            item.get("target_hand_l2", float("inf")) <= 0.1
            and item.get("hand_tracking_error_after", float("inf"))
            < item.get("hand_tracking_error_before", float("-inf"))
            for item in hand_candidates
        ),
    }
    joint_names = [f"arm_{index}" for index in range(7)] + [
        f"hand_{index}" for index in range(6)
    ]
    report = {
        "schema_version": 1,
        "audit_contract": {
            "data_dir": str(data_dir),
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "split_manifest": str(split_path),
            "split_manifest_sha256": sha256_file(split_path),
            "action_semantics": "executed_joint_position",
            "arm_dimensions": [0, 7],
            "hand_dimensions": [7, 13],
            "delta_definition": "L2(action[t+1] - action[t])",
            "threshold_operator": ">",
            "thresholds_rad": list(args.thresholds),
            "candidate_event_threshold_rad": args.candidate_threshold,
            "action_window": {
                "offset": args.offset,
                "horizon": args.horizon,
                "indices": "[t+offset, t+offset+horizon)",
                "crosses_transition_rule": (
                    "t+offset <= transition_index <= t+offset+horizon-2"
                ),
                "complete_windows_only": True,
            },
        },
        "dataset": {
            "data_version": manifest.get("data_version"),
            "episodes": len(manifest_files),
            "train_episodes": len(train_names),
            "val_episodes": len(val_names),
            "frames": total_frames,
            "transitions": int(len(arm_values)),
            "phase_boundaries": int(np.sum(boundaries)),
            "same_phase_transitions": int(np.sum(~boundaries)),
        },
        "global": {
            "arm": {
                "distribution": distribution(arm_values),
                "thresholds": threshold_counts(
                    arm_values, boundaries, episode_slices, args.thresholds
                ),
                "phase_boundary_distribution": distribution(arm_values[boundaries]),
                "same_phase_distribution": distribution(arm_values[~boundaries]),
            },
            "hand": {
                "distribution": distribution(hand_values),
                "thresholds": threshold_counts(
                    hand_values, boundaries, episode_slices, args.thresholds
                ),
                "phase_boundary_distribution": distribution(hand_values[boundaries]),
                "same_phase_distribution": distribution(hand_values[~boundaries]),
            },
        },
        "per_joint_absolute_delta": {
            name: distribution(joint_values[:, index])
            for index, name in enumerate(joint_names)
        },
        "window_impact": windows,
        "target_diagnostic_for_hand_candidates": target_diagnostic,
        "phase_pairs": phase_pairs,
        "episode_summary": episode_summary,
        "top_hand_events": top_hand,
        "top_arm_events": top_arm,
        "candidate_event_count": len(candidate_events),
    }
    return json_ready(report), json_ready(candidate_events)


def write_candidates(path: Path, events: list[dict[str, Any]]) -> None:
    fields = [
        "episode",
        "split",
        "transition_index",
        "from_frame",
        "to_frame",
        "phase_before",
        "phase_after",
        "phase_boundary",
        "arm_l2",
        "hand_l2",
        "arm_delta",
        "hand_delta",
        "hand_before",
        "hand_after",
        "target_arm_l2",
        "target_hand_l2",
        "arm_tracking_error_before",
        "arm_tracking_error_after",
        "hand_tracking_error_before",
        "hand_tracking_error_after",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in sorted(
            events,
            key=lambda item: (-max(item["arm_l2"], item["hand_l2"]), item["episode"]),
        ):
            row = dict(event)
            for key in ("arm_delta", "hand_delta", "hand_before", "hand_after"):
                row[key] = json.dumps(row[key], separators=(",", ":"))
            writer.writerow(row)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    arm = report["global"]["arm"]
    hand = report["global"]["hand"]
    windows = report["window_impact"]
    lines = [
        "# Full action-continuity audit",
        "",
        "This is a read-only audit of adjacent executed joint positions.",
        "",
        "## Dataset",
        "",
        f"- Episodes: {dataset['episodes']} "
        f"(train {dataset['train_episodes']}, val {dataset['val_episodes']})",
        f"- Frames: {dataset['frames']}",
        f"- Adjacent transitions: {dataset['transitions']}",
        f"- Phase boundaries: {dataset['phase_boundaries']}",
        "",
        "## Global L2 delta",
        "",
        "| part | p50 | p95 | p99 | p99.9 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| arm | {arm['distribution']['p50']:.6f} | "
            f"{arm['distribution']['p95']:.6f} | {arm['distribution']['p99']:.6f} | "
            f"{arm['distribution']['p99_9']:.6f} | {arm['distribution']['p100']:.6f} |"
        ),
        (
            f"| hand | {hand['distribution']['p50']:.6f} | "
            f"{hand['distribution']['p95']:.6f} | {hand['distribution']['p99']:.6f} | "
            f"{hand['distribution']['p99_9']:.6f} | {hand['distribution']['p100']:.6f} |"
        ),
        "",
        "## Threshold counts",
        "",
        "| part | threshold | transitions | episodes | phase boundary | same phase |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for part, values in (("arm", arm), ("hand", hand)):
        for threshold, item in values["thresholds"].items():
            lines.append(
                f"| {part} | >{threshold} | {item['transitions']} | "
                f"{item['episodes']} | {item['phase_boundary_transitions']} | "
                f"{item['same_phase_transitions']} |"
            )
    lines.extend(
        [
            "",
            "## Complete-window impact",
            "",
            (
                "Window contract: offset="
                f"{report['audit_contract']['action_window']['offset']}, "
                f"horizon={report['audit_contract']['action_window']['horizon']}."
            ),
            "",
            "| split | windows | criterion | affected | fraction |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    selected_criteria = [
        "all_phase_boundaries",
        "thumb_to_pregrasp",
        "arm_gt_0.25",
        "hand_gt_0.25",
        "hand_gt_0.5",
        "hand_gt_0.5_phase_boundary",
        "hand_gt_0.5_same_phase",
        "hand_gt_1",
    ]
    for split in ("train", "val", "all"):
        total = windows[split]["complete_windows"]
        for criterion in selected_criteria:
            affected = windows[split][criterion]
            fraction = windows[split]["affected_fraction"][criterion]
            lines.append(
                f"| {split} | {total} | {criterion} | {affected} | {fraction:.4%} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The 0.1-rad segmentation keyframe threshold is descriptive, not an anomaly label.",
            "- Phase-boundary and same-phase events are reported separately.",
            "- This audit does not smooth, interpolate, delete, or mask any training data.",
            "- Candidate events require semantic or visual review before becoming an invalid-transition mask.",
            "",
            "## Target diagnostic for hand candidates",
            "",
        ]
    )
    target = report["target_diagnostic_for_hand_candidates"]
    lines.extend(
        [
            f"- Raw source episodes available: {target['source_episodes_available']}",
            f"- Hand candidates: {target['hand_candidate_transitions']}",
            f"- Target nearly steady (target L2 <= 0.1): {target['steady_target_le_0_1']}",
            f"- Target itself jumps (> 0.25): {target['target_jump_gt_0_25']}",
            f"- Actual tracking error decreases: {target['tracking_error_reduced']}",
            (
                "- Target steady and tracking error decreases: "
                f"{target['steady_target_and_tracking_error_reduced']}"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", default="dataset_manifest.json")
    parser.add_argument("--split-manifest", default="split_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--offset", type=int, default=1)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--candidate-threshold", type=float, default=0.25)
    parser.add_argument("--top-events", type=int, default=100)
    args = parser.parse_args()
    args.thresholds = tuple(sorted(set(args.thresholds)))
    if args.horizon < 2:
        parser.error("--horizon must be >= 2 to audit crossed transitions")
    if args.offset < 0:
        parser.error("--offset must be >= 0")
    if any(value < 0 for value in args.thresholds):
        parser.error("--thresholds must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    report, candidates = audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "action_continuity_audit.json"
    candidates_path = args.output_dir / "candidate_transitions.csv"
    markdown_path = args.output_dir / "README.md"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_candidates(candidates_path, candidates)
    write_markdown(markdown_path, report)
    print(f"wrote {report_path}")
    print(f"wrote {candidates_path} ({len(candidates)} candidates)")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
