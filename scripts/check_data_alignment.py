"""Audit structural and timestamp alignment for A2D HDF5/RGB episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


TRAJECTORY_KEYS = (
    "arm2_pos",
    "hand2_pos",
    "arm2_pos_target",
    "hand2_pos_target",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TIMESTAMP_CANDIDATES = (
    "timestamps",
    "timestamp",
    "time_ns",
    "stamp_ns",
    "camera_timestamps",
    "joint_timestamps",
)


def quaternion_wxyz_to_yaw(quaternion: np.ndarray) -> float:
    qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float64)
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def summarize_pose_coverage(poses: np.ndarray, grasp_arm: np.ndarray) -> dict:
    xyz = poses[:, :3]
    yaw = np.asarray([quaternion_wxyz_to_yaw(pose[3:]) for pose in poses])
    count = len(poses)
    xy = xyz[:, :2]
    pairwise_xy = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(pairwise_xy, np.inf)
    nearest_xy = pairwise_xy.min(axis=1) if count > 1 else np.asarray([np.nan])

    x_edges = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 6)
    y_edges = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 6)
    grid, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edges, y_edges))

    yaw_diff = np.abs(np.angle(np.exp(1j * (yaw[:, None] - yaw[None, :]))))
    action_diff = np.linalg.norm(grasp_arm[:, None, :] - grasp_arm[None, :, :], axis=-1)
    near_mask = (pairwise_xy < 0.01) & (yaw_diff < np.deg2rad(5.0))
    upper = np.triu(np.ones_like(near_mask, dtype=bool), 1)
    near_action = action_diff[near_mask & upper]

    return {
        "episodes": count,
        "xyz_min": xyz.min(axis=0).tolist(),
        "xyz_max": xyz.max(axis=0).tolist(),
        "xyz_range": np.ptp(xyz, axis=0).tolist(),
        "xy_p05": np.percentile(xy, 5, axis=0).tolist(),
        "xy_p95": np.percentile(xy, 95, axis=0).tolist(),
        "yaw_deg_min": float(np.rad2deg(yaw).min()),
        "yaw_deg_max": float(np.rad2deg(yaw).max()),
        "yaw_deg_circular_mean": float(np.rad2deg(np.angle(np.mean(np.exp(1j * yaw))))),
        "nearest_xy_distance_m": {
            "p50": float(np.nanpercentile(nearest_xy, 50)),
            "p95": float(np.nanpercentile(nearest_xy, 95)),
            "max": float(np.nanmax(nearest_xy)),
        },
        "grid_5x5": {
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "counts": grid.astype(int).tolist(),
            "occupied_cells": int(np.count_nonzero(grid)),
            "min_nonempty_count": int(grid[grid > 0].min()) if np.any(grid > 0) else 0,
            "median_nonempty_count": float(np.median(grid[grid > 0])) if np.any(grid > 0) else 0.0,
            "max_count": int(grid.max()),
        },
        "near_pose_grasp_arm_difference": {
            "definition": "xy<0.01m and yaw<5deg",
            "pair_count": int(near_action.size),
            "p50_l2_rad": float(np.percentile(near_action, 50)) if near_action.size else None,
            "p95_l2_rad": float(np.percentile(near_action, 95)) if near_action.size else None,
            "max_l2_rad": float(near_action.max()) if near_action.size else None,
        },
    }


def find_timestamp_fields(group: h5py.Group) -> list[str]:
    fields: list[str] = []

    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        leaf = name.rsplit("/", 1)[-1].lower()
        if leaf in TIMESTAMP_CANDIDATES or "timestamp" in leaf:
            fields.append("/" + name)

    group.visititems(visit)
    return fields


def target_activity(values: np.ndarray) -> dict:
    zero_mask = np.all(values == 0, axis=1)
    nonzero = np.flatnonzero(~zero_mask)
    gaps = np.diff(nonzero) if nonzero.size > 1 else np.asarray([], dtype=np.int64)
    steps = np.linalg.norm(np.diff(values[nonzero], axis=0), axis=1) if nonzero.size > 1 else np.asarray([])
    return {
        "frames": int(values.shape[0]),
        "zero_frames": int(zero_mask.sum()),
        "zero_ratio": float(zero_mask.mean()),
        "nonzero_frames": int(nonzero.size),
        "nonzero_indices": nonzero.tolist(),
        "activity_timeline": "".join("#" if active else "." for active in ~zero_mask),
        "nonzero_gap_max": int(gaps.max()) if gaps.size else 0,
        "nonzero_step_l2_median": float(np.median(steps)) if steps.size else 0.0,
        "nonzero_step_l2_max": float(steps.max()) if steps.size else 0.0,
    }


def held_target_consistency(actual: np.ndarray, target: np.ndarray) -> dict:
    zero_mask = np.all(target == 0, axis=1)
    segments: list[dict] = []
    index = 0
    while index < len(zero_mask):
        if not zero_mask[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(zero_mask) and zero_mask[index + 1]:
            index += 1
        end = index
        previous = start - 1
        if previous >= 0 and not zero_mask[previous]:
            values = actual[start:end + 1]
            error = values - target[previous]
            l2_error = np.linalg.norm(error, axis=1)
            segments.append({
                "zero_segment_inclusive": [start, end],
                "held_target_frame": previous,
                "max_l2_error": float(l2_error.max()),
                "mean_l2_error": float(l2_error.mean()),
                "actual_total_path_l2": float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum()),
                "actual_start_end_l2": float(np.linalg.norm(values[-1] - values[0])),
                "per_joint_max_abs_error": np.abs(error).max(axis=0).tolist(),
                "per_joint_actual_range": np.ptp(values, axis=0).tolist(),
                "consistent_with_hold": bool(l2_error.max() < 1e-3),
            })
        index += 1
    return {
        "segments": segments,
        "all_segments_consistent_with_hold": bool(segments) and all(
            segment["consistent_with_hold"] for segment in segments
        ),
        "hold_error_threshold_l2": 1e-3,
    }


def audit_episode(
    path: Path,
    *,
    image_keys: tuple[str, ...],
    check_images: bool = True,
) -> dict:
    report: dict = {"episode": path.name, "lengths": {}, "errors": []}
    with h5py.File(path, "r") as file:
        trajectory = file.get("trajectory")
        if trajectory is None:
            report["errors"].append("missing /trajectory")
            return report

        for key in TRAJECTORY_KEYS:
            if key not in trajectory:
                report["errors"].append(f"missing /trajectory/{key}")
            else:
                report["lengths"][key] = int(trajectory[key].shape[0])

        expected = report["lengths"].get("arm2_pos")
        if check_images:
            cameras = trajectory.get("cameras")
            for key in image_keys:
                if cameras is None or key not in cameras:
                    report["errors"].append(f"missing /trajectory/cameras/{key}")
                else:
                    report["lengths"][key] = int(cameras[key].shape[0])

        timestamp_fields = find_timestamp_fields(file)
        if "arm2_pos_target" in trajectory:
            arm_target = np.asarray(trajectory["arm2_pos_target"][:])
            report["arm2_target_activity"] = target_activity(arm_target)
            zero_mask = np.all(arm_target == 0, axis=1)
            zero_indices = np.flatnonzero(zero_mask)
            segments: list[list[int]] = []
            if zero_indices.size:
                start = previous = int(zero_indices[0])
                for value in zero_indices[1:]:
                    current = int(value)
                    if current != previous + 1:
                        segments.append([start, previous])
                        start = current
                    previous = current
                segments.append([start, previous])
            phase_counts: dict[str, int] = {}
            if "phase" in trajectory:
                phases = trajectory["phase"].asstr()[:]
                for phase in phases[zero_mask]:
                    phase_counts[str(phase)] = phase_counts.get(str(phase), 0) + 1
            report["arm2_zero_target"] = {
                "frames": int(zero_mask.sum()),
                "ratio": float(zero_mask.mean()),
                "segments_inclusive": segments,
                "phase_counts": phase_counts,
                "requires_semantic_decision": bool(zero_mask.any()),
            }
            if "arm2_pos" in trajectory:
                arm_actual = np.asarray(trajectory["arm2_pos"][:])
                report["arm2_held_target_consistency"] = held_target_consistency(arm_actual, arm_target)
                first_end = min(4, len(arm_actual))
                report["arm2_initial_motion"] = {
                    "frames_inclusive": [0, first_end - 1],
                    "step_l2": np.linalg.norm(np.diff(arm_actual[:first_end], axis=0), axis=1).tolist(),
                    "total_path_l2": float(
                        np.linalg.norm(np.diff(arm_actual[:first_end], axis=0), axis=1).sum()
                    ),
                    "start_end_l2": float(np.linalg.norm(arm_actual[first_end - 1] - arm_actual[0])),
                }
                if "arm2_eef_pose_world" in trajectory:
                    eef = np.asarray(trajectory["arm2_eef_pose_world"][:first_end, :3])
                    report["arm2_initial_motion"]["eef_translation_step_m"] = np.linalg.norm(
                        np.diff(eef, axis=0), axis=1
                    ).tolist()
        if "hand2_pos_target" in trajectory:
            report["hand2_target_activity"] = target_activity(
                np.asarray(trajectory["hand2_pos_target"][:])
            )
        executed_activity = {}
        for key in ("arm2_pos", "hand2_pos"):
            if key not in trajectory:
                continue
            values = np.asarray(trajectory[key][:], dtype=np.float64)
            speed = np.linalg.norm(np.diff(values, axis=0), axis=1)
            executed_activity[key] = {
                "dynamic_indices": (np.flatnonzero(speed > 1e-4) + 1).tolist(),
                "dynamic_ratio": float(np.mean(speed > 1e-4)) if len(speed) else 0.0,
                "max_step_l2": float(speed.max()) if len(speed) else 0.0,
            }
        report["executed_activity"] = executed_activity
        report["structure_signature"] = {
            key: value["dynamic_indices"] for key, value in executed_activity.items()
        }

        before_key = "grasp/object_pose_before_execute"
        after_key = "grasp/object_pose_after_execute"
        if before_key in file and after_key in file:
            before = np.asarray(file[before_key][:3], dtype=np.float64)
            after = np.asarray(file[after_key][:3], dtype=np.float64)
            object_delta = after - before
            tail_start = max(0, int(expected or 0) - 10)
            if "phase" in trajectory:
                phases = trajectory["phase"].asstr()[:]
                lift_frames = [index for index, phase in enumerate(phases) if "lift" in phase.lower()]
                if lift_frames:
                    tail_start = lift_frames[0]
            eef_delta = None
            if "arm2_eef_pose_world" in trajectory and expected:
                eef = np.asarray(trajectory["arm2_eef_pose_world"][:, :3], dtype=np.float64)
                eef_delta = eef[-1] - eef[tail_start]
            truncated = bool(
                object_delta[2] > 0.01
                and eef_delta is not None
                and abs(float(eef_delta[2])) < 0.005
            )
            report["recording_completeness"] = {
                "tail_start": int(tail_start),
                "object_delta_xyz": object_delta.tolist(),
                "eef_tail_delta_xyz": eef_delta.tolist() if eef_delta is not None else None,
                "truncated_recording": truncated,
            }

    lengths = list(report["lengths"].values())
    report["structural_alignment"] = bool(
        expected is not None and lengths and len(set(lengths)) == 1 and not report["errors"]
    )
    report["timestamp_fields"] = timestamp_fields
    report["timestamp_alignment"] = "not_verifiable" if not timestamp_fields else "requires_residual_check"
    if not timestamp_fields:
        report["warnings"] = [
            "equal frame counts prove index alignment only; no timestamps exist to verify temporal residuals"
        ]
    arm_activity = report.get("arm2_target_activity", {})
    hand_activity = report.get("hand2_target_activity", {})
    if arm_activity.get("zero_ratio", 0.0) > 0.5 and hand_activity.get("zero_ratio", 1.0) == 0.0:
        report["target_write_inference"] = (
            "arm target is sparse while hand target is dense; event-driven arm writes are plausible"
        )
        report["recommended_action"] = (
            "keep target sparsity as a capture diagnostic; train on dense arm2_pos and hand2_pos"
        )
    elif arm_activity.get("zero_frames", 0):
        report["recommended_action"] = "confirm zero-target semantics before preprocessing"
    hold_check = report.get("arm2_held_target_consistency", {})
    if hold_check.get("all_segments_consistent_with_hold"):
        report["target_write_inference"] = (
            "arm actual stays within 1e-3 L2 of the previous nonzero target while target rows are zero; "
            "zero-as-no-new-command is strongly supported"
        )
        report["recommended_action"] = (
            "held-target behavior is physically consistent; train on dense executed joint positions"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--image-keys",
        nargs="+",
        default=["rgb_head", "rgb_left_hand", "rgb_right_hand"],
    )
    parser.add_argument("--output")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    files = sorted(Path(args.data_dir).glob("episode_*_success*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no episode_*_success.hdf5 under {args.data_dir}")

    audit = partial(
        audit_episode,
        image_keys=tuple(args.image_keys),
        check_images=not args.skip_image_check,
    )
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            reports = list(executor.map(audit, files))
    else:
        reports = [audit(path) for path in files]
    pattern_counts = Counter(
        tuple(item.get("arm2_target_activity", {}).get("nonzero_indices", [])) for item in reports
    )
    hold_pass = [
        item for item in reports
        if item.get("arm2_held_target_consistency", {}).get("all_segments_consistent_with_hold")
    ]
    hold_errors = [
        max(segment["max_l2_error"] for segment in item["arm2_held_target_consistency"]["segments"])
        for item in reports
        if item.get("arm2_held_target_consistency", {}).get("segments")
    ]
    effective_actions = []
    object_poses = []
    grasp_arm_positions = []
    for path in files:
        with h5py.File(path, "r") as file:
            trajectory = file.get("trajectory")
            if trajectory is None or "arm2_pos" not in trajectory or "hand2_pos" not in trajectory:
                continue
            effective_actions.append(
                np.concatenate([trajectory["arm2_pos"][:], trajectory["hand2_pos"][:]], axis=1)
            )
            if "scene/object_pose_world" in file and trajectory["arm2_pos"].shape[0] >= 3:
                object_poses.append(np.asarray(file["scene/object_pose_world"][:], dtype=np.float64))
                grasp_arm_positions.append(np.asarray(trajectory["arm2_pos"][2], dtype=np.float64))
    action_stats = None
    if effective_actions:
        action = np.concatenate(effective_actions, axis=0)
        low, high = action.min(axis=0), action.max(axis=0)
        widths = high - low
        unique_counts = [int(np.unique(np.round(action[:, index], 6)).size) for index in range(action.shape[1])]
        action_stats = {
            "dims": int(action.shape[1]),
            "min": low.tolist(),
            "max": high.tolist(),
            "range": widths.tolist(),
            "unique_count_1e-6": unique_counts,
            "warnings": [
                {"dim": index, "range": float(widths[index]), "unique_count": unique_counts[index]}
                for index in range(action.shape[1])
                if widths[index] < 1e-3 or unique_counts[index] <= 2
            ],
        }
    verifiable_count = sum("recording_completeness" in item for item in reports)
    truncated_count = sum(
        bool(item.get("recording_completeness", {}).get("truncated_recording"))
        for item in reports
    )
    complete_count = verifiable_count - truncated_count
    unknown_count = len(reports) - verifiable_count
    complete_ratio = complete_count / verifiable_count if verifiable_count else None
    if unknown_count:
        lift_protocol_decision = "blocked_missing_recording_completeness_metadata"
    elif complete_ratio is not None and complete_ratio > 0.7:
        lift_protocol_decision = "B_autonomous_lift_filter_truncated"
    elif complete_ratio is not None and complete_ratio < 0.3:
        lift_protocol_decision = "A_external_lift_uniform_truncation"
    else:
        lift_protocol_decision = "B_autonomous_lift_plus_truncated_prefix_only_after_mask_support"

    summary = {
        "episodes": len(reports),
        "structurally_aligned": sum(bool(item.get("structural_alignment")) for item in reports),
        "timestamp_verifiable": sum(bool(item.get("timestamp_fields")) for item in reports),
        "episodes_with_arm2_zero_target": sum(
            bool(item.get("arm2_zero_target", {}).get("frames")) for item in reports
        ),
        "held_target_pass": len(hold_pass),
        "held_target_fail": len(reports) - len(hold_pass),
        "lift_protocol": {
            "complete_recordings": complete_count,
            "truncated_recordings": truncated_count,
            "unknown_recordings": unknown_count,
            "complete_ratio": complete_ratio,
            "decision_rule": "complete>70%: B; complete<30%: A; otherwise: B with truncated-prefix only after mask support",
            "recommended_path": lift_protocol_decision,
        },
        "truncated_recordings": truncated_count,
        "structure_signature_counts": [
            {"signature": list(signature), "episodes": count}
            for signature, count in Counter(
                (
                    tuple(item.get("executed_activity", {}).get("arm2_pos", {}).get("dynamic_indices", [])),
                    tuple(item.get("executed_activity", {}).get("hand2_pos", {}).get("dynamic_indices", [])),
                )
                for item in reports
            ).most_common()
        ],
        "held_target_max_l2_error": {
            "p50": float(np.percentile(hold_errors, 50)) if hold_errors else None,
            "p95": float(np.percentile(hold_errors, 95)) if hold_errors else None,
            "max": float(np.max(hold_errors)) if hold_errors else None,
        },
        "held_target_error_buckets": {
            "lt_1e-3": sum(error < 1e-3 for error in hold_errors),
            "1e-3_to_2e-3": sum(1e-3 <= error < 2e-3 for error in hold_errors),
            "2e-3_to_5e-3": sum(2e-3 <= error < 5e-3 for error in hold_errors),
            "ge_5e-3": sum(error >= 5e-3 for error in hold_errors),
        },
        "arm_nonzero_pattern_counts": [
            {"nonzero_indices": list(pattern), "episodes": count}
            for pattern, count in pattern_counts.most_common()
        ],
        "executed_action_stats": action_stats,
        "object_pose_coverage": summarize_pose_coverage(
            np.asarray(object_poses), np.asarray(grasp_arm_positions)
        ) if object_poses else None,
    }
    if not args.summary_only:
        summary["reports"] = reports
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
