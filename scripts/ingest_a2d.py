from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flow_matching_test.action_contract import (
    ACTION_LAYOUTS,
    EXECUTED_ACTION_SEMANTICS,
)


def admission_reason(report: dict, admission: str) -> str | None:
    if report.get("errors") or not report.get("structural_alignment"):
        return "structural_alignment_failed"
    if admission == "complete-lift":
        completeness = report.get("recording_completeness")
        if completeness is None:
            return "recording_completeness_unknown"
        if completeness.get("truncated_recording"):
            return "truncated_lift_recording"
    return None


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"existing link points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(destination)
    os.symlink(source.resolve(), destination)


def run(command: list[str], *, cwd: Path, quiet: bool = False) -> None:
    print("+", " ".join(command))
    subprocess.run(
        command,
        check=True,
        cwd=cwd,
        stdout=subprocess.DEVNULL if quiet else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Versioned raw HDF5 audit and ingestion")
    parser.add_argument("--src", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--admission", choices=("structural", "complete-lift"), default="complete-lift")
    parser.add_argument(
        "--action-semantics",
        choices=sorted(ACTION_LAYOUTS),
        default=EXECUTED_ACTION_SEMANTICS,
    )
    parser.add_argument("--image-keys", nargs="+", default=["rgb_head", "rgb_right_hand"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--motion-threshold", type=float, default=1.0e-4)
    parser.add_argument("--keyframe-threshold", type=float, default=0.1)
    parser.add_argument("--norm-stats", default="norm_stats.json")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    src = Path(args.src).resolve()
    dataset_dir = Path(args.output_root).resolve() / args.dataset_version
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(src.glob("episode_*_success*.hdf5"))
    if not raw_files:
        raise FileNotFoundError(f"no raw episodes under {src}")
    source_inventory = [
        {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in raw_files
    ]
    ingest_contract = {
        "admission": args.admission,
        "action_semantics": args.action_semantics,
        "image_keys": args.image_keys,
        "image_size": args.image_size,
        "jpeg_quality": args.jpeg_quality,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "motion_threshold": args.motion_threshold,
        "keyframe_threshold": args.keyframe_threshold,
        "norm_stats": args.norm_stats,
    }
    manifest_path = dataset_dir / "episode_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_inventory") != source_inventory:
            raise ValueError(
                f"dataset version {args.dataset_version!r} already binds a different source inventory; "
                "choose a new --dataset-version"
            )
        existing_contract = dict(existing.get("ingest_contract", {}))
        # Manifests produced before action semantics became configurable used
        # the executed-joint contract implicitly.
        existing_contract.setdefault("action_semantics", EXECUTED_ACTION_SEMANTICS)
        if existing_contract != ingest_contract:
            raise ValueError(
                f"dataset version {args.dataset_version!r} already binds different ingest parameters; "
                "choose a new --dataset-version"
            )

    audit_path = dataset_dir / "audit_report.json"
    run([
        sys.executable,
        str(repo / "scripts/check_data_alignment.py"),
        "--data-dir", str(src),
        "--image-keys", *args.image_keys,
        "--workers", str(args.workers),
        "--output", str(audit_path),
    ], cwd=repo, quiet=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    reports = {report["episode"]: report for report in audit.get("reports", [])}
    if len(reports) != len(raw_files):
        raise ValueError("audit report does not contain one report per source episode")

    accepted_dir = dataset_dir / "_accepted_raw"
    quarantine_dir = dataset_dir / "quarantine"
    accepted: list[str] = []
    quarantined: dict[str, str] = {}
    for path in raw_files:
        reason = admission_reason(reports[path.name], args.admission)
        if reason is None:
            accepted.append(path.name)
            link_file(path, accepted_dir / path.name)
        else:
            quarantined[path.name] = reason
            link_file(path, quarantine_dir / path.name)
    if len(accepted) < 2:
        raise ValueError("at least two admitted episodes are required for episode-level train/val split")

    run([
        sys.executable,
        str(repo / "scripts/preprocess_a2d.py"),
        "--src", str(accepted_dir),
        "--dst", str(dataset_dir),
        "--image-keys", *args.image_keys,
        "--image-size", str(args.image_size),
        "--jpeg-quality", str(args.jpeg_quality),
        "--workers", str(args.workers),
        "--motion-threshold", str(args.motion_threshold),
        "--keyframe-threshold", str(args.keyframe_threshold),
        "--action-semantics", args.action_semantics,
    ], cwd=repo)
    run([
        sys.executable,
        "-m", "flow_matching_test.a2d_dataset",
        "--data-dir", str(dataset_dir),
        "--image-keys", *args.image_keys,
        "--compute-stats",
        "--action-semantics", args.action_semantics,
        "--norm-stats", args.norm_stats,
        "--seed", str(args.seed),
        "--val-ratio", str(args.val_ratio),
        "--motion-threshold", str(args.motion_threshold),
        "--keyframe-threshold", str(args.keyframe_threshold),
    ], cwd=repo)

    stats = json.loads((dataset_dir / args.norm_stats).read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "dataset_version": args.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(src),
        "source_inventory": source_inventory,
        "ingest_contract": ingest_contract,
        "admission": args.admission,
        "accepted": accepted,
        "quarantined": quarantined,
        "image_keys": args.image_keys,
        "segmentation": {
            "version": 1,
            "motion_threshold": args.motion_threshold,
            "keyframe_threshold": args.keyframe_threshold,
        },
        "stats_file": args.norm_stats,
        "train_episode_digest": stats["train_episode_digest"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"done: {dataset_dir} accepted={len(accepted)} "
        f"quarantined={len(quarantined)} digest={stats['train_episode_digest']}"
    )


if __name__ == "__main__":
    main()
