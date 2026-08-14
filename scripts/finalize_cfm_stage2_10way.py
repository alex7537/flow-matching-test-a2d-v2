#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from flow_matching_test.export_bundle import export_eval_bundle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = bundle / name
        if not path.is_file():
            raise FileNotFoundError(f"bundle file missing: {path}")
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise ValueError(f"bundle SHA-256 mismatch: {path}")
    return manifest


def trajectory_diagnostics(action: list[list[float]]) -> dict[str, float]:
    def l2(values: list[float]) -> float:
        return math.sqrt(sum(value * value for value in values))

    deltas = [
        [current - previous for current, previous in zip(action[index], action[index - 1])]
        for index in range(1, len(action))
    ]
    accelerations = [
        [current - previous for current, previous in zip(deltas[index], deltas[index - 1])]
        for index in range(1, len(deltas))
    ]
    delta_norms = [l2(delta) for delta in deltas]
    acceleration_norms = [l2(value) for value in accelerations]
    arm_delta_norms = [l2(delta[:7]) for delta in deltas]
    hand_delta_norms = [l2(delta[7:]) for delta in deltas]
    return {
        "step_delta_l2_mean": sum(delta_norms) / len(delta_norms),
        "step_delta_l2_max": max(delta_norms),
        "step_acceleration_l2_mean": sum(acceleration_norms) / len(acceleration_norms),
        "step_acceleration_l2_max": max(acceleration_norms),
        "arm_step_delta_l2_mean": sum(arm_delta_norms) / len(arm_delta_norms),
        "hand_step_delta_l2_mean": sum(hand_delta_norms) / len(hand_delta_norms),
    }


def create_test_artifact(
    *,
    repo_root: Path,
    checkpoint: Path,
    bundle: Path,
    reference_dir: Path,
    variant: str,
    rank: int | None,
    run_name: str,
    checkpoint_sha256: str,
    num_inference_steps: int,
) -> dict[str, Any]:
    if not bundle.exists():
        export_eval_bundle(
            checkpoint,
            bundle,
            execute_horizon=16,
            num_inference_steps=num_inference_steps,
            camera_resolutions={"rgb_head": (640, 480), "rgb_right_hand": (640, 480)},
            data_version="a2d_450gb_rgb_v1",
            archive=False,
        )
    bundle_manifest = verify_bundle(bundle)
    reference_output = reference_dir / "reference_output.json"
    if not reference_output.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.export_reference_inference",
                "--ckpt",
                str(checkpoint),
                "--output-dir",
                str(reference_dir),
                "--split",
                "val",
                "--episode-index",
                "0",
                "--frame",
                "0",
                "--seed",
                "20260722",
                "--num-inference-steps",
                str(num_inference_steps),
                "--device",
                "cuda",
            ],
            cwd=repo_root,
            check=True,
        )
    reference = json.loads(reference_output.read_text(encoding="utf-8"))
    return {
        "rank": rank,
        "variant": variant,
        "run_name": run_name,
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": checkpoint_sha256,
        "bundle": str(bundle),
        "bundle_model_sha256": bundle_manifest["files"]["ckpt.pt"]["sha256"],
        "reference_input": str(reference_dir / reference["reference_input"]),
        "reference_input_sha256": reference["reference_input_sha256"],
        "reference_output": str(reference_output),
        "num_inference_steps": num_inference_steps,
        "trajectory_diagnostics": trajectory_diagnostics(reference["action"]),
        "normalized_trajectory_diagnostics": trajectory_diagnostics(reference["action_normalized"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create test artifacts for a completed CFM sweep")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--queue-id", required=True)
    args = parser.parse_args()

    queue_dir = args.run_root / args.queue_id
    if not (queue_dir / "QUEUE_COMPLETE").is_file():
        raise RuntimeError("training queue is not complete")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.summarize_cfm_stage2_10way",
            "--run-root",
            str(args.run_root),
            "--queue-id",
            args.queue_id,
            "--hash-checkpoints",
        ],
        cwd=args.repo_root,
        check=True,
    )
    report = json.loads((queue_dir / "acceptance_report.json").read_text(encoding="utf-8"))
    if not report["ready_for_acceptance"]:
        raise RuntimeError("acceptance audit did not pass")

    bundles_root = queue_dir / "test_bundles"
    references_root = queue_dir / "reference_inference"
    bundles_root.mkdir(exist_ok=True)
    references_root.mkdir(exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for candidate in report["candidates"]:
        variant = candidate["variant"]
        checkpoint = Path(candidate["best_checkpoint"])
        bundle = bundles_root / variant
        reference_dir = references_root / variant
        artifact = create_test_artifact(
            repo_root=args.repo_root,
            checkpoint=checkpoint,
            bundle=bundle,
            reference_dir=reference_dir,
            variant=variant,
            rank=candidate["rank"],
            run_name=candidate["run_name"],
            checkpoint_sha256=candidate["best_checkpoint_sha256"],
            num_inference_steps=candidate["num_inference_steps"],
        )
        artifact["val_sample_action_mse"] = candidate["val_sample_action_mse"]
        artifact["val_loss"] = candidate["val_loss"]
        artifacts.append(artifact)

    source_checkpoint = (
        args.run_root
        / "cfm_1090ep_vit_finetune_01x_5ep_seed42_20260721_142310"
        / "best.ckpt"
    )
    baseline_5step = create_test_artifact(
        repo_root=args.repo_root,
        checkpoint=source_checkpoint,
        bundle=bundles_root / "source_baseline_5step",
        reference_dir=references_root / "source_baseline_5step",
        variant="source_baseline_5step",
        rank=None,
        run_name="cfm_1090ep_vit_finetune_01x_5ep_seed42_20260721_142310",
        checkpoint_sha256=report["source_checkpoint_sha256"],
        num_inference_steps=5,
    )
    baseline_5step.update(report["source_baseline"])
    baseline_10step = create_test_artifact(
        repo_root=args.repo_root,
        checkpoint=source_checkpoint,
        bundle=bundles_root / "source_baseline_10step",
        reference_dir=references_root / "source_baseline_10step",
        variant="source_baseline_10step",
        rank=None,
        run_name="cfm_1090ep_vit_finetune_01x_5ep_seed42_20260721_142310",
        checkpoint_sha256=report["source_checkpoint_sha256"],
        num_inference_steps=10,
    )

    payload = {
        "queue_id": args.queue_id,
        "status": "ready_for_rollout_acceptance",
        "artifact_count": len(artifacts),
        "camera_resolutions": {"rgb_head": [640, 480], "rgb_right_hand": [640, 480]},
        "execute_horizon": 16,
        "reference_seed": 20260722,
        "baseline_5step": baseline_5step,
        "baseline_10step": baseline_10step,
        "artifacts": artifacts,
    }
    (queue_dir / "TEST_ARTIFACTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (queue_dir / "FINALIZATION_COMPLETE").touch()
    print(json.dumps({"status": payload["status"], "artifact_count": len(artifacts)}))


if __name__ == "__main__":
    main()
