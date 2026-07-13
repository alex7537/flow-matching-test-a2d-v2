from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


DEFAULT_JOINT_ORDER = [
    *(f"joint_arm2_link_{index}" for index in range(1, 8)),
    "hand2_joint_link_1_1",
    "hand2_joint_link_2_1",
    "hand2_joint_link_3_1",
    "hand2_joint_link_4_1",
    "hand2_joint_link_5_1",
    "hand2_joint_link_1_2",
]


def _git_sha(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inference_config(
    checkpoint: dict[str, Any],
    *,
    execute_horizon: int,
    camera_resolutions: dict[str, tuple[int, int]],
    joint_order: list[str],
) -> dict[str, Any]:
    train_cfg = checkpoint["config"]
    data_cfg = train_cfg["data"]
    model_cfg = train_cfg["model"]
    image_keys = list(data_cfg["image_keys"])
    bundle_cfg = train_cfg.get("deployment", {}).get("eval_bundle", {})
    chunk_size = int(data_cfg["action_horizon"])
    if not 1 <= execute_horizon <= chunk_size:
        raise ValueError(f"execute_horizon must be in [1,{chunk_size}]")
    if len(joint_order) != 13 or len(set(joint_order)) != 13:
        raise ValueError("joint_order must contain 13 unique articulation joint names")

    image_size = int(data_cfg["image_size"])
    cameras = []
    for key in image_keys:
        width, height = camera_resolutions.get(key, (image_size, image_size))
        cameras.append({"name": key, "model_key": key, "resolution": [width, height]})

    return {
        "schema_version": 2,
        "model": {
            "encoder_type": str(model_cfg.get("encoder_type", "cnn")),
            "timm_model_name": str(model_cfg.get("timm_model_name", "vit_small_r26_s32_224")),
            "timm_pretrained": False,
            "timm_token_mode": str(model_cfg.get("timm_token_mode", "spatial")),
            "timm_tokens_per_frame": int(model_cfg.get("timm_tokens_per_frame", 1)),
            "use_proprio": bool(model_cfg.get("use_proprio", True)),
            "d_model": int(model_cfg.get("d_model", 128)),
            "n_head": int(model_cfg.get("n_head", 4)),
            "n_layer": int(model_cfg.get("n_layer", 4)),
            "dropout": float(model_cfg.get("dropout", 0.0)),
            "cfm": {
                "time_eps": float(model_cfg.get("time_eps", 1.0e-3)),
                "num_inference_steps": int(model_cfg.get("num_inference_steps", 40)),
            },
        },
        "action": {
            "dim": 13,
            "chunk_size": chunk_size,
            "execute_horizon": execute_horizon,
            "target": "executed_joint_position",
            "range_guard_margin_ratio": float(bundle_cfg.get("action_range_guard_margin_ratio", 0.1)),
            "layout": [
                {"name": "arm2_pos", "dim": 7},
                {"name": "hand2_pos", "dim": 6},
            ],
        },
        "obs": {
            "cameras": cameras,
            "image_size": image_size,
            "history_steps": int(data_cfg.get("history_steps", 1)),
            "proprio_dim": 13,
            "freq_hz": 30,
            "freq_hz_status": "deployment_assumption_unverified_no_dataset_timestamps",
        },
        "joint_order": joint_order,
        "sampling": {"seed": int(train_cfg.get("training", {}).get("seed", 42))},
    }


def export_eval_bundle(
    ckpt_path: Path,
    out_dir: Path,
    *,
    execute_horizon: int | None = None,
    camera_resolutions: dict[str, tuple[int, int]] | None = None,
    joint_order: list[str] | None = None,
    data_version: str = "unknown",
    archive: bool = False,
) -> Path:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict", "config", "normalizer", "stats_digest",
        "data_provenance", "training_environment",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {missing}")

    chunk_size = int(checkpoint["config"]["data"]["action_horizon"])
    config = _inference_config(
        checkpoint,
        execute_horizon=chunk_size if execute_horizon is None else execute_horizon,
        camera_resolutions=camera_resolutions or {},
        joint_order=joint_order or list(DEFAULT_JOINT_ORDER),
    )

    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty bundle directory: {out_dir}")
    else:
        out_dir.mkdir(parents=True)

    torch.save(checkpoint["model_state_dict"], out_dir / "ckpt.pt")
    (out_dir / "norm_stats.json").write_text(
        json.dumps(checkpoint["normalizer"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    rollout_environment_spec = checkpoint["config"].get("deployment", {}).get(
        "eval_bundle", {}
    ).get("rollout_environment", {})
    manifest = {
        "schema_version": 2,
        "git_sha": _git_sha(Path(__file__).resolve().parents[1]),
        "data_version": data_version,
        "train_step": int(checkpoint.get("global_step", -1)),
        "train_epoch": int(checkpoint.get("epoch", -1)),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_checkpoint": str(ckpt_path.resolve()),
        "stats_digest": checkpoint["stats_digest"],
        "data_provenance": checkpoint["data_provenance"],
        "training_environment": checkpoint["training_environment"],
        "rollout_environment_spec": rollout_environment_spec,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        f"Flow-matching rollout bundle from `{ckpt_path.name}` at step "
        f"{manifest['train_step']}.\n",
        encoding="utf-8",
    )

    manifest["files"] = {
        name: {"sha256": _sha256(out_dir / name), "bytes": (out_dir / name).stat().st_size}
        for name in ("ckpt.pt", "norm_stats.json", "config.yaml", "README.md")
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if archive:
        archive_path = out_dir.with_suffix(".tgz")
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(out_dir, arcname=out_dir.name)
        return archive_path
    return out_dir


def _camera_resolutions(values: list[str]) -> dict[str, tuple[int, int]]:
    result = {}
    for value in values:
        try:
            key, resolution = value.split("=", 1)
            width, height = resolution.lower().split("x", 1)
            result[key] = (int(width), int(height))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"camera resolution must look like rgb_head=640x480, got {value!r}"
            ) from exc
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a self-contained rollout eval bundle")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute-horizon", type=int)
    parser.add_argument("--camera-resolution", action="append", default=[])
    parser.add_argument("--joint-order", nargs=13, default=DEFAULT_JOINT_ORDER)
    parser.add_argument("--data-version", default="unknown")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    output = export_eval_bundle(
        args.ckpt,
        args.out,
        execute_horizon=args.execute_horizon,
        camera_resolutions=_camera_resolutions(args.camera_resolution),
        joint_order=list(args.joint_order),
        data_version=args.data_version,
        archive=args.archive,
    )
    print(output)


if __name__ == "__main__":
    main()
