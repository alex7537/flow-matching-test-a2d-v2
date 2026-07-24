from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from flow_matching_test.train import _build_dataset, _build_policy, evaluate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_summary(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def _summarize_trials(trials: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = sorted(
        {
            key
            for trial in trials
            for key, value in trial["metrics"].items()
            if isinstance(value, (int, float))
        }
    )
    return {
        name: _metric_summary([float(trial["metrics"][name]) for trial in trials])
        for name in metric_names
        if all(name in trial["metrics"] for trial in trials)
    }


def _repeatability(
    trials: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    baseline = trials[0]["metrics"]
    metric_names = sorted(
        key
        for key, value in baseline.items()
        if isinstance(value, (int, float))
        and all(key in trial["metrics"] for trial in trials)
    )
    maximum_deltas = {
        name: max(
            abs(float(trial["metrics"][name]) - float(baseline[name]))
            for trial in trials[1:]
        )
        if len(trials) > 1
        else 0.0
        for name in metric_names
    }
    maximum = max(maximum_deltas.values(), default=0.0)
    return {
        "fixed_seed": trials[0]["seed"],
        "repeats": len(trials),
        "metric_max_abs_delta": maximum_deltas,
        "overall_max_abs_delta": maximum,
        "tolerance": tolerance,
        "classification": (
            "below_measurement_resolution"
            if maximum <= tolerance
            else "measurable_nondeterminism"
        ),
    }


def _load_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure CFM validation protocol variance and A800 repeatability."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-val-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sample-draws", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--protocol-seed-start", type=int, default=1000)
    parser.add_argument("--protocol-trials", type=int, default=10)
    parser.add_argument("--repeat-seed", type=int, default=424242)
    parser.add_argument("--repeat-trials", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    args = parser.parse_args()

    if args.protocol_trials < 2:
        raise ValueError("--protocol-trials must be >= 2")
    if args.repeat_trials < 2:
        raise ValueError("--repeat-trials must be >= 2")
    if args.max_val_steps < 1:
        raise ValueError("--max-val-steps must be >= 1")

    checkpoint_path = Path(args.ckpt).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    split_seed = int(training_cfg["seed"])

    train_dataset = _build_dataset(data_cfg=data_cfg, split="train", seed=split_seed)
    val_dataset = _build_dataset(data_cfg=data_cfg, split="val", seed=split_seed)
    val_dataset.action_mean = train_dataset.action_mean.copy()
    val_dataset.action_std = train_dataset.action_std.copy()
    batch_size = int(args.batch_size or training_cfg.get("batch_size", 32))
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = torch.device(args.device)
    model = _build_policy(
        policy_cfg=cfg.get("policy"),
        model_cfg=cfg["model"],
        data_cfg=data_cfg,
        train_dataset=train_dataset,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.set_action_stats(
        action_mean=torch.as_tensor(payload["action_mean"], dtype=torch.float32, device=device),
        action_std=torch.as_tensor(payload["action_std"], dtype=torch.float32, device=device),
    )
    model.num_inference_steps = int(args.inference_steps)
    model.eval()

    def run_trial(*, phase: str, trial: int, total: int, seed: int) -> dict[str, Any]:
        started = time.perf_counter()
        print(
            json.dumps(
                {"event": "trial_started", "phase": phase, "trial": trial, "total": total, "seed": seed}
            ),
            flush=True,
        )
        metrics = evaluate(
            model=model,
            loader=loader,
            device=device,
            max_steps=args.max_val_steps,
            deterministic_seed=seed,
            sample_draws=args.sample_draws,
        )
        result = {
            "seed": seed,
            "duration_seconds": time.perf_counter() - started,
            "metrics": metrics,
        }
        print(
            json.dumps(
                {
                    "event": "trial_completed",
                    "phase": phase,
                    "trial": trial,
                    "total": total,
                    **result,
                }
            ),
            flush=True,
        )
        return result

    protocol_trials = [
        run_trial(
            phase="protocol_noise",
            trial=index + 1,
            total=args.protocol_trials,
            seed=args.protocol_seed_start + index,
        )
        for index in range(args.protocol_trials)
    ]
    repeat_trials = [
        run_trial(
            phase="gpu_floating_noise",
            trial=index + 1,
            total=args.repeat_trials,
            seed=args.repeat_seed,
        )
        for index in range(args.repeat_trials)
    ]
    manifest = _load_manifest(manifest_path)
    evaluated_samples = min(len(val_dataset), args.max_val_steps * batch_size)
    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": (
            "Fixed validation subset. Protocol noise varies deterministic noise/t/sampling "
            "seeds; GPU floating noise repeats the identical seed."
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "selection": payload.get("selection_criterion", "legacy_unspecified"),
            "weights_variant": "raw",
            "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step"),
        },
        "bundle_provenance": (
            {
                "manifest_path": str(manifest_path),
                "weights_variant": manifest.get("weights_variant", "legacy_unspecified"),
                "checkpoint_selection": manifest.get(
                    "source_checkpoint_selection", "legacy_unspecified"
                ),
                "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
                "bundle_checkpoint_sha256": manifest.get("files", {})
                .get("ckpt.pt", {})
                .get("sha256"),
            }
            if manifest is not None
            else None
        ),
        "evaluation": {
            "validation_windows_total": len(val_dataset),
            "validation_windows_evaluated": evaluated_samples,
            "max_val_steps": args.max_val_steps,
            "batch_size": batch_size,
            "sample_draws": args.sample_draws,
            "inference_steps": args.inference_steps,
        },
        "protocol_noise": {
            "definition": "Variance across deterministic noise/t/sampling seeds.",
            "trials": protocol_trials,
            "summary": _summarize_trials(protocol_trials),
        },
        "gpu_floating_noise": {
            "definition": "Metric delta across repeats with identical inputs and RNG seed.",
            "trials": repeat_trials,
            **_repeatability(repeat_trials, tolerance=args.tolerance),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
