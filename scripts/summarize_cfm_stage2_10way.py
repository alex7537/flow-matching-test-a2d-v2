#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_SOURCE_SHA256 = "2e3552469842f56c85804d4c044d6c522120b7b91ae176e14e462afca21e5ea9"
EXPECTED_EPOCHS = 3
EXPECTED_STEPS = 15_267
SOURCE_BASELINE = {
    "val_loss": 0.022643056238861117,
    "val_continuous_loss": 0.01775636018232738,
    "val_keyframe_loss": 0.030505888219253947,
    "val_sample_action_mse": 0.005561013299827838,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and rank the 10-way CFM stage-2 sweep")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--hash-checkpoints", action="store_true")
    args = parser.parse_args()

    queue_dir = args.run_root / args.queue_id
    plan_rows = list(csv.DictReader((queue_dir / "plan.tsv").open(encoding="utf-8"), delimiter="\t"))
    result_rows = list(
        csv.DictReader((queue_dir / "results.tsv").open(encoding="utf-8"), delimiter="\t")
    )
    result_by_variant = {row["variant"]: row for row in result_rows}

    candidates: list[dict[str, Any]] = []
    violations: list[str] = []
    for plan in plan_rows:
        variant = plan["variant"]
        result = result_by_variant.get(variant)
        if result is None:
            violations.append(f"{variant}: no completed queue result")
            continue
        if result["status"] != "COMPLETED":
            violations.append(f"{variant}: queue status={result['status']}")
            continue

        run_name = result["run_name"]
        run_dir = args.run_root / run_name
        action_mse_checkpoint = run_dir / "best_action_mse.ckpt"
        selected_checkpoint = (
            action_mse_checkpoint if action_mse_checkpoint.is_file() else run_dir / "best.ckpt"
        )
        checkpoint_selection = (
            "val_sample_action_mse"
            if action_mse_checkpoint.is_file()
            else "historical_val_loss"
        )
        required = [
            run_dir / "config_resolved.yaml",
            run_dir / "init_events.jsonl",
            run_dir / "metrics.jsonl",
            run_dir / "summary.json",
            selected_checkpoint,
            run_dir / "latest.ckpt",
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            violations.append(f"{variant}: missing {', '.join(missing)}")
            continue
        if (run_dir / "failure.json").exists():
            violations.append(f"{variant}: failure.json exists")
            continue

        config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
        init_events = read_jsonl(run_dir / "init_events.jsonl")
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        summary = read_json(run_dir / "summary.json")
        if len(metrics) != EXPECTED_EPOCHS:
            violations.append(f"{variant}: expected {EXPECTED_EPOCHS} metrics rows, got {len(metrics)}")
        if int(metrics[-1].get("global_step", -1)) != EXPECTED_STEPS:
            violations.append(
                f"{variant}: expected final global_step {EXPECTED_STEPS}, "
                f"got {metrics[-1].get('global_step')}"
            )
        source_sha = str(init_events[-1].get("checkpoint_sha256", "")) if init_events else ""
        if source_sha != EXPECTED_SOURCE_SHA256:
            violations.append(f"{variant}: unexpected initialization checkpoint SHA-256")

        best = (
            summary["best_action_mse"]
            if checkpoint_selection == "val_sample_action_mse"
            else summary["best"]
        )
        candidate = {
            "variant": variant,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "wandb_url": summary.get("wandb", {}).get("run_url"),
            "epochs": len(metrics),
            "global_step": int(metrics[-1]["global_step"]),
            "head_lr": float(config["training"]["lr"]),
            "backbone_lr_multiplier": float(config["training"]["backbone_lr_multiplier"]),
            "freeze_encoder_backbone": bool(config["model"]["freeze_encoder_backbone"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "betas": list(config["training"]["betas"]),
            "grad_clip": float(config["training"]["grad_clip"]),
            "num_inference_steps": int(config["model"]["num_inference_steps"]),
            "best_epoch": int(summary["best_epoch"]),
            "val_loss": float(best["val_loss"]),
            "val_continuous_loss": float(best["val_continuous_loss"]),
            "val_keyframe_loss": float(best["val_keyframe_loss"]),
            "val_sample_action_mse": float(best["val_sample_action_mse"]),
            "best_checkpoint": str(selected_checkpoint),
            "checkpoint_selection": checkpoint_selection,
            "latest_checkpoint": str(run_dir / "latest.ckpt"),
            "source_checkpoint_sha256": source_sha,
        }
        candidate["relative_to_source_percent"] = {
            name: 100.0 * (SOURCE_BASELINE[name] - candidate[name]) / SOURCE_BASELINE[name]
            for name in SOURCE_BASELINE
        }
        candidate["beats_source_val_loss"] = candidate["val_loss"] < SOURCE_BASELINE["val_loss"]
        candidate["beats_source_sample_action_mse"] = (
            candidate["val_sample_action_mse"] < SOURCE_BASELINE["val_sample_action_mse"]
        )
        if args.hash_checkpoints:
            candidate["best_checkpoint_sha256"] = sha256(selected_checkpoint)
        candidates.append(candidate)

    ranked = sorted(candidates, key=lambda item: (item["val_sample_action_mse"], item["val_loss"]))
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank

    completed = len(result_rows)
    ready = completed == len(plan_rows) == 10 and len(ranked) == 10 and not violations
    report = {
        "queue_id": args.queue_id,
        "ready_for_acceptance": ready,
        "planned_variants": len(plan_rows),
        "completed_queue_results": completed,
        "audited_candidates": len(ranked),
        "expected_epochs_per_variant": EXPECTED_EPOCHS,
        "expected_steps_per_variant": EXPECTED_STEPS,
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "source_baseline": SOURCE_BASELINE,
        "source_baseline_context": (
            "Historical source metrics used 5 CFM inference steps; candidates use 10 steps "
            "except V10. Relative percentages are descriptive, not a controlled training-only "
            "comparison. Use the final source_baseline_10step rollout artifact for matched testing."
        ),
        "violations": violations,
        "ranking_key": ["val_sample_action_mse", "val_loss"],
        "ranking_scope": (
            "Only checkpoints saved by each run are eligible. Deterministic re-evaluation "
            "cannot recover historical epochs that were never saved."
        ),
        "candidates": ranked,
    }
    (queue_dir / "acceptance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# CFM Stage-2 10-Way Acceptance Report",
        "",
        f"- Ready: **{ready}**",
        f"- Completed: **{completed}/10**",
        f"- Audited: **{len(ranked)}/10**",
        f"- Violations: **{len(violations)}**",
        "- Source comparison: historical 5-step metrics; **not inference-step matched**",
        "",
        "| Rank | Variant | Checkpoint selection | Best epoch | val sample MSE | vs source | val loss | vs source | keyframe loss | W&B |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in ranked:
        url = item.get("wandb_url") or ""
        wandb = f"[run]({url})" if url else "-"
        lines.append(
            f"| {item['rank']} | {item['variant']} | {item['checkpoint_selection']} | "
            f"{item['best_epoch']} | "
            f"{item['val_sample_action_mse']:.8f} | "
            f"{item['relative_to_source_percent']['val_sample_action_mse']:+.2f}% | "
            f"{item['val_loss']:.8f} | {item['relative_to_source_percent']['val_loss']:+.2f}% | "
            f"{item['val_keyframe_loss']:.8f} | {wandb} |"
        )
    if violations:
        lines.extend(["", "## Violations", ""] + [f"- {value}" for value in violations])
    (queue_dir / "ACCEPTANCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({key: report[key] for key in report if key != "candidates"}, ensure_ascii=False))
    raise SystemExit(0 if ready else 2)


if __name__ == "__main__":
    main()
