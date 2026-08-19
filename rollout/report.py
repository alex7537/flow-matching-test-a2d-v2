from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_report(
    results: list[dict[str, Any]],
    *,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not results:
        raise ValueError("results.jsonl contains no results")

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        return {
            "trials": count,
            "success_rate": sum(bool(item["success"]) for item in items) / count,
            "first_attempt_success_rate": sum(
                bool(item.get("first_attempt_success", item["success"])) for item in items
            )
            / count,
            "recovered_success_rate": sum(
                bool(item.get("recovered_success", False)) for item in items
            )
            / count,
            "retry_rate": sum(int(item.get("retry_count", 0)) > 0 for item in items) / count,
            "mean_attempt_count": sum(int(item.get("attempt_count", 1)) for item in items)
            / count,
            "approach_rate": sum(bool(item["approach_success"]) for item in items) / count,
            "close_rate": sum(bool(item["close_success"]) for item in items) / count,
            "lift_rate": sum(bool(item["lift_success"]) for item in items) / count,
            "failure_stages": dict(Counter(item.get("failure_stage") or "none" for item in items)),
        }

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[str(result.get("metric_group", "default"))].append(result)
    report = {
        "overall": summarize(results),
        "metric_groups": {name: summarize(items) for name, items in sorted(groups.items())},
    }
    if run_metadata is not None:
        report["provenance"] = {
            key: run_metadata.get(key)
            for key in (
                "weights_variant",
                "checkpoint_selection",
                "source_checkpoint_sha256",
                "bundle_checkpoint_sha256",
            )
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rollout results.jsonl")
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    input_path = args.input_path
    if input_path.is_dir():
        input_path = input_path / "results.jsonl"
    results = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    metadata_path = input_path.with_name("run_metadata.json")
    run_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    report = build_report(results, run_metadata=run_metadata)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output_path = args.out or input_path.with_name("report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
