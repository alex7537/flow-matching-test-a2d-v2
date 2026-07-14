#!/usr/bin/env python3
"""Verify that offline W&B runs and their metric histories reached the server."""

from __future__ import annotations

import argparse

import wandb


def parse_run(value: str) -> tuple[str, int]:
    run_id, separator, expected_rows = value.partition(":")
    if not separator or not run_id or not expected_rows.isdigit():
        raise argparse.ArgumentTypeError("expected RUN_ID:EXPECTED_ROWS")
    return run_id, int(expected_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--metric", default="train_flow_loss")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        metavar="RUN_ID:EXPECTED_ROWS",
    )
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    failures: list[str] = []
    for run_id, expected_rows in args.run:
        path = f"{args.entity}/{args.project}/{run_id}"
        try:
            run = api.run(path)
            rows = sum(1 for _ in run.scan_history(keys=[args.metric], page_size=1000))
        except Exception as exc:  # W&B exposes several backend-specific errors.
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        if rows != expected_rows:
            failures.append(
                f"{path}: {args.metric} rows={rows}, expected={expected_rows}"
            )
            continue
        print(f"SERVER_OK: {run.name} {run.url} rows={rows}")

    if failures:
        for failure in failures:
            print(f"SERVER_FAIL: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
