from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PHYSICS_DT = 1.0 / 120.0
RENDER_DT = 1.0 / 30.0


def _yaw_quaternion(degrees: float) -> list[float]:
    radians = math.radians(degrees) * 0.5
    return [math.cos(radians), 0.0, 0.0, math.sin(radians)]


def load_trials(grid: dict[str, Any]) -> list[dict[str, Any]]:
    if "trials" in grid:
        return list(grid["trials"])
    trials = []
    for group_name in ("primary_xy_interpolation", "sparse_boundary_reference"):
        group = grid.get(group_name)
        if not group:
            continue
        for x in group["x_m"]:
            for y in group["y_m"]:
                for seed in group["sampling_seeds"]:
                    trials.append(
                        {
                            "id": f"{group_name}_x{x:.6f}_y{y:.6f}_s{seed}",
                            "metric_group": group["metric_group"],
                            "sampling_seed": seed,
                            "object_position": [x, y, float(grid["sim"]["object_z_m"])],
                            "object_orientation_wxyz": _yaw_quaternion(float(group.get("yaw_offset_deg", 0.0))),
                        }
                    )
    yaw = grid.get("yaw_sweep")
    if yaw:
        for degrees in yaw["yaw_offset_deg"]:
            for seed in yaw["sampling_seeds"]:
                trials.append(
                    {
                        "id": f"yaw_{degrees:+04d}_s{seed}",
                        "metric_group": yaw["metric_group"],
                        "sampling_seed": seed,
                        "object_position": [yaw["x_m"], yaw["y_m"], float(grid["sim"]["object_z_m"])],
                        "object_orientation_wxyz": _yaw_quaternion(float(degrees)),
                    }
                )
    return trials


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _bundle_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "weights_variant": manifest.get("weights_variant", "legacy_unspecified"),
        "checkpoint_selection": manifest.get(
            "source_checkpoint_selection", "legacy_unspecified"
        ),
        "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
        "bundle_checkpoint_sha256": manifest.get("files", {})
        .get("ckpt.pt", {})
        .get("sha256"),
    }


def _execute_trial(
    *,
    policy: Any,
    env: Any,
    checker: Any,
    obs: dict[str, Any],
    horizon: int,
    max_chunks: int,
    base_seed: int,
    retry_config: Any,
) -> dict[str, Any]:
    from rollout.retry_controller import GraspRetryController

    controller = GraspRetryController(retry_config)
    attempts: list[dict[str, Any]] = []
    recovery_steps_total = 0

    for attempt_index in range(retry_config.attempt_limit):
        controller.begin_attempt(attempt_index)
        attempt_seed = controller.sampling_seed(base_seed)
        policy.seed = attempt_seed
        policy.reset()
        checker.reset()
        checker.prime(env.state())
        retry_reason = None

        for _ in range(max_chunks):
            chunk = policy.infer(obs, execute_horizon=horizon)
            for action in chunk[:horizon]:
                obs = env.step(action)
                checker.update(env.state())
                controller.observe(checker)
                if checker.done():
                    break
                retry_reason = controller.retry_reason(checker)
                if retry_reason is not None:
                    break
            if checker.done() or retry_reason is not None:
                break

        if (
            retry_config.enabled
            and not checker.done()
            and not checker.closed
            and retry_reason is None
        ):
            retry_reason = "attempt_budget_exhausted"

        attempt_summary = {
            "attempt_index": attempt_index,
            "sampling_seed": attempt_seed,
            **checker.summary(),
            "retry_reason": retry_reason,
            "retry_performed": bool(retry_reason is not None and controller.can_retry()),
        }
        attempts.append(attempt_summary)

        if checker.done() or not attempt_summary["retry_performed"]:
            break
        obs = env.recover(
            steps=retry_config.recovery_steps,
            joint_positions=retry_config.recovery_joint_positions,
        )
        recovery_steps_total += retry_config.recovery_steps

    final = checker.summary()
    final_attempt_steps = int(final["steps"])
    final.update(
        {
            "steps": sum(int(attempt["steps"]) for attempt in attempts),
            "final_attempt_steps": final_attempt_steps,
            "attempt_count": len(attempts),
            "retry_count": max(0, len(attempts) - 1),
            "first_attempt_success": bool(attempts and attempts[0]["success"]),
            "recovered_success": bool(final["success"] and len(attempts) > 1),
            "recovery_steps": recovery_steps_total,
            "attempts": attempts,
        }
    )
    return final


def run(
    bundle_dir: Path,
    grid_path: Path,
    out_dir: Path,
    *,
    trial_id: str | None = None,
    execute_horizon: int | None = None,
) -> None:
    from rollout.policy_wrapper import Policy
    from rollout.retry_controller import GraspRetryConfig
    from rollout.sim_env import GraspEnv
    from rollout.success_checker import ThreePhaseChecker

    bundle_cfg = yaml.safe_load((bundle_dir / "config.yaml").read_text())
    grid = yaml.safe_load(grid_path.read_text())
    execution_cfg = grid.get("execution", {})
    retry_config = GraspRetryConfig.from_execution(execution_cfg)
    physics_dt = float(execution_cfg.get("physics_dt_s", PHYSICS_DT))
    render_dt = float(execution_cfg.get("render_dt_s", RENDER_DT))
    chunk_size = int(bundle_cfg["action"]["chunk_size"])
    max_chunks = int(execution_cfg.get("max_chunks", 16))
    if max_chunks < 1:
        raise ValueError("execution.max_chunks must be positive")
    horizon = int(
        bundle_cfg["action"]["execute_horizon"] if execute_horizon is None else execute_horizon
    )
    if not 1 <= horizon <= chunk_size:
        raise ValueError(f"execute_horizon must be in [1,{chunk_size}]")
    env_cfg = {**bundle_cfg, "sim": grid.get("sim", {})}
    policy = Policy(bundle_dir, device="cuda")
    env = GraspEnv(physics_dt=physics_dt, render_dt=render_dt, cfg=env_cfg)
    checker = ThreePhaseChecker(grid.get("success_criteria", {}))
    trials = load_trials(grid)
    if trial_id is not None:
        trials = [trial for trial in trials if trial["id"] == trial_id]
        if not trials:
            raise ValueError(f"unknown trial id: {trial_id}")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle": str(bundle_dir.resolve()),
        "grid": str(grid_path.resolve()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "isaac_sim": _package_version("isaacsim"),
        "torch": _package_version("torch"),
        "timm": _package_version("timm"),
        "physics_dt_s": physics_dt,
        "render_dt_s": render_dt,
        "timing_status": execution_cfg.get(
            "timing_status", "deployment_assumption_unverified_no_dataset_timestamps"
        ),
        "chunk_size": chunk_size,
        "execute_horizon": horizon,
        "replan": execution_cfg.get("replan", "after_execute_horizon"),
        "task_grasp_retry": asdict(retry_config),
        "physics_snapshot": grid.get("sim", {}).get("physics_snapshot", {}),
        **_bundle_provenance(policy.manifest),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        for trial in trials:
            base_seed = int(trial.get("sampling_seed", policy.seed))
            obs = env.reset(trial)
            error = None
            execution_result = None
            try:
                execution_result = _execute_trial(
                    policy=policy,
                    env=env,
                    checker=checker,
                    obs=obs,
                    horizon=horizon,
                    max_chunks=max_chunks,
                    base_seed=base_seed,
                    retry_config=retry_config,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            result = {
                "trial_id": trial["id"],
                "metric_group": trial.get("metric_group", "default"),
                "sampling_seed": trial.get("sampling_seed"),
                "chunk_size": chunk_size,
                "execute_horizon": horizon,
                **(execution_result or checker.summary()),
                "error": error,
            }
            if error:
                result["success"] = False
                result["failure_stage"] = "simulator_error"
            results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            results_file.flush()
            env.save_video(out_dir / "videos" / f"{trial['id']}.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-grid Isaac Sim rollouts")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--execute-horizon", type=int)
    parser.add_argument("--livestream", type=int, default=0)
    args = parser.parse_args()

    from isaacsim import SimulationApp

    launch_config: dict[str, Any] = {"headless": args.livestream == 0, "enable_cameras": True}
    if args.livestream:
        launch_config["livestream"] = args.livestream
    app = SimulationApp(launch_config)
    try:
        run(
            args.bundle,
            args.grid,
            args.out,
            trial_id=args.trial_id,
            execute_horizon=args.execute_horizon,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
