from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from flow_matching_test.a2d_dataset import A2DConfig, A2DProcessedWindowDataset
from flow_matching_test.export_bundle import DEFAULT_JOINT_ORDER, _git_sha, export_eval_bundle
from flow_matching_test.policies.base import ActionPolicy
from flow_matching_test.policies.factory import (
    build_policy,
    materialize_policy_config,
    resolve_policy_type,
)
from flow_matching_test.policies.flow_matching import FlowMatchingPolicy
from flow_matching_test.rerun_logger import RerunTrainVisualizer
from flow_matching_test.segmentation import SEGMENTATION_VERSION
from flow_matching_test.training_watchdog import StallDetails, TrainingWatchdog
from flow_matching_test.wandb_logger import WandbLogger


def _deep_update(base: dict[str, Any], key_path: str, raw_value: str) -> None:
    cursor = base
    parts = key_path.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = yaml.safe_load(raw_value)


def load_config(path: str, overrides: list[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError("Config root must be a mapping")
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, value = item.split("=", 1)
        _deep_update(cfg, key, value)
    return cfg


def _to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _to_device(value, device) for key, value in batch.items()}
    return batch


def _grad_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            norm = parameter.grad.detach().float().norm(2)
            squared += float(norm.item()) ** 2
    return math.sqrt(squared)


def _parameter_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        norm = parameter.detach().float().norm(2)
        squared += float(norm.item()) ** 2
    return math.sqrt(squared)


def _snapshot_parameters(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    """Keep an epoch-start CPU snapshot for a low-frequency update diagnostic."""
    return [parameter.detach().float().cpu().clone() for parameter in parameters]


def _parameter_update_ratio(
    parameters: list[torch.nn.Parameter],
    reference: list[torch.Tensor],
) -> float:
    if len(parameters) != len(reference):
        raise ValueError("Parameter snapshot length mismatch")
    reference_squared = 0.0
    delta_squared = 0.0
    for parameter, baseline in zip(parameters, reference, strict=True):
        current = parameter.detach().float().cpu()
        reference_squared += float(torch.sum(baseline * baseline).item())
        delta = current - baseline
        delta_squared += float(torch.sum(delta * delta).item())
    return math.sqrt(delta_squared) / max(math.sqrt(reference_squared), 1.0e-12)


@torch.no_grad()
def _update_ema_model(
    ema_model: ActionPolicy,
    model: ActionPolicy,
    *,
    decay: float,
) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    if ema_state.keys() != model_state.keys():
        raise ValueError("EMA model state does not match the training model")
    for name, ema_value in ema_state.items():
        model_value = model_state[name].detach()
        if ema_value.is_floating_point():
            ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


@torch.no_grad()
def _encoder_feature_std(model: ActionPolicy, obs: dict[str, torch.Tensor]) -> float:
    """Mean per-channel std of raw visual encoder tokens on one diagnostic batch."""
    composer = getattr(model, "obs_composer", None)
    encoders = getattr(composer, "encoders", None)
    if encoders is None:
        return 0.0
    per_encoder: list[float] = []
    for encoder in encoders:
        token_map = encoder(obs)
        features = [
            value.detach().float().reshape(-1, value.shape[-1])
            for value in token_map.values()
            if torch.is_tensor(value) and value.ndim >= 2
        ]
        if not features:
            continue
        merged = torch.cat(features, dim=0)
        per_encoder.append(float(merged.std(dim=0, unbiased=False).mean().item()))
    return float(np.mean(per_encoder)) if per_encoder else 0.0


def _set_backbone_frozen(model: ActionPolicy, *, frozen: bool) -> int:
    """Freeze only the pretrained visual backbone, leaving adapters and policy head trainable."""
    parameters = model.backbone_parameters()
    for parameter in parameters:
        parameter.requires_grad_(not frozen)
    return sum(parameter.numel() for parameter in parameters)


def _runtime_environment() -> dict[str, Any]:
    try:
        import timm
        timm_version = timm.__version__
    except Exception:
        timm_version = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "timm": timm_version,
        "numpy": np.__version__,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def _build_dataset(*, data_cfg: dict[str, Any], split: str, seed: int):
    cfg = A2DConfig(
        data_dir=str(data_cfg["data_dir"]),
        image_keys=tuple(data_cfg.get("image_keys", ["rgb_head", "rgb_left_hand", "rgb_right_hand"])),
        image_size=int(data_cfg.get("image_size", 224)),
        history_steps=int(data_cfg.get("history_steps", 1)),
        action_horizon=int(data_cfg.get("action_horizon", 16)),
        action_offset_steps=int(data_cfg.get("action_offset_steps", 1)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        aug_brightness=float(data_cfg.get("aug_brightness", 0.2)),
        aug_contrast=float(data_cfg.get("aug_contrast", 0.4)),
        aug_saturation=float(data_cfg.get("aug_saturation", 0.2)),
        aug_hue=float(data_cfg.get("aug_hue", 0.05)),
        aug_random_crop_pad=int(data_cfg.get("aug_random_crop_pad", 8)),
        transition_oversample_factor=int(data_cfg.get("transition_oversample_factor", 1)),
        transition_threshold=float(data_cfg.get("transition_threshold", 0.1)),
        motion_threshold=float(data_cfg.get("motion_threshold", 1.0e-4)),
        allow_duplicate_episodes=bool(data_cfg.get("allow_duplicate_episodes", False)),
        max_open_hdf5_files=int(data_cfg.get("max_open_hdf5_files", 64)),
        range_eps=float(data_cfg.get("range_eps", 1.0e-4)),
        norm_stats=str(data_cfg.get("norm_stats", "norm_stats.json")),
        dataset_manifest=data_cfg.get("dataset_manifest"),
        split_manifest=data_cfg.get("split_manifest"),
    )
    return A2DProcessedWindowDataset(cfg=cfg, split=split)


def _build_policy(
    *,
    policy_cfg: dict[str, Any] | None,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    train_dataset: A2DProcessedWindowDataset,
) -> ActionPolicy:
    return build_policy(
        policy_cfg=policy_cfg,
        model_cfg=model_cfg,
        image_keys=tuple(data_cfg.get("image_keys", ["rgb_head", "rgb_left_hand", "rgb_right_hand"])),
        action_dim=train_dataset.action_dim,
        history_steps=train_dataset.history_steps,
        action_horizon=train_dataset.action_horizon,
    )


def _build_model(
    *,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    train_dataset: A2DProcessedWindowDataset,
) -> ActionPolicy:
    """Backward-compatible helper for existing scripts and old configs."""
    return _build_policy(
        policy_cfg={"type": "flow_matching"},
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_dataset=train_dataset,
    )


def _sample_normalized_actions(model: ActionPolicy, batch: dict[str, Any]) -> torch.Tensor:
    return model.sample_actions(batch["obs"]).action_normalized


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_template(template: str, *, timestamp: str, config_name: str, run_name: str) -> str:
    return template.format(timestamp=timestamp, config_name=config_name, run_name=run_name)


def _resolve_output_dir(training_cfg: dict[str, Any], *, config_path: str) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(config_path).stem
    run_name_template = str(training_cfg.get("run_name", "{config_name}_{timestamp}"))
    run_name = _format_template(
        run_name_template,
        timestamp=timestamp,
        config_name=config_name,
        run_name="",
    )

    output_dir_value = training_cfg.get("output_dir")
    if output_dir_value is None:
        output_root = Path(str(training_cfg.get("output_root", "outputs"))).expanduser()
        output_dir = output_root / run_name
    else:
        output_dir = Path(
            _format_template(
                str(output_dir_value),
                timestamp=timestamp,
                config_name=config_name,
                run_name=run_name,
            )
        ).expanduser()
    return output_dir.resolve(), run_name


def _build_rerun_visualizer(
    cfg: dict[str, Any],
    *,
    config_path: str,
    output_dir: Path,
    run_name: str,
) -> tuple[RerunTrainVisualizer, dict[str, Any]]:
    vis_cfg = cfg.get("visualization", {})
    rerun_cfg = vis_cfg.get("rerun", {}) if isinstance(vis_cfg, dict) else {}
    enabled = bool(rerun_cfg.get("enabled", False))
    save_path_value = rerun_cfg.get("save_path")
    if save_path_value is None:
        save_path = output_dir / "training.rrd"
    else:
        save_path = Path(
            _format_template(
                str(save_path_value),
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                config_name=Path(config_path).stem,
                run_name=run_name,
            )
        ).expanduser()
        if not save_path.is_absolute():
            save_path = output_dir / save_path

    visualizer = RerunTrainVisualizer(
        enabled=enabled,
        app_name=str(rerun_cfg.get("app_name", f"flow-matching-train-{run_name}")),
        spawn=bool(rerun_cfg.get("spawn", False)),
        save_path=str(save_path),
    )
    resolved = {
        "enabled": bool(visualizer.enabled),
        "spawn": bool(rerun_cfg.get("spawn", False)),
        "save_path": str(save_path.resolve()),
        "sample_source": str(rerun_cfg.get("sample_source", "val")),
        "sample_every_n_epochs": int(rerun_cfg.get("sample_every_n_epochs", 1)),
    }
    return visualizer, resolved


def _build_wandb_logger(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    run_name: str,
) -> tuple[WandbLogger, dict[str, Any]]:
    logging_cfg = cfg.get("logging", {})
    wandb_cfg = logging_cfg.get("wandb", {}) if isinstance(logging_cfg, dict) else {}
    enabled = bool(wandb_cfg.get("enabled", False))
    logger = WandbLogger(
        enabled=enabled,
        output_dir=str(output_dir),
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        run_name=str(wandb_cfg.get("run_name", run_name)),
        mode=str(wandb_cfg.get("mode", "offline")),
        tags=[str(item) for item in wandb_cfg.get("tags", [])],
        group=wandb_cfg.get("group"),
        job_type=wandb_cfg.get("job_type", "train"),
        config=cfg,
    )
    run_url = logger.run_url
    resolved = {
        "enabled": bool(logger.enabled),
        "project": wandb_cfg.get("project"),
        "entity": wandb_cfg.get("entity"),
        "run_name": str(wandb_cfg.get("run_name", run_name)),
        "mode": logger.mode,
        "tags": [str(item) for item in wandb_cfg.get("tags", [])],
        "group": wandb_cfg.get("group"),
        "job_type": wandb_cfg.get("job_type", "train"),
        "run_url": run_url,
    }
    return logger, resolved


@torch.no_grad()
def evaluate(
    *,
    model: ActionPolicy,
    loader: DataLoader,
    device: torch.device,
    max_steps: int | None = None,
    heartbeat: Callable[[int], None] | None = None,
    deterministic_seed: int | None = None,
    sample_draws: int = 1,
) -> dict[str, float]:
    if sample_draws < 1:
        raise ValueError("sample_draws must be >= 1")
    model.eval()
    losses: list[float] = []
    component_values: dict[str, list[float]] = defaultdict(list)
    sample_mse: list[float] = []
    for step_idx, batch in enumerate(loader):
        if max_steps is not None and step_idx >= max_steps:
            break
        if heartbeat is not None:
            heartbeat(step_idx)
        batch = _to_device(batch, device)
        sample_indices = batch.get("sample_index")
        if deterministic_seed is not None:
            if not isinstance(sample_indices, torch.Tensor):
                raise KeyError("deterministic validation requires batch['sample_index']")
            indices = [int(value) for value in sample_indices.detach().cpu().tolist()]
            loss_seeds = [
                (int(deterministic_seed) * 1_000_003 + index * 100_003 + 17) % (2**63 - 1)
                for index in indices
            ]
        else:
            indices = []
            loss_seeds = []
        if deterministic_seed is not None and type(model) is FlowMatchingPolicy:
            loss, components = model.compute_loss_seeded(batch, loss_seeds)
        else:
            loss, components = model.compute_loss(batch)
        losses.append(float(loss.detach().item()))
        for name, value in components.items():
            if value is not None:
                component_values[name].append(float(value))
        draw_mse = []
        for draw in range(sample_draws):
            if deterministic_seed is not None and type(model) is FlowMatchingPolicy:
                sample_seeds = [
                    (
                        int(deterministic_seed) * 1_000_003
                        + index * 100_003
                        + (draw + 1) * 10_007
                    )
                    % (2**63 - 1)
                    for index in indices
                ]
                sampled = model.sample_actions_seeded(batch["obs"], sample_seeds).action_normalized
            else:
                sampled = _sample_normalized_actions(model, batch)
            draw_mse.append(float(torch.mean((sampled - batch["action"]) ** 2).item()))
        sample_mse.append(float(np.mean(draw_mse)))
    if not losses:
        return {"loss": 0.0, "sample_action_mse": 0.0}
    return {
        "loss": float(np.mean(losses)),
        **{name: float(np.mean(values)) for name, values in component_values.items()},
        "sample_action_mse": float(np.mean(sample_mse)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RGB-conditioned flow matching trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("overrides", nargs="*", help="Optional key=value overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    # Materialize the temporal contract so checkpoints and exported bundles never
    # have to guess whether chunk[0] means action[t] or action[t+1].
    data_cfg["action_offset_steps"] = int(data_cfg.get("action_offset_steps", 1))
    data_cfg["max_open_hdf5_files"] = int(data_cfg.get("max_open_hdf5_files", 64))
    training_cfg["watchdog_timeout_sec"] = float(training_cfg.get("watchdog_timeout_sec", 300.0))
    training_cfg["watchdog_check_interval_sec"] = float(
        training_cfg.get("watchdog_check_interval_sec", 10.0)
    )
    model_cfg = cfg["model"]
    policy_cfg = materialize_policy_config(
        copy.deepcopy(cfg.get("policy", {"type": "flow_matching"}))
    )
    policy_type = resolve_policy_type(policy_cfg)
    cfg["policy"] = policy_cfg
    started_at = time.time()

    seed = int(training_cfg.get("seed", 42))
    validation_cfg = cfg.setdefault("validation", {})
    validation_cfg["deterministic"] = bool(
        validation_cfg.get("deterministic", policy_type == "flow_matching")
    )
    validation_cfg["seed"] = int(validation_cfg.get("seed", seed))
    validation_cfg["sample_draws"] = int(validation_cfg.get("sample_draws", 3))
    if validation_cfg["sample_draws"] < 1:
        raise ValueError("validation.sample_draws must be >= 1")
    if validation_cfg["deterministic"] and policy_type != "flow_matching":
        raise ValueError("deterministic validation is currently implemented only for flow_matching")
    training_cfg["ema_enabled"] = bool(training_cfg.get("ema_enabled", True))
    training_cfg["ema_decay"] = float(training_cfg.get("ema_decay", 0.999))
    if not 0.0 <= training_cfg["ema_decay"] < 1.0:
        raise ValueError("training.ema_decay must be in [0,1)")
    configured_bundle_variant = str(
        cfg.get("deployment", {}).get("eval_bundle", {}).get("weights_variant", "raw")
    ).strip().lower()
    if configured_bundle_variant == "ema" and not training_cfg["ema_enabled"]:
        raise ValueError("EMA bundle export requires training.ema_enabled=true")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir, run_name = _resolve_output_dir(training_cfg, config_path=args.config)
    training_cfg["output_dir"] = str(output_dir)
    training_cfg["run_name"] = run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    rerun_visualizer, rerun_info = _build_rerun_visualizer(
        cfg,
        config_path=args.config,
        output_dir=output_dir,
        run_name=run_name,
    )
    device = torch.device(str(training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    train_dataset = _build_dataset(data_cfg=data_cfg, split="train", seed=seed)
    val_dataset = _build_dataset(data_cfg=data_cfg, split="val", seed=seed)

    val_dataset.action_mean = train_dataset.action_mean.copy()
    val_dataset.action_std = train_dataset.action_std.copy()

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", False)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", False)),
    )

    model = _build_policy(
        policy_cfg=policy_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_dataset=train_dataset,
    ).to(device)
    freeze_encoder_backbone = bool(model_cfg.get("freeze_encoder_backbone", False))
    backbone_parameter_count = _set_backbone_frozen(
        model,
        frozen=freeze_encoder_backbone,
    )
    stats = train_dataset.export_stats()
    model.set_action_stats(action_mean=stats["action_mean"].to(device), action_std=stats["action_std"].to(device))

    base_lr = float(training_cfg.get("lr", 1.0e-4))
    backbone_lr_multiplier = float(training_cfg.get("backbone_lr_multiplier", 1.0))
    parameter_groups, head_params, backbone_params = model.optimizer_parameter_groups(
        base_lr=base_lr,
        backbone_lr_multiplier=backbone_lr_multiplier,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=base_lr,
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-4)),
        betas=tuple(training_cfg.get("betas", [0.9, 0.95])),
    )
    steps_per_epoch = min(len(train_loader), int(training_cfg.get("max_train_steps") or len(train_loader)))
    total_steps = max(1, steps_per_epoch * int(training_cfg.get("num_epochs", 50)))
    configured_warmup = training_cfg.get("warmup_steps")
    warmup_steps = int(configured_warmup) if configured_warmup is not None else round(total_steps * 0.05)

    def lr_scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)

    log_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    experiment_index_path = output_dir.parent / "experiment_results.jsonl"
    resume_events_path = output_dir / "resume_events.jsonl"
    best_val = float("inf")
    best_action_mse = float("inf")
    best_ema_action_mse = float("inf")
    best_model_state = None
    best_action_mse_model_state = None
    best_metrics: dict[str, Any] | None = None
    best_action_mse_metrics: dict[str, Any] | None = None
    best_ema_action_mse_metrics: dict[str, Any] | None = None
    last_metrics: dict[str, Any] | None = None
    ema_model: ActionPolicy | None = None
    resume_ema_state: dict[str, Any] | None = None
    global_step = 0
    start_epoch = 0
    max_train_steps = training_cfg.get("max_train_steps")
    max_val_steps = training_cfg.get("max_val_steps")
    max_epochs_this_run = training_cfg.get("max_epochs_this_run")
    epochs_completed_this_run = 0

    split_manifest = copy.deepcopy(train_dataset.split_manifest)
    data_provenance = {
        "stats_digest": train_dataset.stats["train_episode_digest"],
        "segmentation_version": SEGMENTATION_VERSION,
        "segmentation_motion_threshold": float(data_cfg.get("motion_threshold", 1.0e-4)),
        "segmentation_keyframe_threshold": float(data_cfg.get("transition_threshold", 0.1)),
        "action_offset_steps": train_dataset.cfg.action_offset_steps,
        "dataset_manifest_sha256": split_manifest.get("dataset_manifest_sha256"),
        "split_manifest_sha256": split_manifest.get("split_manifest_sha256"),
    }
    wandb_config = copy.deepcopy(cfg)
    wandb_config["provenance"] = {
        "git_sha": _git_sha(Path(__file__).resolve().parents[1]),
        "dataset_version": cfg.get("deployment", {}).get("eval_bundle", {}).get(
            "data_version", "unknown"
        ),
        **data_provenance,
        "split_manifest": split_manifest,
    }
    rng_before_wandb = _rng_state()
    wandb_logger, wandb_info = _build_wandb_logger(
        wandb_config,
        output_dir=output_dir,
        run_name=run_name,
    )
    _restore_rng_state(rng_before_wandb)

    failure_reported = threading.Event()

    def report_failure(*, kind: str, message: str, stage: str, step: int) -> None:
        if failure_reported.is_set():
            return
        failure_reported.set()
        payload = {
            "run_name": run_name,
            "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "kind": kind,
            "message": message,
            "stage": stage,
            "global_step": int(step),
        }
        failure_path = output_dir / "failure.json"
        temporary = failure_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(failure_path)
        print("TRAINING_FAILURE " + json.dumps(payload, ensure_ascii=False), flush=True)
        wandb_logger.alert(
            title=f"Training failed: {run_name}",
            text=f"{kind} during {stage} at step {step}:\n{message[:4000]}",
            level="ERROR",
        )
        wandb_logger.update_summary({"status": "failed", "failure": payload})
        wandb_logger.finish(exit_code=1)

    original_excepthook = sys.excepthook

    def report_unhandled_exception(exc_type, exc_value, exc_traceback) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            report_failure(
                kind="unhandled_exception",
                message="".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
                stage="main_thread",
                step=global_step,
            )
        original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = report_unhandled_exception

    def report_stall(details: StallDetails) -> None:
        report_failure(
            kind="watchdog_timeout",
            message=details.message,
            stage=details.stage,
            step=details.step,
        )

    watchdog = TrainingWatchdog(
        timeout_seconds=float(training_cfg["watchdog_timeout_sec"]),
        check_interval_seconds=float(training_cfg["watchdog_check_interval_sec"]),
        on_stall=report_stall,
    )

    def checkpoint_payload(
        model_state: dict[str, Any],
        epoch: int,
        *,
        selection_criterion: str,
    ) -> dict[str, Any]:
        return {
            "model_state_dict": model_state,
            "selection_criterion": selection_criterion,
            "policy_type": policy_type,
            "policy_spec": {"type": policy_type},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
            "global_step": global_step,
            "epoch": epoch,
            "best_val": best_val,
            "best_action_mse": best_action_mse,
            "best_ema_action_mse": best_ema_action_mse,
            "best_metrics": best_metrics,
            "best_action_mse_metrics": best_action_mse_metrics,
            "best_ema_action_mse_metrics": best_ema_action_mse_metrics,
            "last_metrics": last_metrics,
            "best_model_state_dict": best_model_state,
            "best_action_mse_model_state_dict": best_action_mse_model_state,
            "ema_model_state_dict": (
                ema_model.state_dict() if ema_model is not None else None
            ),
            "rng_state": _rng_state(),
            "action_mean": train_dataset.action_mean,
            "action_std": train_dataset.action_std,
            "normalizer": copy.deepcopy(train_dataset.stats),
            "stats_digest": train_dataset.stats["train_episode_digest"],
            "split_manifest": split_manifest,
            "data_provenance": data_provenance,
            "training_environment": _runtime_environment(),
            "model_schema": {
                "obs_keys": list(data_cfg.get("image_keys", []))
                + (["proprio"] if model_cfg.get("use_proprio", True) else []),
                "action_keys": ["arm2_pos", "hand2_pos"],
            },
            "model_spec": {
                "policy_type": policy_type,
                "obs_horizon": train_dataset.history_steps,
                "action_dim": train_dataset.action_dim,
                "action_horizon": train_dataset.action_horizon,
                "action_offset_steps": train_dataset.cfg.action_offset_steps,
                "action_layout": [
                    {"name": "arm2_pos", "dim": 7},
                    {"name": "hand2_pos", "dim": 6},
                ],
            },
            "adapter_metadata": {
                "dataset": "A2DProcessedWindowDataset",
                "action_semantics": "executed_joint_position",
            },
            "inference_spec": {
                "schema_version": 1,
                "image_keys": list(data_cfg.get("image_keys", [])),
                "use_proprio": bool(model_cfg.get("use_proprio", True)),
                "num_inference_steps": int(model_cfg.get("num_inference_steps", 40)),
            },
        }

    init_from = training_cfg.get("init_from")
    resume_from = training_cfg.get("resume_from")
    if init_from and resume_from:
        raise ValueError("training.init_from and training.resume_from are mutually exclusive")
    if init_from:
        init_path = Path(str(init_from)).expanduser().resolve()
        checkpoint = torch.load(init_path, map_location=device, weights_only=False)
        checkpoint_policy_type = resolve_policy_type(
            checkpoint.get("policy_spec", {"type": checkpoint.get("policy_type", "flow_matching")})
        )
        if checkpoint_policy_type != policy_type:
            raise ValueError(
                f"init checkpoint policy_type={checkpoint_policy_type!r} does not match "
                f"current policy_type={policy_type!r}"
            )
        checkpoint_provenance = checkpoint.get("data_provenance", {})
        checkpoint_action_offset = int(
            checkpoint_provenance.get(
                "action_offset_steps",
                checkpoint.get("config", {}).get("data", {}).get("action_offset_steps", 0),
            )
        )
        if checkpoint_action_offset != data_provenance["action_offset_steps"]:
            raise ValueError(
                "init checkpoint action_offset_steps does not match the current dataset"
            )
        for key in ("stats_digest", "dataset_manifest_sha256", "split_manifest_sha256"):
            if checkpoint_provenance.get(key) != data_provenance.get(key):
                raise ValueError(f"init checkpoint {key} does not match the current dataset")
        model.load_state_dict(checkpoint["model_state_dict"])
        init_event = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "checkpoint": str(init_path),
            "checkpoint_sha256": _sha256(init_path),
            "source_epoch": int(checkpoint.get("epoch", -1)),
            "source_global_step": int(checkpoint.get("global_step", 0)),
            "optimizer_scheduler_reset": True,
        }
        _append_jsonl(output_dir / "init_events.jsonl", init_event)
        print("INIT_MODEL_ONLY_OK " + json.dumps(init_event, ensure_ascii=False))
    if resume_from:
        resume_path = Path(str(resume_from)).expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        checkpoint_policy_type = resolve_policy_type(
            checkpoint.get("policy_spec", {"type": checkpoint.get("policy_type", "flow_matching")})
        )
        if checkpoint_policy_type != policy_type:
            raise ValueError(
                f"resume checkpoint policy_type={checkpoint_policy_type!r} does not match "
                f"current policy_type={policy_type!r}"
            )
        checkpoint_provenance = checkpoint.get("data_provenance", {})
        checkpoint_action_offset = int(
            checkpoint_provenance.get(
                "action_offset_steps",
                checkpoint.get("config", {}).get("data", {}).get("action_offset_steps", 0),
            )
        )
        if checkpoint_action_offset != data_provenance["action_offset_steps"]:
            raise ValueError(
                "resume checkpoint action_offset_steps does not match the current dataset"
            )
        for key in ("stats_digest", "dataset_manifest_sha256", "split_manifest_sha256"):
            if checkpoint_provenance.get(key) != data_provenance.get(key):
                raise ValueError(f"resume checkpoint {key} does not match the current dataset")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", float("inf")))
        checkpoint_best_mse_metrics = checkpoint.get("best_action_mse_metrics")
        if checkpoint_best_mse_metrics is None:
            checkpoint_best_mse_metrics = checkpoint.get("best_metrics")
        best_action_mse = float(
            checkpoint.get(
                "best_action_mse",
                (
                    checkpoint_best_mse_metrics.get("val_sample_action_mse", float("inf"))
                    if checkpoint_best_mse_metrics is not None
                    else float("inf")
                ),
            )
        )
        best_metrics = copy.deepcopy(checkpoint.get("best_metrics"))
        best_action_mse_metrics = copy.deepcopy(checkpoint_best_mse_metrics)
        best_ema_action_mse = float(
            checkpoint.get("best_ema_action_mse", float("inf"))
        )
        best_ema_action_mse_metrics = copy.deepcopy(
            checkpoint.get("best_ema_action_mse_metrics")
        )
        last_metrics = copy.deepcopy(checkpoint.get("last_metrics"))
        best_model_state = copy.deepcopy(checkpoint.get("best_model_state_dict"))
        best_action_mse_model_state = copy.deepcopy(
            checkpoint.get("best_action_mse_model_state_dict")
        )
        if best_action_mse_model_state is None:
            best_action_mse_model_state = copy.deepcopy(best_model_state)
        resume_ema_state = checkpoint.get("ema_model_state_dict")
        _restore_rng_state(checkpoint["rng_state"])
        resume_event = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "checkpoint": str(resume_path),
            "checkpoint_sha256": _sha256(resume_path),
            "resume_from_epoch": int(checkpoint["epoch"]),
            "resume_from_global_step": global_step,
            "new_process_id": os.getpid(),
        }
        _append_jsonl(resume_events_path, resume_event)
        print("RESUME_OK " + json.dumps(resume_event, ensure_ascii=False))

    if training_cfg["ema_enabled"]:
        ema_model = copy.deepcopy(model).to(device)
        if resume_ema_state is not None:
            ema_model.load_state_dict(resume_ema_state)
        ema_model.requires_grad_(False)
        ema_model.eval()

    watchdog.heartbeat(stage="training_start", step=global_step)
    watchdog.start()
    for epoch in range(start_epoch, int(training_cfg.get("num_epochs", 50))):
        watchdog.heartbeat(stage=f"train_epoch_{epoch}", step=global_step)
        train_dataset.set_epoch(epoch)
        model.train()
        epoch_losses: list[float] = []
        epoch_components: dict[str, list[float]] = defaultdict(list)
        epoch_grad_norm_head: list[float] = []
        epoch_grad_norm_backbone: list[float] = []
        epoch_encoder_grad_param_ratio: list[float] = []
        epoch_grad_clip_triggered: list[float] = []
        encoder_reference = _snapshot_parameters(backbone_params)
        encoder_param_norm = _parameter_l2_norm(backbone_params)

        for batch_idx, batch in enumerate(train_loader):
            if max_train_steps is not None and batch_idx >= int(max_train_steps):
                break
            watchdog.heartbeat(stage=f"train_epoch_{epoch}", step=global_step)
            batch = _to_device(batch, device)
            loss, components = model.compute_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            epoch_grad_norm_head.append(_grad_l2_norm(head_params))
            backbone_grad_norm = _grad_l2_norm(backbone_params)
            epoch_grad_norm_backbone.append(backbone_grad_norm)
            epoch_encoder_grad_param_ratio.append(
                backbone_grad_norm / max(encoder_param_norm, 1.0e-12)
            )
            if training_cfg.get("grad_clip") is not None:
                grad_clip = float(training_cfg["grad_clip"])
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                epoch_grad_clip_triggered.append(float(total_grad_norm > grad_clip))
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                _update_ema_model(
                    ema_model,
                    model,
                    decay=float(training_cfg["ema_decay"]),
                )

            epoch_losses.append(float(loss.detach().item()))
            for name, value in components.items():
                if value is not None:
                    epoch_components[name].append(float(value))
            global_step += 1

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            max_steps=int(max_val_steps) if max_val_steps is not None else None,
            heartbeat=lambda step: watchdog.heartbeat(
                stage=f"val_epoch_{epoch}", step=step
            ),
            deterministic_seed=(
                int(validation_cfg["seed"]) if validation_cfg["deterministic"] else None
            ),
            sample_draws=int(validation_cfg["sample_draws"]),
        )
        ema_val_metrics = None
        if ema_model is not None:
            rng_before_ema_evaluation = _rng_state()
            try:
                ema_val_metrics = evaluate(
                    model=ema_model,
                    loader=val_loader,
                    device=device,
                    max_steps=int(max_val_steps) if max_val_steps is not None else None,
                    heartbeat=lambda step: watchdog.heartbeat(
                        stage=f"ema_val_epoch_{epoch}", step=step
                    ),
                    deterministic_seed=(
                        int(validation_cfg["seed"])
                        if validation_cfg["deterministic"]
                        else None
                    ),
                    sample_draws=int(validation_cfg["sample_draws"]),
                )
            finally:
                _restore_rng_state(rng_before_ema_evaluation)
        watchdog.heartbeat(stage=f"epoch_{epoch}_diagnostics", step=global_step)
        train_sample_loader = DataLoader(train_dataset, batch_size=min(16, int(training_cfg.get("batch_size", 64))), shuffle=False)
        train_sample_batch = next(iter(train_sample_loader))
        train_sample_batch = _to_device(train_sample_batch, device)
        train_sample = _sample_normalized_actions(model, train_sample_batch)
        train_sample_mse = float(torch.mean((train_sample - train_sample_batch["action"]) ** 2).item())
        encoder_update_ratio = _parameter_update_ratio(backbone_params, encoder_reference)
        encoder_feature_std = _encoder_feature_std(model, train_sample_batch["obs"])

        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            **{
                f"train_{name}": float(np.mean(values)) if values else None
                for name, values in epoch_components.items()
            },
            "train_sample_action_mse": train_sample_mse,
            "head_lr": float(optimizer.param_groups[0]["lr"]),
            "backbone_lr": float(optimizer.param_groups[1]["lr"]),
            "grad_norm_head_mean": float(np.mean(epoch_grad_norm_head)),
            "grad_norm_head_max": float(np.max(epoch_grad_norm_head)),
            "grad_norm_backbone_mean": float(np.mean(epoch_grad_norm_backbone)),
            "grad_norm_backbone_max": float(np.max(epoch_grad_norm_backbone)),
            "grad_clip_trigger_rate": (
                float(np.mean(epoch_grad_clip_triggered))
                if epoch_grad_clip_triggered
                else 0.0
            ),
            "encoder_update_ratio": encoder_update_ratio,
            "encoder_grad_param_ratio_mean": float(np.mean(epoch_encoder_grad_param_ratio)),
            "encoder_feature_std": encoder_feature_std,
            "encoder_backbone_frozen": int(freeze_encoder_backbone),
            "val_loss": float(val_metrics["loss"]),
            **{
                f"val_{name}": value
                for name, value in val_metrics.items()
                if name not in {"loss", "sample_action_mse"}
            },
            "val_sample_action_mse": float(val_metrics["sample_action_mse"]),
            **(
                {
                    "ema_val_loss": float(ema_val_metrics["loss"]),
                    **{
                        f"ema_val_{name}": value
                        for name, value in ema_val_metrics.items()
                        if name not in {"loss", "sample_action_mse"}
                    },
                    "ema_val_sample_action_mse": float(
                        ema_val_metrics["sample_action_mse"]
                    ),
                }
                if ema_val_metrics is not None
                else {}
            ),
        }
        last_metrics = copy.deepcopy(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        _append_jsonl(log_path, metrics)
        rerun_visualizer.log_epoch_metrics(
            epoch=epoch,
            global_step=global_step,
            metrics={
                key: (float(value) if value is not None else None)
                for key, value in metrics.items()
                if key not in {"epoch", "global_step"}
            },
        )
        wandb_logger.log_metrics(metrics, step=global_step)

        sample_every_n_epochs = max(1, int(rerun_info["sample_every_n_epochs"]))
        if rerun_visualizer.enabled and (epoch % sample_every_n_epochs == 0):
            sample_source = str(rerun_info["sample_source"]).strip().lower()
            if sample_source == "train":
                sample_batch = train_sample_batch
                sample_split = "train"
            else:
                val_sample_batch = next(iter(val_loader))
                sample_batch = _to_device(val_sample_batch, device)
                sample_split = "val"
            sample_prediction = model.sample_actions(sample_batch["obs"])
            rerun_visualizer.log_sample(
                sample_index=epoch,
                split=sample_split,
                obs=sample_batch["obs"],
                gt_action_normalized=sample_batch["action"],
                pred_action_normalized=sample_prediction.action_normalized,
                gt_action=model.denormalize_action(sample_batch["action"]),
                pred_action=sample_prediction.action,
            )

        improved_val_loss = metrics["val_loss"] < best_val
        improved_action_mse = metrics["val_sample_action_mse"] < best_action_mse
        improved_ema_action_mse = (
            ema_val_metrics is not None
            and metrics["ema_val_sample_action_mse"] < best_ema_action_mse
        )
        if improved_val_loss:
            best_val = metrics["val_loss"]
            best_metrics = copy.deepcopy(metrics)
            best_model_state = copy.deepcopy(model.state_dict())
        if improved_action_mse:
            best_action_mse = metrics["val_sample_action_mse"]
            best_action_mse_metrics = copy.deepcopy(metrics)
            best_action_mse_model_state = copy.deepcopy(model.state_dict())
        if improved_ema_action_mse:
            best_ema_action_mse = metrics["ema_val_sample_action_mse"]
            best_ema_action_mse_metrics = copy.deepcopy(metrics)

        latest_payload = checkpoint_payload(
            model.state_dict(),
            epoch,
            selection_criterion="latest",
        )
        _atomic_torch_save(latest_payload, output_dir / "latest.ckpt")

        if improved_val_loss:
            checkpoint_path = output_dir / "best.ckpt"
            val_loss_payload = checkpoint_payload(
                best_model_state,
                epoch,
                selection_criterion="val_loss",
            )
            _atomic_torch_save(val_loss_payload, checkpoint_path)
            _atomic_torch_save(
                val_loss_payload,
                output_dir / "best_val_loss.ckpt",
            )
        if improved_action_mse:
            action_mse_checkpoint_path = output_dir / "best_action_mse.ckpt"
            _atomic_torch_save(
                checkpoint_payload(
                    best_action_mse_model_state,
                    epoch,
                    selection_criterion="val_sample_action_mse",
                ),
                action_mse_checkpoint_path,
            )
        if improved_ema_action_mse:
            _atomic_torch_save(
                checkpoint_payload(
                    model.state_dict(),
                    epoch,
                    selection_criterion="ema_val_sample_action_mse",
                ),
                output_dir / "best_ema_action_mse.ckpt",
            )

        export_cfg = cfg.get("deployment", {}).get("eval_bundle", {})
        export_variant = str(export_cfg.get("weights_variant", "raw")).strip().lower()
        if export_variant not in {"raw", "ema"}:
            raise ValueError("deployment.eval_bundle.weights_variant must be 'raw' or 'ema'")
        should_export = (
            improved_action_mse if export_variant == "raw" else improved_ema_action_mse
        )
        if bool(export_cfg.get("enabled", False)) and should_export:
            selected_checkpoint = output_dir / (
                "best_action_mse.ckpt"
                if export_variant == "raw"
                else "best_ema_action_mse.ckpt"
            )
            bundle_dir = (
                output_dir
                / "eval_bundles"
                / f"eval_bundle_{run_name}_step{global_step}"
            )
            camera_resolutions = {
                str(name): tuple(int(value) for value in resolution)
                for name, resolution in export_cfg.get("camera_resolutions", {}).items()
            }
            exported = export_eval_bundle(
                selected_checkpoint,
                bundle_dir,
                execute_horizon=int(
                    export_cfg.get("execute_horizon", train_dataset.action_horizon)
                ),
                camera_resolutions=camera_resolutions,
                joint_order=list(export_cfg.get("joint_order", DEFAULT_JOINT_ORDER)),
                data_version=str(export_cfg.get("data_version", "unknown")),
                archive=bool(export_cfg.get("archive", True)),
                weights_variant=export_variant,
            )
            print(f"[eval_bundle] {exported}")

        epochs_completed_this_run += 1
        if max_epochs_this_run is not None and epochs_completed_this_run >= int(max_epochs_this_run):
            print(f"RUN_EPOCH_LIMIT_REACHED epochs_completed={epochs_completed_this_run}")
            break

    if best_model_state is None:
        raise RuntimeError("Training finished without producing a checkpoint")
    if best_action_mse_model_state is None:
        raise RuntimeError("Training finished without producing an action-MSE checkpoint")
    if last_metrics is None:
        raise RuntimeError("Training finished without producing any metrics")

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    final_metrics = last_metrics
    summary = {
        "run_name": str(training_cfg.get("run_name", output_dir.name)),
        "output_dir": str(output_dir),
        "finished_at": finished_at,
        "duration_sec": round(time.time() - started_at, 3),
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_base_samples": train_dataset.base_sample_count,
        "train_transition_samples": train_dataset.transition_sample_count,
        "train_effective_transition_samples": train_dataset.effective_transition_sample_count,
        "data_dir": str(data_cfg["data_dir"]),
        "image_keys": list(data_cfg.get("image_keys", ["rgb_head", "rgb_left_hand", "rgb_right_hand"])),
        "action_dim": train_dataset.action_dim,
        "history_steps": train_dataset.history_steps,
        "action_horizon": train_dataset.action_horizon,
        "action_offset_steps": train_dataset.cfg.action_offset_steps,
        "policy_type": policy_type,
        "encoder_type": str(model_cfg.get("encoder_type", "cnn")),
        "timm_model_name": str(model_cfg.get("timm_model_name", "vit_small_r26_s32_224")),
        "timm_pretrained": bool(model_cfg.get("timm_pretrained", True)),
        "freeze_encoder_backbone": freeze_encoder_backbone,
        "backbone_parameter_count": backbone_parameter_count,
        "timm_tokens_per_frame": int(model_cfg.get("timm_tokens_per_frame", 1)),
        "timm_token_mode": str(model_cfg.get("timm_token_mode", "spatial")),
        "use_proprio": bool(model_cfg.get("use_proprio", True)),
        "batch_size": int(training_cfg.get("batch_size", 64)),
        "lr": float(training_cfg.get("lr", 1.0e-4)),
        "backbone_lr_multiplier": backbone_lr_multiplier,
        "weight_decay": float(training_cfg.get("weight_decay", 1.0e-4)),
        "num_epochs": int(training_cfg.get("num_epochs", 50)),
        "max_train_steps": max_train_steps,
        "max_val_steps": max_val_steps,
        "d_model": int(model_cfg.get("d_model", 128)),
        "n_head": int(model_cfg.get("n_head", 4)),
        "n_layer": int(model_cfg.get("n_layer", 4)),
        "num_inference_steps": int(model_cfg.get("num_inference_steps", 40)),
        "time_eps": float(model_cfg.get("time_eps", 1.0e-3)),
        "ema_enabled": bool(training_cfg["ema_enabled"]),
        "ema_decay": float(training_cfg["ema_decay"]),
        "validation": copy.deepcopy(validation_cfg),
        "rerun": rerun_info,
        "wandb": wandb_info,
        "best_epoch": int(best_metrics["epoch"]) if best_metrics is not None else None,
        "best_val_loss": float(best_metrics["val_loss"]) if best_metrics is not None else None,
        "best_action_mse_epoch": (
            int(best_action_mse_metrics["epoch"])
            if best_action_mse_metrics is not None
            else None
        ),
        "best_val_sample_action_mse": (
            float(best_action_mse_metrics["val_sample_action_mse"])
            if best_action_mse_metrics is not None
            else None
        ),
        "best_ema_action_mse_epoch": (
            int(best_ema_action_mse_metrics["epoch"])
            if best_ema_action_mse_metrics is not None
            else None
        ),
        "best_ema_val_sample_action_mse": (
            float(best_ema_action_mse_metrics["ema_val_sample_action_mse"])
            if best_ema_action_mse_metrics is not None
            else None
        ),
        "best": best_metrics,
        "best_action_mse": best_action_mse_metrics,
        "best_ema_action_mse": best_ema_action_mse_metrics,
        "final": final_metrics,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _append_jsonl(experiment_index_path, summary)
    wandb_logger.update_summary(summary)
    wandb_logger.save_text("summary.json", summary_path.read_text(encoding="utf-8"))
    wandb_logger.save_text("config_resolved.yaml", (output_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    watchdog.stop()
    sys.excepthook = original_excepthook
    wandb_logger.finish()


if __name__ == "__main__":
    main()
