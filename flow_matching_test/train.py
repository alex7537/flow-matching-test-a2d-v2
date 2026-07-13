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
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from flow_matching_test.a2d_dataset import A2DConfig, A2DProcessedWindowDataset
from flow_matching_test.export_bundle import DEFAULT_JOINT_ORDER, export_eval_bundle
from flow_matching_test.model import RGBConditionedFlowModel
from flow_matching_test.rerun_logger import RerunTrainVisualizer
from flow_matching_test.segmentation import SEGMENTATION_VERSION
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
        range_eps=float(data_cfg.get("range_eps", 1.0e-4)),
        norm_stats=str(data_cfg.get("norm_stats", "norm_stats.json")),
        dataset_manifest=data_cfg.get("dataset_manifest"),
        split_manifest=data_cfg.get("split_manifest"),
    )
    return A2DProcessedWindowDataset(cfg=cfg, split=split)


def _build_model(*, model_cfg: dict[str, Any], data_cfg: dict[str, Any], train_dataset: A2DProcessedWindowDataset) -> RGBConditionedFlowModel:
    return RGBConditionedFlowModel(
        image_keys=tuple(data_cfg.get("image_keys", ["rgb_head", "rgb_left_hand", "rgb_right_hand"])),
        encoder_type=str(model_cfg.get("encoder_type", "cnn")),
        timm_model_name=str(model_cfg.get("timm_model_name", "vit_small_r26_s32_224")),
        timm_pretrained=bool(model_cfg.get("timm_pretrained", True)),
        timm_tokens_per_frame=int(model_cfg.get("timm_tokens_per_frame", 1)),
        timm_token_mode=str(model_cfg.get("timm_token_mode", "spatial")),
        use_proprio=bool(model_cfg.get("use_proprio", True)),
        action_dim=train_dataset.action_dim,
        history_steps=train_dataset.history_steps,
        action_horizon=train_dataset.action_horizon,
        d_model=int(model_cfg.get("d_model", 128)),
        n_head=int(model_cfg.get("n_head", 4)),
        n_layer=int(model_cfg.get("n_layer", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        time_eps=float(model_cfg.get("time_eps", 1.0e-3)),
        num_inference_steps=int(model_cfg.get("num_inference_steps", 40)),
    )

def _sample_normalized_actions(model: RGBConditionedFlowModel, batch: dict[str, Any]) -> torch.Tensor:
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
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


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
        mode=str(wandb_cfg.get("mode", "online")),
        tags=[str(item) for item in wandb_cfg.get("tags", [])],
        group=wandb_cfg.get("group"),
        job_type=wandb_cfg.get("job_type", "train"),
        config=cfg,
    )
    resolved = {
        "enabled": bool(logger.enabled),
        "project": wandb_cfg.get("project"),
        "entity": wandb_cfg.get("entity"),
        "run_name": str(wandb_cfg.get("run_name", run_name)),
        "mode": str(wandb_cfg.get("mode", "online")),
        "tags": [str(item) for item in wandb_cfg.get("tags", [])],
        "group": wandb_cfg.get("group"),
        "job_type": wandb_cfg.get("job_type", "train"),
        "run_url": logger.run_url,
    }
    return logger, resolved


@torch.no_grad()
def evaluate(
    *,
    model: RGBConditionedFlowModel,
    loader: DataLoader,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    segmented_losses: dict[str, list[float]] = {
        "static_loss": [], "continuous_loss": [], "keyframe_loss": []
    }
    sample_mse: list[float] = []
    for step_idx, batch in enumerate(loader):
        if max_steps is not None and step_idx >= max_steps:
            break
        batch = _to_device(batch, device)
        loss, components = model.compute_loss(batch)
        losses.append(float(loss.detach().item()))
        for name in segmented_losses:
            if components[name] is not None:
                segmented_losses[name].append(components[name])
        sampled = _sample_normalized_actions(model, batch)
        sample_mse.append(float(torch.mean((sampled - batch["action"]) ** 2).item()))
    if not losses:
        return {"loss": 0.0, "static_loss": None, "continuous_loss": None, "keyframe_loss": None,
                "sample_action_mse": 0.0}
    return {
        "loss": float(np.mean(losses)),
        **{name: float(np.mean(values)) if values else None for name, values in segmented_losses.items()},
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
    model_cfg = cfg["model"]
    started_at = time.time()

    seed = int(training_cfg.get("seed", 42))
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
    wandb_logger, wandb_info = _build_wandb_logger(cfg, output_dir=output_dir, run_name=run_name)

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

    model = _build_model(model_cfg=model_cfg, data_cfg=data_cfg, train_dataset=train_dataset).to(device)
    stats = train_dataset.export_stats()
    model.set_action_stats(action_mean=stats["action_mean"].to(device), action_std=stats["action_std"].to(device))

    base_lr = float(training_cfg.get("lr", 1.0e-4))
    backbone_lr_multiplier = float(training_cfg.get("backbone_lr_multiplier", 1.0))
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if "obs_composer.encoders.0.backbone" in name:
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)
    parameter_groups = [
        {"params": head_params, "lr": base_lr},
        {"params": backbone_params, "lr": base_lr * backbone_lr_multiplier},
    ]
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
    best_model_state = None
    best_metrics: dict[str, Any] | None = None
    last_metrics: dict[str, Any] | None = None
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
        "dataset_manifest_sha256": split_manifest.get("dataset_manifest_sha256"),
        "split_manifest_sha256": split_manifest.get("split_manifest_sha256"),
    }

    def checkpoint_payload(model_state: dict[str, Any], epoch: int) -> dict[str, Any]:
        return {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
            "global_step": global_step,
            "epoch": epoch,
            "best_val": best_val,
            "best_metrics": best_metrics,
            "last_metrics": last_metrics,
            "best_model_state_dict": best_model_state,
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
                "obs_horizon": train_dataset.history_steps,
                "action_dim": train_dataset.action_dim,
                "action_horizon": train_dataset.action_horizon,
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

    resume_from = training_cfg.get("resume_from")
    if resume_from:
        resume_path = Path(str(resume_from)).expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        checkpoint_provenance = checkpoint.get("data_provenance", {})
        for key in ("stats_digest", "dataset_manifest_sha256", "split_manifest_sha256"):
            if checkpoint_provenance.get(key) != data_provenance.get(key):
                raise ValueError(f"resume checkpoint {key} does not match the current dataset")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", float("inf")))
        best_metrics = copy.deepcopy(checkpoint.get("best_metrics"))
        last_metrics = copy.deepcopy(checkpoint.get("last_metrics"))
        best_model_state = copy.deepcopy(checkpoint.get("best_model_state_dict"))
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

    for epoch in range(start_epoch, int(training_cfg.get("num_epochs", 50))):
        model.train()
        epoch_losses: list[float] = []
        epoch_flow: list[float] = []
        epoch_t: list[float] = []
        epoch_segments: dict[str, list[float]] = {
            "static_loss": [], "continuous_loss": [], "keyframe_loss": []
        }

        for batch_idx, batch in enumerate(train_loader):
            if max_train_steps is not None and batch_idx >= int(max_train_steps):
                break
            batch = _to_device(batch, device)
            loss, components = model.compute_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training_cfg.get("grad_clip") is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["grad_clip"]))
            optimizer.step()
            scheduler.step()

            epoch_losses.append(float(loss.detach().item()))
            epoch_flow.append(float(components["flow_loss"]))
            epoch_t.append(float(components["t_mean"]))
            for name in epoch_segments:
                if components[name] is not None:
                    epoch_segments[name].append(float(components[name]))
            global_step += 1

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            max_steps=int(max_val_steps) if max_val_steps is not None else None,
        )
        train_sample_loader = DataLoader(train_dataset, batch_size=min(16, int(training_cfg.get("batch_size", 64))), shuffle=False)
        train_sample_batch = next(iter(train_sample_loader))
        train_sample_batch = _to_device(train_sample_batch, device)
        train_sample = _sample_normalized_actions(model, train_sample_batch)
        train_sample_mse = float(torch.mean((train_sample - train_sample_batch["action"]) ** 2).item())

        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "train_flow_loss": float(np.mean(epoch_flow)) if epoch_flow else 0.0,
            "train_t_mean": float(np.mean(epoch_t)) if epoch_t else 0.0,
            **{
                f"train_{name}": float(np.mean(values)) if values else None
                for name, values in epoch_segments.items()
            },
            "train_sample_action_mse": train_sample_mse,
            "val_loss": float(val_metrics["loss"]),
            "val_static_loss": val_metrics["static_loss"],
            "val_continuous_loss": val_metrics["continuous_loss"],
            "val_keyframe_loss": val_metrics["keyframe_loss"],
            "val_sample_action_mse": float(val_metrics["sample_action_mse"]),
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

        improved = metrics["val_loss"] < best_val
        if improved:
            best_val = metrics["val_loss"]
            best_metrics = copy.deepcopy(metrics)
            best_model_state = copy.deepcopy(model.state_dict())

        latest_payload = checkpoint_payload(model.state_dict(), epoch)
        _atomic_torch_save(latest_payload, output_dir / "latest.ckpt")

        if improved:
            checkpoint_path = output_dir / "best.ckpt"
            _atomic_torch_save(checkpoint_payload(best_model_state, epoch), checkpoint_path)
            export_cfg = cfg.get("deployment", {}).get("eval_bundle", {})
            if bool(export_cfg.get("enabled", False)):
                bundle_dir = output_dir / "eval_bundles" / f"eval_bundle_{run_name}_step{global_step}"
                camera_resolutions = {
                    str(name): tuple(int(value) for value in resolution)
                    for name, resolution in export_cfg.get("camera_resolutions", {}).items()
                }
                exported = export_eval_bundle(
                    checkpoint_path,
                    bundle_dir,
                    execute_horizon=int(export_cfg.get("execute_horizon", train_dataset.action_horizon)),
                    camera_resolutions=camera_resolutions,
                    joint_order=list(export_cfg.get("joint_order", DEFAULT_JOINT_ORDER)),
                    data_version=str(export_cfg.get("data_version", "unknown")),
                    archive=bool(export_cfg.get("archive", True)),
                )
                print(f"[eval_bundle] {exported}")

        epochs_completed_this_run += 1
        if max_epochs_this_run is not None and epochs_completed_this_run >= int(max_epochs_this_run):
            print(f"RUN_EPOCH_LIMIT_REACHED epochs_completed={epochs_completed_this_run}")
            break

    if best_model_state is None:
        raise RuntimeError("Training finished without producing a checkpoint")
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
        "encoder_type": str(model_cfg.get("encoder_type", "cnn")),
        "timm_model_name": str(model_cfg.get("timm_model_name", "vit_small_r26_s32_224")),
        "timm_pretrained": bool(model_cfg.get("timm_pretrained", True)),
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
        "rerun": rerun_info,
        "wandb": wandb_info,
        "best": best_metrics,
        "final": final_metrics,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _append_jsonl(experiment_index_path, summary)
    wandb_logger.update_summary(summary)
    wandb_logger.save_text("summary.json", summary_path.read_text(encoding="utf-8"))
    wandb_logger.save_text("config_resolved.yaml", (output_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    wandb_logger.finish()


if __name__ == "__main__":
    main()
