from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flow_matching_test.train import _build_dataset, _build_model, _to_device
from flow_matching_test.rerun_logger import RerunTrainVisualizer


def _resolve_device(raw: str | None) -> torch.device:
    if raw is not None:
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint eval samples to Rerun RRD")
    parser.add_argument("--ckpt", required=True, help="Path to best.ckpt")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split to visualize")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of dataset windows to export")
    parser.add_argument("--device", default=None, help="Torch device, default cuda if available else cpu")
    parser.add_argument("--spawn", action="store_true", help="Spawn a Rerun viewer while exporting")
    parser.add_argument("--output", default=None, help="Output .rrd path")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    device = _resolve_device(args.device)
    seed = int(training_cfg.get("seed", 42))
    dataset = _build_dataset(data_cfg=data_cfg, split=str(args.split), seed=seed)
    model = _build_model(model_cfg=model_cfg, data_cfg=data_cfg, train_dataset=dataset).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.set_action_stats(
        action_mean=torch.from_numpy(np.asarray(payload["action_mean"], dtype=np.float32)).to(device),
        action_std=torch.from_numpy(np.asarray(payload["action_std"], dtype=np.float32)).to(device),
    )
    model.eval()

    if args.output is None:
        output_path = ckpt_path.parent / f"checkpoint_eval_{args.split}.rrd"
    else:
        output_path = Path(args.output).expanduser().resolve()

    visualizer = RerunTrainVisualizer(
        enabled=True,
        app_name=f"flow-matching-eval-{ckpt_path.parent.name}",
        spawn=bool(args.spawn),
        save_path=str(output_path),
    )

    metrics: list[dict[str, float]] = []
    total = min(int(args.num_samples), len(dataset))
    for sample_idx in range(total):
        sample = dataset[sample_idx]
        batch = {
            "obs": {key: value.unsqueeze(0) for key, value in sample["obs"].items()},
            "action": sample["action"].unsqueeze(0),
        }
        batch = _to_device(batch, device)
        loss, _ = model.compute_loss(batch)
        prediction = model.sample_actions(batch["obs"])
        mse = float(torch.mean((prediction.action_normalized - batch["action"]) ** 2).item())
        metrics.append({"sample_index": sample_idx, "loss": float(loss.item()), "sample_action_mse": mse})
        visualizer.log_epoch_metrics(
            epoch=sample_idx,
            global_step=sample_idx,
            metrics={"loss": float(loss.item()), "sample_action_mse": mse},
        )
        visualizer.log_sample(
            sample_index=sample_idx,
            split=str(args.split),
            obs=batch["obs"],
            gt_action_normalized=batch["action"],
            pred_action_normalized=prediction.action_normalized,
            gt_action=model.denormalize_action(batch["action"]),
            pred_action=prediction.action,
        )

    summary = {
        "ckpt": str(ckpt_path),
        "split": str(args.split),
        "num_samples": total,
        "device": str(device),
        "output_rrd": str(output_path),
        "mean_loss": float(np.mean([item["loss"] for item in metrics])) if metrics else None,
        "mean_sample_action_mse": float(np.mean([item["sample_action_mse"] for item in metrics])) if metrics else None,
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
