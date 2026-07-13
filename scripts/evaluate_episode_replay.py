from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from flow_matching_test.train import _build_dataset, _build_model, _to_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one-episode action replay")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 16])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    dataset = _build_dataset(data_cfg=cfg["data"], split="val", seed=int(cfg["training"]["seed"]))
    device = torch.device(args.device)
    model = _build_model(model_cfg=cfg["model"], data_cfg=cfg["data"], train_dataset=dataset).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.set_action_stats(
        action_mean=torch.as_tensor(payload["action_mean"], dtype=torch.float32, device=device),
        action_std=torch.as_tensor(payload["action_std"], dtype=torch.float32, device=device),
    )
    model.eval()

    episode = dataset.episodes[0]
    with h5py.File(episode["path"], "r") as file:
        gt = np.asarray(file[dataset.cfg.action_key], dtype=np.float32)
    length = len(gt)
    sample_lookup = {start: index for index, (_, start) in enumerate(dataset.samples)}
    last_start = max(sample_lookup)
    starts = np.asarray([min(frame, last_start) for frame in range(length)])
    offsets = np.arange(length) - starts
    unique_starts = sorted(set(starts.tolist()))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {"checkpoint": str(Path(args.ckpt).resolve()), "episode": episode["path"]}
    for inference_steps in args.steps:
        predictions = np.zeros((args.samples, length, gt.shape[1]), dtype=np.float32)
        model.num_inference_steps = int(inference_steps)
        for begin in range(0, len(unique_starts), 16):
            batch_starts = unique_starts[begin:begin + 16]
            items = [dataset[sample_lookup[start]] for start in batch_starts]
            obs = {
                key: torch.stack([item["obs"][key] for item in items]).to(device)
                for key in items[0]["obs"]
            }
            for draw in range(args.samples):
                chunks = model.sample_actions(obs).action.cpu().numpy()
                for local, start in enumerate(batch_starts):
                    frames = np.flatnonzero(starts == start)
                    predictions[draw, frames] = chunks[local, offsets[frames]]

        mean = predictions.mean(axis=0)
        error = mean - gt
        arm_dynamic = np.r_[False, np.linalg.norm(np.diff(gt[:, :7], axis=0), axis=1) > 1e-4]
        hand_dynamic = np.r_[False, np.linalg.norm(np.diff(gt[:, 7:], axis=0), axis=1) > 1e-4]

        def segment_metrics(mask: np.ndarray, dims: slice) -> dict[str, float | int]:
            selected_error = error[mask, dims]
            selected_predictions = predictions[:, mask, dims]
            return {
                "frames": int(mask.sum()),
                "mse_rad2": float(np.mean(selected_error ** 2)),
                "max_abs_rad": float(np.max(np.abs(selected_error))),
                "sample_std_rad": float(np.mean(selected_predictions.std(axis=0))),
            }

        metrics = {
            "arm_dynamic": segment_metrics(arm_dynamic, slice(0, 7)),
            "arm_static": segment_metrics(~arm_dynamic, slice(0, 7)),
            "hand_dynamic": segment_metrics(hand_dynamic, slice(7, 13)),
            "hand_static": segment_metrics(~hand_dynamic, slice(7, 13)),
            "arm_key_frames": {
                str(frame): {
                    "mse_rad2": float(np.mean(error[frame, :7] ** 2)),
                    "max_abs_rad": float(np.max(np.abs(error[frame, :7]))),
                }
                for frame in (80, 81, 82) if frame < length
            },
        }
        metrics["all"] = {
            "mse_rad2": float(np.mean(error ** 2)),
            "max_abs_rad": float(np.max(np.abs(error))),
            "sample_std_rad": float(np.mean(predictions.std(axis=0))),
        }
        report[f"euler_{inference_steps}"] = metrics

        for name, lo, hi in (("arm", 0, 7), ("hand", 7, 13)):
            fig, axes = plt.subplots(hi - lo, 1, figsize=(13, 2.1 * (hi - lo)), sharex=True)
            frames = np.arange(length)
            for dim, axis in zip(range(lo, hi), axes):
                axis.plot(frames, predictions[:, :, dim].T, color="tab:blue", alpha=0.12, linewidth=0.7)
                axis.plot(frames, mean[:, dim], color="tab:blue", linewidth=1.5, label="prediction mean")
                axis.plot(frames, gt[:, dim], color="black", linewidth=1.2, label="GT")
                axis.axvspan(0, 2, color="orange", alpha=0.15)
                axis.axvspan(128, length - 1, color="green", alpha=0.12)
                axis.set_ylabel(f"q{dim - lo}")
            axes[0].legend(loc="upper right")
            axes[-1].set_xlabel("frame")
            fig.suptitle(f"{name}: GT vs {args.samples} samples, Euler steps={inference_steps}")
            fig.tight_layout()
            fig.savefig(out_dir / f"replay_{name}_steps{inference_steps}.png", dpi=150)
            plt.close(fig)

    (out_dir / "replay_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
