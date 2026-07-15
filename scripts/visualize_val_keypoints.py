from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from flow_matching_test.train import _build_dataset, _build_policy, _to_device


def _phase_start(phases: list[str], prefix: str) -> int:
    for index, phase in enumerate(phases):
        if phase.startswith(prefix):
            return index
    raise ValueError(f"episode has no phase starting with {prefix!r}")


def _rgb(frame: torch.Tensor) -> np.ndarray:
    image = frame.detach().cpu().float().numpy().transpose(1, 2, 0)
    return np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize pregrasp/grasp predictions for every val episode")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = _build_dataset(
        data_cfg=cfg["data"],
        split="val",
        seed=int(cfg["training"].get("seed", 42)),
    )
    model = _build_policy(
        policy_cfg=cfg.get("policy"),
        model_cfg=cfg["model"],
        data_cfg=cfg["data"],
        train_dataset=dataset,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.set_action_stats(
        action_mean=torch.as_tensor(payload["action_mean"], dtype=torch.float32, device=device),
        action_std=torch.as_tensor(payload["action_std"], dtype=torch.float32, device=device),
    )
    model.eval()

    sample_lookup = {sample: index for index, sample in enumerate(dataset.samples)}
    rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(len(dataset.episodes), 4, figsize=(18, 4 * len(dataset.episodes)))
    if len(dataset.episodes) == 1:
        axes = axes[None, :]

    for episode_index, episode in enumerate(dataset.episodes):
        with h5py.File(episode["path"], "r") as file:
            phases = [value.decode() if isinstance(value, bytes) else str(value) for value in file["phase"][:]]
        frames = {
            "pregrasp": _phase_start(phases, "pregrasp"),
            "grasp": _phase_start(phases, "grasp-servo"),
        }
        predictions: dict[str, dict[str, object]] = {}
        for stage_index, (stage, frame) in enumerate(frames.items()):
            sample_index = sample_lookup[(episode_index, frame)]
            sample = dataset[sample_index]
            batch = _to_device(
                {
                    "obs": {key: value.unsqueeze(0) for key, value in sample["obs"].items()},
                    "action": sample["action"].unsqueeze(0),
                },
                device,
            )
            torch.manual_seed(int(cfg["training"].get("seed", 42)) + episode_index * 2 + stage_index)
            prediction = model.sample_actions(batch["obs"])
            gt = model.denormalize_action(batch["action"])[0].cpu().numpy()
            pred = prediction.action[0].cpu().numpy()
            predictions[stage] = {
                "frame": frame,
                "gt": gt,
                "pred": pred,
                "obs": batch["obs"],
                "chunk_mse_rad2": float(np.mean((pred - gt) ** 2)),
                "arm_chunk_mse_rad2": float(np.mean((pred[:, :7] - gt[:, :7]) ** 2)),
                "hand_chunk_mse_rad2": float(np.mean((pred[:, 7:] - gt[:, 7:]) ** 2)),
            }

        pregrasp = predictions["pregrasp"]
        grasp = predictions["grasp"]
        axes[episode_index, 0].imshow(_rgb(pregrasp["obs"]["rgb_head"][0, -1]))
        axes[episode_index, 0].set_title(f"{episode['file_name']}\npregrasp frame {pregrasp['frame']}")
        axes[episode_index, 2].imshow(_rgb(grasp["obs"]["rgb_right_hand"][0, -1]))
        axes[episode_index, 2].set_title(f"grasp frame {grasp['frame']} | right hand")
        for column in (0, 2):
            axes[episode_index, column].axis("off")

        arm_x = np.arange(7)
        axes[episode_index, 1].plot(arm_x, pregrasp["gt"][-1, :7], "o-k", label="GT")
        axes[episode_index, 1].plot(arm_x, pregrasp["pred"][-1, :7], "o-", color="tab:orange", label="Pred")
        axes[episode_index, 1].set_title(f"pregrasp arm endpoint | MSE={pregrasp['arm_chunk_mse_rad2']:.4f}")
        axes[episode_index, 1].set_xlabel("arm joint")
        axes[episode_index, 1].legend()

        hand_x = np.arange(6)
        axes[episode_index, 3].plot(hand_x, grasp["gt"][-1, 7:], "o-k", label="GT")
        axes[episode_index, 3].plot(hand_x, grasp["pred"][-1, 7:], "o-", color="tab:orange", label="Pred")
        axes[episode_index, 3].set_title(f"grasp hand endpoint | MSE={grasp['hand_chunk_mse_rad2']:.4f}")
        axes[episode_index, 3].set_xlabel("hand channel")
        axes[episode_index, 3].legend()

        rows.append(
            {
                "episode": episode["file_name"],
                "pregrasp": {key: value for key, value in pregrasp.items() if key not in {"gt", "pred", "obs"}},
                "grasp": {key: value for key, value in grasp.items() if key not in {"gt", "pred", "obs"}},
            }
        )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out_dir / "val_keypoints.png", dpi=150)
    plt.close(figure)
    report = {
        "checkpoint": str(ckpt_path),
        "split": "val",
        "episodes": rows,
        "mean_pregrasp_arm_mse_rad2": float(np.mean([row["pregrasp"]["arm_chunk_mse_rad2"] for row in rows])),
        "mean_grasp_hand_mse_rad2": float(np.mean([row["grasp"]["hand_chunk_mse_rad2"] for row in rows])),
    }
    (out_dir / "val_keypoints.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
