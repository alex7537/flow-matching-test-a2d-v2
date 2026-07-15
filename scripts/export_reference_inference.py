from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import timm
import torch

from flow_matching_test.train import _build_dataset, _build_policy, _to_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export a deterministic cross-machine inference reference")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = _build_dataset(
        data_cfg=cfg["data"],
        split=args.split,
        seed=int(cfg["training"].get("seed", 42)),
    )
    if not 0 <= args.episode_index < len(dataset.episodes):
        raise IndexError(f"episode-index {args.episode_index} outside [0, {len(dataset.episodes)})")
    sample_lookup = {sample: index for index, sample in enumerate(dataset.samples)}
    sample_key = (args.episode_index, args.frame)
    if sample_key not in sample_lookup:
        raise ValueError(f"frame {args.frame} is not a valid window start for episode {args.episode_index}")

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

    sample = dataset[sample_lookup[sample_key]]
    obs = _to_device({key: value.unsqueeze(0) for key, value in sample["obs"].items()}, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prediction = model.sample_actions(obs)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "reference_input.npz"
    np.savez_compressed(input_path, **{key: value.detach().cpu().numpy() for key, value in obs.items()})
    episode = dataset.episodes[args.episode_index]
    report = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": _sha256(ckpt_path),
        "reference_input": input_path.name,
        "reference_input_sha256": _sha256(input_path),
        "split": args.split,
        "episode_index": args.episode_index,
        "episode": episode["file_name"],
        "frame": args.frame,
        "seed": args.seed,
        "tf32": False,
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "device": str(device),
        "action_shape": list(prediction.action.shape),
        "action": prediction.action[0].detach().cpu().tolist(),
        "action_normalized": prediction.action_normalized[0].detach().cpu().tolist(),
    }
    output_path = output_dir / "reference_output.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"action", "action_normalized"}}))


if __name__ == "__main__":
    main()
