from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from flow_matching_test.policies.factory import resolve_policy_type
from flow_matching_test.train import _build_dataset, _build_policy, _to_device, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit one fixed A2D batch.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    policy_cfg = cfg.get("policy", {"type": "flow_matching"})
    policy_type = resolve_policy_type(policy_cfg)
    seed = int(training_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(str(training_cfg.get("device", "cuda")))
    dataset = _build_dataset(data_cfg=data_cfg, split="train", seed=seed)
    batch_size = args.batch_size or int(training_cfg.get("batch_size", 32))
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)))
    batch = _to_device(batch, device)

    model = _build_policy(
        policy_cfg=policy_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_dataset=dataset,
    ).to(device)
    stats = dataset.export_stats()
    model.set_action_stats(
        action_mean=stats["action_mean"].to(device),
        action_std=stats["action_std"].to(device),
    )

    base_lr = float(training_cfg.get("lr", 1.0e-4))
    backbone_lr_multiplier = float(training_cfg.get("backbone_lr_multiplier", 1.0))
    parameter_groups, _, _ = model.optimizer_parameter_groups(
        base_lr=base_lr,
        backbone_lr_multiplier=backbone_lr_multiplier,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=base_lr,
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-4)),
        betas=tuple(training_cfg.get("betas", [0.9, 0.95])),
    )

    fixed_objective_seed = seed + 1

    def fixed_loss() -> torch.Tensor:
        torch.manual_seed(fixed_objective_seed)
        torch.cuda.manual_seed_all(fixed_objective_seed)
        return model.compute_loss(batch)[0]

    model.train()
    initial_loss = float(fixed_loss().detach().item())
    for step in range(args.steps):
        loss = fixed_loss()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_clip = training_cfg.get("grad_clip")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach().item())}))

    final_loss = float(fixed_loss().detach().item())
    result = {
        "status": "PASS" if final_loss < initial_loss else "FAIL",
        "policy_type": policy_type,
        "steps": args.steps,
        "batch_size": batch_size,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / initial_loss,
    }
    print("SINGLE_BATCH_OVERFIT " + json.dumps(result))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
