from __future__ import annotations

import torch

from flow_matching_test.policies.factory import build_policy
from flow_matching_test.train import (
    _encoder_feature_std,
    _grad_l2_norm,
    _parameter_l2_norm,
    _parameter_update_ratio,
    _set_backbone_frozen,
    _snapshot_parameters,
)


def _policy():
    return build_policy(
        policy_cfg={"type": "flow_matching"},
        model_cfg={
            "encoder_type": "cnn",
            "use_proprio": True,
            "d_model": 16,
            "n_head": 4,
            "n_layer": 1,
            "dropout": 0.0,
            "time_eps": 1.0e-3,
            "num_inference_steps": 2,
        },
        image_keys=("rgb_head", "rgb_right_hand"),
        action_dim=13,
        history_steps=1,
        action_horizon=4,
    )


def test_encoder_monitoring_metrics_are_finite_and_nonzero_after_update() -> None:
    torch.manual_seed(7)
    policy = _policy()
    obs = {
        "rgb_head": torch.randn(2, 1, 3, 32, 32),
        "rgb_right_hand": torch.randn(2, 1, 3, 32, 32),
        "proprio": torch.randn(2, 13),
    }
    batch = {
        "obs": obs,
        "action": torch.randn(2, 4, 13),
        "segment_type": torch.tensor([1, 2]),
    }
    backbone = policy.backbone_parameters()
    reference = _snapshot_parameters(backbone)
    parameter_norm = _parameter_l2_norm(backbone)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0e-3)

    loss, _ = policy.compute_loss(batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = _grad_l2_norm(backbone)
    gradient_parameter_ratio = gradient_norm / parameter_norm
    optimizer.step()

    update_ratio = _parameter_update_ratio(backbone, reference)
    feature_std = _encoder_feature_std(policy, obs)

    assert parameter_norm > 0.0
    assert gradient_norm > 0.0
    assert torch.isfinite(torch.tensor(gradient_parameter_ratio))
    assert gradient_parameter_ratio > 0.0
    assert torch.isfinite(torch.tensor(update_ratio))
    assert update_ratio > 0.0
    assert torch.isfinite(torch.tensor(feature_std))
    assert feature_std > 0.0


def test_encoder_update_ratio_is_zero_without_an_optimizer_step() -> None:
    policy = _policy()
    backbone = policy.backbone_parameters()
    reference = _snapshot_parameters(backbone)
    assert _parameter_update_ratio(backbone, reference) == 0.0


def test_set_backbone_frozen_preserves_trainable_policy_head() -> None:
    policy = _policy()
    backbone = policy.backbone_parameters()
    backbone_ids = {id(parameter) for parameter in backbone}

    count = _set_backbone_frozen(policy, frozen=True)

    assert count == sum(parameter.numel() for parameter in backbone)
    assert all(not parameter.requires_grad for parameter in backbone)
    assert all(
        parameter.requires_grad
        for parameter in policy.parameters()
        if id(parameter) not in backbone_ids
    )
