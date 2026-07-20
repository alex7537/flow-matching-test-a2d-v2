from __future__ import annotations

import pytest
import torch

from flow_matching_test.model import RGBConditionedFlowModel
from flow_matching_test.policies.factory import build_policy, resolve_policy_type
from flow_matching_test.policies.flow_matching import FlowMatchingPolicy
from flow_matching_test.policies.diffusion import DiffusionPolicy
from flow_matching_test.policies.imle import ImlePolicy


def _policy(policy_cfg: dict[str, object] | None = None):
    return build_policy(
        policy_cfg=policy_cfg or {"type": "flow_matching"},
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


def _batch() -> dict[str, object]:
    return {
        "obs": {
            "rgb_head": torch.randn(2, 1, 3, 32, 32),
            "rgb_right_hand": torch.randn(2, 1, 3, 32, 32),
            "proprio": torch.randn(2, 13),
        },
        "action": torch.randn(2, 4, 13),
        "segment_type": torch.tensor([1, 2]),
    }


def test_policy_factory_and_aliases() -> None:
    assert resolve_policy_type(None) == "flow_matching"
    assert resolve_policy_type({"type": "cfm"}) == "flow_matching"
    assert isinstance(_policy(), FlowMatchingPolicy)
    assert resolve_policy_type({"type": "dp"}) == "diffusion"
    assert resolve_policy_type({"type": "rs_imle"}) == "imle"
    with pytest.raises(ValueError, match="Unsupported policy.type='unknown'"):
        resolve_policy_type({"type": "unknown"})


def test_policy_loss_backprop_sampling_and_parameter_groups() -> None:
    policy = _policy()
    batch = _batch()
    loss, metrics = policy.compute_loss(batch)
    loss.backward()

    assert loss.ndim == 0
    assert metrics["flow_loss"] > 0.0
    assert any(parameter.grad is not None for parameter in policy.parameters())

    groups, head, backbone = policy.optimizer_parameter_groups(
        base_lr=1.0e-4,
        backbone_lr_multiplier=0.1,
    )
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(grouped_ids) == len(set(grouped_ids)) == len(list(policy.parameters()))
    assert len(head) + len(backbone) == len(list(policy.parameters()))
    assert groups[0]["lr"] == pytest.approx(1.0e-4)
    assert groups[1]["lr"] == pytest.approx(1.0e-5)

    sampled = policy.sample_actions(batch["obs"])
    assert sampled.action_normalized.shape == (2, 4, 13)
    assert sampled.action.shape == (2, 4, 13)


def test_frozen_backbone_is_excluded_from_optimizer_groups() -> None:
    policy = _policy()
    backbone = policy.backbone_parameters()
    for parameter in backbone:
        parameter.requires_grad_(False)

    groups, head, returned_backbone = policy.optimizer_parameter_groups(
        base_lr=1.0e-4,
        backbone_lr_multiplier=0.0,
    )

    grouped_ids = {id(parameter) for group in groups for parameter in group["params"]}
    assert grouped_ids == {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    assert groups[1]["params"] == []
    assert groups[1]["lr"] == 0.0
    assert returned_backbone == backbone
    assert all(not parameter.requires_grad for parameter in returned_backbone)
    assert all(parameter.requires_grad for parameter in head)


def test_legacy_model_import_and_state_dict_remain_compatible() -> None:
    assert RGBConditionedFlowModel is FlowMatchingPolicy
    source = _policy()
    target = _policy()
    target.load_state_dict(source.state_dict(), strict=True)


@pytest.mark.parametrize(
    ("policy_cfg", "expected_type", "metric_name"),
    [
        ({"type": "imle", "n_samples_per_condition": 3, "rs_imle_epsilon": 0.0}, ImlePolicy, "imle_loss"),
        ({"type": "diffusion", "diffusion_train_steps": 10, "diffusion_inference_steps": 3}, DiffusionPolicy, "epsilon_loss"),
    ],
)
def test_new_policy_loss_backprop_and_sampling(policy_cfg, expected_type, metric_name) -> None:
    policy = _policy(policy_cfg)
    assert isinstance(policy, expected_type)
    batch = _batch()
    loss, metrics = policy.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert metric_name in metrics
    assert any(parameter.grad is not None for parameter in policy.parameters())
    sampled = policy.sample_actions(batch["obs"])
    assert sampled.action_normalized.shape == (2, 4, 13)
    assert torch.isfinite(sampled.action_normalized).all()
    if isinstance(policy, DiffusionPolicy):
        assert sampled.action_normalized.abs().max() <= 1.0


def test_imle_bidirectional_selection_ignores_arm_and_randomizes_weighted_ties() -> None:
    policy = _policy(
        {
            "type": "imle",
            "bidirection_arm_weight": 0.0,
            "bidirection_hand_weight": 1.0,
        }
    )
    assert isinstance(policy, ImlePolicy)
    previous = torch.zeros(1, 4, 13)
    previous[:, 2:, 7:] = 1.0
    candidates = torch.zeros(1, 3, 4, 13)
    candidates[:, 0, :2, :7] = 100.0
    candidates[:, 0, :2, 7:] = 1.0
    candidates[:, 1, :2, :7] = -100.0
    candidates[:, 1, :2, 7:] = 1.0
    candidates[:, 2, :2, 7:] = -1.0

    selected_arms = set()
    for seed in range(20):
        torch.manual_seed(seed)
        selected = policy._select_bidirectional_candidates(
            previous_action=previous,
            candidates=candidates,
            execute_horizon=2,
        )
        assert torch.all(selected[:, :2, 7:] == 1.0)
        selected_arms.add(float(selected[0, 0, 0]))
    assert selected_arms == {-100.0, 100.0}
