from __future__ import annotations

import pytest
import torch

from flow_matching_test.model import RGBConditionedFlowModel
from flow_matching_test.policies.base import masked_action_mse_per_sample
from flow_matching_test.policies.factory import build_policy, resolve_policy_type
from flow_matching_test.policies.flow_matching import FlowMatchingPolicy
from flow_matching_test.policies.diffusion import DiffusionPolicy
from flow_matching_test.policies.imle import ImlePolicy
from flow_matching_test.train import evaluate


def _policy(
    policy_cfg: dict[str, object] | None = None,
    *,
    use_proprio: bool = True,
    enhanced_proprio: bool = False,
):
    return build_policy(
        policy_cfg=policy_cfg or {"type": "flow_matching"},
        model_cfg={
            "encoder_type": "cnn",
            "use_proprio": use_proprio,
            "enhanced_proprio": enhanced_proprio,
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
        "action_mask": torch.ones(2, 4),
        "segment_type": torch.tensor([1, 2]),
        "is_lift": torch.tensor([False, True]),
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
    assert metrics["lift_loss"] is not None
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


def test_rgb_only_policy_does_not_require_or_use_proprio() -> None:
    policy = _policy(use_proprio=False)
    batch = _batch()
    obs_without_proprio = {
        key: value for key, value in batch["obs"].items() if key != "proprio"
    }
    batch["obs"] = obs_without_proprio

    loss, _ = policy.compute_loss(batch)
    loss.backward()
    sampled = policy.sample_actions(obs_without_proprio)

    assert policy.use_proprio is False
    assert policy.proprio_proj is None
    assert sampled.action.shape == (2, 4, 13)


def test_enhanced_proprio_warm_start_is_zero_initialized_and_trainable() -> None:
    base = _policy()
    enhanced = _policy(enhanced_proprio=True)
    load_result = enhanced.load_state_dict(base.state_dict(), strict=False)
    expected_missing = {
        f"{name}.{field}"
        for name in (
            "joint_delta_proj",
            "previous_action_proj",
            "hand_tracking_error_proj",
        )
        for field in ("weight", "bias")
    }
    assert set(load_result.missing_keys) == expected_missing
    assert load_result.unexpected_keys == []
    for projection in (
        enhanced.joint_delta_proj,
        enhanced.previous_action_proj,
        enhanced.hand_tracking_error_proj,
    ):
        assert projection is not None
        assert torch.count_nonzero(projection.weight) == 0
        assert torch.count_nonzero(projection.bias) == 0

    batch = _batch()
    batch["obs"].update(
        {
            "joint_delta": torch.randn(2, 13),
            "previous_action": torch.randn(2, 13),
            "hand_tracking_error": torch.randn(2, 6),
        }
    )
    loss, _ = enhanced.compute_loss(batch)
    loss.backward()
    assert enhanced.joint_delta_proj is not None
    assert enhanced.joint_delta_proj.weight.grad is not None
    assert torch.count_nonzero(enhanced.joint_delta_proj.weight.grad) > 0


def test_masked_action_mse_ignores_padded_timesteps() -> None:
    prediction = torch.zeros(2, 4, 3)
    target = torch.zeros_like(prediction)
    target[0, 0] = 1.0
    target[0, 1:] = 1000.0
    target[1] = 2.0
    action_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]])

    per_sample = masked_action_mse_per_sample(prediction, target, action_mask)

    torch.testing.assert_close(per_sample, torch.tensor([1.0, 4.0]))


def test_seeded_cfm_validation_is_reproducible() -> None:
    policy = _policy()
    batch = _batch()
    batch["sample_index"] = torch.tensor([3, 9])

    first = evaluate(
        model=policy,
        loader=[batch],
        device=torch.device("cpu"),
        deterministic_seed=42,
        sample_draws=3,
    )
    torch.manual_seed(999)
    second = evaluate(
        model=policy,
        loader=[batch],
        device=torch.device("cpu"),
        deterministic_seed=42,
        sample_draws=3,
    )

    assert first.keys() == second.keys()
    for key in first:
        assert first[key] == pytest.approx(second[key], rel=1.0e-5, abs=1.0e-7)


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
