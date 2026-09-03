from __future__ import annotations

import pytest
import torch

from flow_matching_test.policies.factory import build_policy, resolve_policy_type
from flow_matching_test.policies.joint_wam import JointLatentWamPolicy


def make_policy(*, action_weight: float = 1.0, video_weight: float = 0.25):
    return JointLatentWamPolicy(
        image_keys=("rgb_head", "rgb_right_hand"),
        encoder_type="cnn",
        use_proprio=True,
        history_steps=1,
        action_dim=13,
        action_horizon=4,
        d_model=16,
        n_head=4,
        n_layer=1,
        dropout=0.0,
        time_eps=1.0e-3,
        num_inference_steps=2,
        video_latent_channels=2,
        video_condition_latent_steps=3,
        video_future_latent_steps=2,
        video_latent_spatial_size=4,
        video_patch_size=2,
        action_loss_weight=action_weight,
        video_loss_weight=video_weight,
    )


def make_batch(batch_size: int = 2):
    return {
        "obs": {
            "rgb_head": torch.randn(batch_size, 1, 3, 32, 32),
            "rgb_right_hand": torch.randn(batch_size, 1, 3, 32, 32),
            "proprio": torch.randn(batch_size, 13),
            "video_condition_latent": torch.randn(batch_size, 2, 3, 4, 4),
        },
        "action": torch.randn(batch_size, 4, 13),
        "action_mask": torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32),
        "video_future_latent": torch.randn(batch_size, 2, 2, 4, 4),
        "video_future_mask": torch.tensor([[True, True], [False, False]]),
        "segment_type": torch.zeros(batch_size, dtype=torch.long),
        "is_lift": torch.zeros(batch_size, dtype=torch.bool),
    }


def test_joint_wam_combines_two_separate_flow_losses() -> None:
    policy = make_policy(action_weight=1.0, video_weight=0.25)
    loss, metrics = policy.compute_loss(make_batch())

    expected = metrics["weighted_action_flow_loss"] + metrics["weighted_video_flow_loss"]
    assert float(loss.detach().item()) == pytest.approx(expected)
    assert metrics["action_flow_loss"] > 0
    assert metrics["video_flow_loss"] > 0
    assert metrics["video_latent_valid_fraction"] == pytest.approx(0.5)


def test_each_loss_reaches_shared_joint_transformer_and_other_modality_tokens() -> None:
    action_policy = make_policy(action_weight=1.0, video_weight=0.0)
    action_loss, _ = action_policy.compute_loss(make_batch())
    action_loss.backward()
    assert torch.count_nonzero(action_policy.blocks[0].self_attn.in_proj_weight.grad) > 0
    assert torch.count_nonzero(action_policy.video_patch_embed.weight.grad) > 0

    video_policy = make_policy(action_weight=0.0, video_weight=1.0)
    video_loss, _ = video_policy.compute_loss(make_batch())
    video_loss.backward()
    assert torch.count_nonzero(video_policy.blocks[0].self_attn.in_proj_weight.grad) > 0
    assert torch.count_nonzero(video_policy.action_proj.weight.grad) > 0


def test_joint_wam_joint_ode_sampling_returns_action_and_video() -> None:
    policy = make_policy()
    obs = make_batch(batch_size=1)["obs"]
    sampled = policy.sample_actions_seeded(obs, [42])

    assert sampled.action.shape == (1, 4, 13)
    assert sampled.action_normalized.shape == (1, 4, 13)
    assert sampled.video_latent.shape == (1, 2, 2, 4, 4)
    assert torch.isfinite(sampled.video_latent).all()


def test_factory_builds_joint_wam_without_checkpoint_initialization() -> None:
    policy_cfg = {
        "type": "joint_wam",
        "video_latent_channels": 2,
        "video_condition_latent_steps": 3,
        "video_future_latent_steps": 2,
        "video_latent_spatial_size": 4,
        "video_patch_size": 2,
    }
    policy = build_policy(
        policy_cfg=policy_cfg,
        model_cfg={
            "encoder_type": "cnn",
            "use_proprio": True,
            "d_model": 16,
            "n_head": 4,
            "n_layer": 1,
            "num_inference_steps": 2,
        },
        image_keys=("rgb_head", "rgb_right_hand"),
        action_dim=13,
        history_steps=1,
        action_horizon=4,
    )
    assert isinstance(policy, JointLatentWamPolicy)
    assert resolve_policy_type(policy_cfg) == "joint_latent_wam"
