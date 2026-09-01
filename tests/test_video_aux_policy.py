from __future__ import annotations

import copy

import pytest
import torch

from flow_matching_test.policies.factory import resolve_policy_type
from flow_matching_test.policies.video_aux import VideoAuxFlowMatchingPolicy


class FakeVideoCodec:
    latent_channels = 2
    condition_latent_steps = 1
    future_latent_steps = 2

    def __init__(self) -> None:
        self.calls = 0
        self.last_batch = 0

    def __deepcopy__(self, memo):
        del memo
        return self

    def encode_condition_future(self, condition, future):
        self.calls += 1
        self.last_batch = int(condition.shape[0])
        batch = condition.shape[0]
        condition_last = torch.zeros(batch, 2, 2, 2, device=condition.device)
        future_target = torch.ones(batch, 2, 2, 2, 2, device=condition.device)
        return condition_last, future_target


def make_policy(codec: FakeVideoCodec) -> VideoAuxFlowMatchingPolicy:
    return VideoAuxFlowMatchingPolicy(
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
        video_loss_weight=0.1,
        video_aux_hidden_dim=8,
        video_codec=codec,
    )


def make_batch(video_valid_mask: torch.Tensor) -> dict[str, object]:
    batch = int(video_valid_mask.shape[0])
    return {
        "obs": {
            "rgb_head": torch.randn(batch, 1, 3, 32, 32),
            "rgb_right_hand": torch.randn(batch, 1, 3, 32, 32),
            "proprio": torch.randn(batch, 13),
        },
        "action": torch.randn(batch, 4, 13),
        "action_mask": torch.ones(batch, 4),
        "segment_type": torch.ones(batch, dtype=torch.long),
        "is_lift": torch.zeros(batch, dtype=torch.bool),
        "video_condition": torch.randn(batch, 9, 3, 8, 8),
        "video_future": torch.randn(batch, 16, 3, 8, 8),
        "video_valid_mask": video_valid_mask,
    }


def test_video_aux_policy_adds_weighted_loss_and_updates_shared_parameters() -> None:
    codec = FakeVideoCodec()
    policy = make_policy(codec)
    batch = make_batch(torch.tensor([True, False]))

    loss, metrics = policy.compute_loss(batch)
    expected = metrics["flow_loss"] + 0.1 * metrics["video_aux_loss"]
    assert float(loss.detach().item()) == pytest.approx(expected)
    assert metrics["video_aux_valid_fraction"] == pytest.approx(0.5)
    assert metrics["video_aux_loss"] > 0.0
    assert codec.last_batch == 1

    loss.backward()
    assert policy.video_aux_head.context_proj.weight.grad is not None
    assert policy.blocks[0].cross_attn.in_proj_weight.grad is not None
    assert torch.count_nonzero(policy.blocks[0].cross_attn.in_proj_weight.grad) > 0


def test_video_aux_policy_skips_codec_when_no_future_clip_is_complete() -> None:
    codec = FakeVideoCodec()
    policy = make_policy(codec)
    loss, metrics = policy.compute_loss(make_batch(torch.tensor([False, False])))

    assert torch.isfinite(loss)
    assert metrics["video_aux_loss"] == 0.0
    assert metrics["video_aux_valid_fraction"] == 0.0
    assert codec.calls == 0


def test_video_aux_inference_needs_only_action_observation() -> None:
    codec = FakeVideoCodec()
    policy = make_policy(codec)
    obs = make_batch(torch.tensor([True]))["obs"]
    sampled = policy.sample_actions(obs)

    assert sampled.action.shape == (1, 4, 13)
    assert codec.calls == 0
    assert copy.deepcopy(policy).video_codec is codec
    assert resolve_policy_type({"type": "cfm_video_aux"}) == "flow_matching_video_aux"
