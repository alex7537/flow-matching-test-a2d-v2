from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flow_matching_test.policies.base import ActionPolicy
from flow_matching_test.policies.flow_matching import FlowMatchingPolicy


_ALIASES = {
    "cfm": "flow_matching",
    "flow": "flow_matching",
    "flow_matching": "flow_matching",
}


def resolve_policy_type(policy_cfg: Mapping[str, Any] | None) -> str:
    raw = "flow_matching" if policy_cfg is None else str(policy_cfg.get("type", "flow_matching"))
    policy_type = _ALIASES.get(raw.strip().lower())
    if policy_type is None:
        available = ", ".join(sorted(set(_ALIASES.values())))
        raise ValueError(f"Unsupported policy.type={raw!r}; available: {available}")
    return policy_type


def build_policy(
    *,
    policy_cfg: Mapping[str, Any] | None,
    model_cfg: Mapping[str, Any],
    image_keys: tuple[str, ...],
    action_dim: int,
    history_steps: int,
    action_horizon: int,
) -> ActionPolicy:
    policy_type = resolve_policy_type(policy_cfg)
    if policy_type == "flow_matching":
        return FlowMatchingPolicy(
            image_keys=image_keys,
            encoder_type=str(model_cfg.get("encoder_type", "cnn")),
            timm_model_name=str(model_cfg.get("timm_model_name", "vit_small_r26_s32_224")),
            timm_pretrained=bool(model_cfg.get("timm_pretrained", True)),
            timm_tokens_per_frame=int(model_cfg.get("timm_tokens_per_frame", 1)),
            timm_token_mode=str(model_cfg.get("timm_token_mode", "spatial")),
            use_proprio=bool(model_cfg.get("use_proprio", True)),
            action_dim=int(action_dim),
            history_steps=int(history_steps),
            action_horizon=int(action_horizon),
            d_model=int(model_cfg.get("d_model", 128)),
            n_head=int(model_cfg.get("n_head", 4)),
            n_layer=int(model_cfg.get("n_layer", 4)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            time_eps=float(model_cfg.get("time_eps", 1.0e-3)),
            num_inference_steps=int(model_cfg.get("num_inference_steps", 40)),
        )
    raise AssertionError(f"Unhandled policy type: {policy_type}")
