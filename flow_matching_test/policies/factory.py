from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flow_matching_test.policies.base import ActionPolicy
from flow_matching_test.policies.diffusion import DiffusionPolicy
from flow_matching_test.policies.flow_matching import FlowMatchingPolicy
from flow_matching_test.policies.imle import ImlePolicy
from flow_matching_test.policies.joint_wam import JointLatentWamPolicy
from flow_matching_test.policies.video_aux import VideoAuxFlowMatchingPolicy


_ALIASES = {
    "cfm": "flow_matching",
    "flow": "flow_matching",
    "flow_matching": "flow_matching",
    "cfm_video_aux": "flow_matching_video_aux",
    "flow_matching_video_aux": "flow_matching_video_aux",
    "joint_wam": "joint_latent_wam",
    "joint_latent_wam": "joint_latent_wam",
    "diffusion": "diffusion",
    "diffusion_policy": "diffusion",
    "dp": "diffusion",
    "imle": "imle",
    "rs_imle": "imle",
}


def materialize_policy_config(policy_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a provenance-ready policy config with inference defaults made explicit."""
    config = dict(policy_cfg or {})
    policy_type = resolve_policy_type(config)
    config["type"] = policy_type
    if policy_type == "flow_matching_video_aux":
        config.setdefault("video_loss_weight", 0.01)
        config.setdefault("video_aux_hidden_dim", 128)
        config.setdefault("video_condition_steps", 9)
        config.setdefault("video_future_steps", 16)
        config.setdefault("wan_vae_dtype", "bfloat16")
        config.setdefault("video_codec_batch_size", 1)
    elif policy_type == "joint_latent_wam":
        config.setdefault("action_loss_weight", 1.0)
        config.setdefault("video_loss_weight", 0.1)
        config.setdefault("video_latent_channels", 48)
        config.setdefault("video_condition_latent_steps", 3)
        config.setdefault("video_future_latent_steps", 4)
        config.setdefault("video_latent_spatial_size", 14)
        config.setdefault("video_patch_size", 2)
    elif policy_type == "imle":
        config.setdefault("n_samples_per_condition", 20)
        config.setdefault("rs_imle_epsilon", 0.03)
        config.setdefault("bidirection_enabled", True)
        config.setdefault("bidirection_num_candidates", 32)
        config.setdefault("bidirection_arm_weight", 0.0)
        config.setdefault("bidirection_hand_weight", 1.0)
    elif policy_type == "diffusion":
        config.setdefault("diffusion_train_steps", 100)
        config.setdefault("diffusion_inference_steps", 15)
        config.setdefault("diffusion_beta_schedule", "cosine")
        config.setdefault("diffusion_beta_start", 1.0e-4)
        config.setdefault("diffusion_beta_end", 2.0e-2)
    return config


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
    policy_cfg = materialize_policy_config(policy_cfg)
    policy_type = str(policy_cfg["type"])
    enhanced_proprio = bool(model_cfg.get("enhanced_proprio", False))
    if enhanced_proprio and policy_type not in {
        "flow_matching",
        "flow_matching_video_aux",
        "joint_latent_wam",
    }:
        raise ValueError(
            "enhanced_proprio is currently supported only for flow_matching policies"
        )
    if policy_type == "flow_matching":
        return FlowMatchingPolicy(
            image_keys=image_keys,
            encoder_type=str(model_cfg.get("encoder_type", "cnn")),
            timm_model_name=str(model_cfg.get("timm_model_name", "vit_small_r26_s32_224")),
            timm_pretrained=bool(model_cfg.get("timm_pretrained", True)),
            timm_tokens_per_frame=int(model_cfg.get("timm_tokens_per_frame", 1)),
            timm_token_mode=str(model_cfg.get("timm_token_mode", "spatial")),
            use_proprio=bool(model_cfg.get("use_proprio", True)),
            enhanced_proprio=enhanced_proprio,
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
    if policy_type == "flow_matching_video_aux":
        return VideoAuxFlowMatchingPolicy(
            image_keys=image_keys,
            encoder_type=str(model_cfg.get("encoder_type", "cnn")),
            timm_model_name=str(model_cfg.get("timm_model_name", "vit_small_r26_s32_224")),
            timm_pretrained=bool(model_cfg.get("timm_pretrained", True)),
            timm_tokens_per_frame=int(model_cfg.get("timm_tokens_per_frame", 1)),
            timm_token_mode=str(model_cfg.get("timm_token_mode", "spatial")),
            use_proprio=bool(model_cfg.get("use_proprio", True)),
            enhanced_proprio=enhanced_proprio,
            action_dim=int(action_dim),
            history_steps=int(history_steps),
            action_horizon=int(action_horizon),
            d_model=int(model_cfg.get("d_model", 128)),
            n_head=int(model_cfg.get("n_head", 4)),
            n_layer=int(model_cfg.get("n_layer", 4)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            time_eps=float(model_cfg.get("time_eps", 1.0e-3)),
            num_inference_steps=int(model_cfg.get("num_inference_steps", 40)),
            video_loss_weight=float(policy_cfg.get("video_loss_weight", 0.01)),
            video_aux_hidden_dim=int(policy_cfg.get("video_aux_hidden_dim", 128)),
            video_condition_steps=int(policy_cfg.get("video_condition_steps", 9)),
            video_future_steps=int(policy_cfg.get("video_future_steps", 16)),
            wan_vae_checkpoint_path=str(policy_cfg.get("wan_vae_checkpoint_path", "")),
            wan_runtime_repo=str(policy_cfg.get("wan_runtime_repo", "")),
            wan_runtime_site_packages=str(
                policy_cfg.get("wan_runtime_site_packages", "")
            ),
            wan_vae_dtype=str(policy_cfg.get("wan_vae_dtype", "bfloat16")),
            video_codec_batch_size=int(policy_cfg.get("video_codec_batch_size", 1)),
        )
    if policy_type == "joint_latent_wam":
        return JointLatentWamPolicy(
            image_keys=image_keys,
            encoder_type=str(model_cfg.get("encoder_type", "cnn")),
            timm_model_name=str(model_cfg.get("timm_model_name", "vit_small_r26_s32_224")),
            timm_pretrained=bool(model_cfg.get("timm_pretrained", True)),
            timm_tokens_per_frame=int(model_cfg.get("timm_tokens_per_frame", 1)),
            timm_token_mode=str(model_cfg.get("timm_token_mode", "spatial")),
            use_proprio=bool(model_cfg.get("use_proprio", True)),
            enhanced_proprio=enhanced_proprio,
            action_dim=int(action_dim),
            history_steps=int(history_steps),
            action_horizon=int(action_horizon),
            d_model=int(model_cfg.get("d_model", 128)),
            n_head=int(model_cfg.get("n_head", 4)),
            n_layer=int(model_cfg.get("n_layer", 4)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            time_eps=float(model_cfg.get("time_eps", 1.0e-3)),
            num_inference_steps=int(model_cfg.get("num_inference_steps", 40)),
            action_loss_weight=float(policy_cfg.get("action_loss_weight", 1.0)),
            video_loss_weight=float(policy_cfg.get("video_loss_weight", 0.1)),
            video_latent_channels=int(policy_cfg.get("video_latent_channels", 48)),
            video_condition_latent_steps=int(
                policy_cfg.get("video_condition_latent_steps", 3)
            ),
            video_future_latent_steps=int(policy_cfg.get("video_future_latent_steps", 4)),
            video_latent_spatial_size=int(policy_cfg.get("video_latent_spatial_size", 14)),
            video_patch_size=int(policy_cfg.get("video_patch_size", 2)),
        )
    if policy_type == "imle":
        return ImlePolicy(
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
            n_samples_per_condition=int(policy_cfg.get("n_samples_per_condition", 20)),
            rs_imle_epsilon=float(policy_cfg.get("rs_imle_epsilon", 0.03)),
            bidirection_enabled=bool(policy_cfg.get("bidirection_enabled", True)),
            bidirection_num_candidates=int(policy_cfg.get("bidirection_num_candidates", 32)),
            bidirection_arm_weight=float(policy_cfg.get("bidirection_arm_weight", 0.0)),
            bidirection_hand_weight=float(policy_cfg.get("bidirection_hand_weight", 1.0)),
        )
    if policy_type == "diffusion":
        return DiffusionPolicy(
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
            diffusion_train_steps=int(policy_cfg.get("diffusion_train_steps", 100)),
            diffusion_inference_steps=int(policy_cfg.get("diffusion_inference_steps", 15)),
            diffusion_beta_schedule=str(
                policy_cfg.get("diffusion_beta_schedule", "cosine")
            ),
            diffusion_beta_start=float(
                policy_cfg.get("diffusion_beta_start", 1.0e-4)
            ),
            diffusion_beta_end=float(
                policy_cfg.get("diffusion_beta_end", 2.0e-2)
            ),
        )
    raise AssertionError(f"Unhandled policy type: {policy_type}")
