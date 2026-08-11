from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from flow_matching_test.export_bundle import DEFAULT_JOINT_ORDER, _git_sha, export_eval_bundle
from flow_matching_test.policies.factory import build_policy, resolve_policy_type
from rollout.policy_wrapper import Policy
from rollout.report import build_report
from rollout.run_rollout import _bundle_provenance, load_trials
from rollout.success_checker import ThreePhaseChecker


def test_git_sha_is_optional_when_git_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr("flow_matching_test.export_bundle.subprocess.run", missing_git)
    assert _git_sha(Path(".")) is None


def _checkpoint(
    path: Path,
    policy_cfg: dict | None = None,
    *,
    include_ema: bool = False,
) -> None:
    policy_cfg = policy_cfg or {"type": "flow_matching"}
    config = {
        "data": {
            "image_keys": ["rgb_head", "rgb_right_hand"],
            "image_size": 32,
            "history_steps": 1,
            "action_horizon": 4,
        },
        "model": {
            "encoder_type": "cnn",
            "use_proprio": True,
            "d_model": 16,
            "n_head": 4,
            "n_layer": 1,
            "dropout": 0.0,
            "time_eps": 0.001,
            "num_inference_steps": 2,
        },
        "training": {"seed": 7},
        "policy": policy_cfg,
        "deployment": {"eval_bundle": {"action_range_guard_margin_ratio": 10.0}},
    }
    model = build_policy(
        policy_cfg=policy_cfg,
        model_cfg=config["model"],
        image_keys=("rgb_head", "rgb_right_hand"),
        history_steps=1,
        action_dim=13,
        action_horizon=4,
    )
    stats = {
        "schema_version": 2,
        "action_semantics": "executed_joint_position",
        "normalization": "train_minmax",
        "range_eps": 1.0e-4,
        "train_episode_digest": "fixture-stats-digest",
        "state": {"min": [-1.0] * 13, "max": [1.0] * 13, "span": [2.0] * 13},
        "action": {"min": [-1.0] * 13, "max": [1.0] * 13, "span": [2.0] * 13},
    }
    payload = {
            "model_state_dict": model.state_dict(),
            "selection_criterion": "val_sample_action_mse",
            "policy_type": resolve_policy_type(policy_cfg),
            "policy_spec": {"type": resolve_policy_type(policy_cfg)},
            "config": config,
            "normalizer": stats,
            "stats_digest": "fixture-stats-digest",
            "split_manifest": {
                "schema_version": 1,
                "dataset_manifest_sha256": "fixture-dataset-manifest",
                "split_manifest_sha256": "fixture-split-manifest",
                "train_episodes": ["train.hdf5"],
                "val_episodes": ["val.hdf5"],
            },
            "data_provenance": {
                "stats_digest": "fixture-stats-digest",
                "segmentation_version": 1,
                "segmentation_motion_threshold": 1.0e-4,
                "segmentation_keyframe_threshold": 0.1,
            },
            "training_environment": {
                "python": __import__("sys").version.split()[0],
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "timm": __import__("timm").__version__,
            },
            "global_step": 12,
            "epoch": 2,
        }
    if include_ema:
        payload["ema_model_state_dict"] = {
            name: torch.zeros_like(value)
            for name, value in model.state_dict().items()
        }
    torch.save(payload, path)


def test_bundle_policy_round_trip(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.ckpt"
    bundle = tmp_path / "bundle"
    _checkpoint(ckpt)
    export_eval_bundle(ckpt, bundle, execute_horizon=2)
    exported_config = yaml.safe_load((bundle / "config.yaml").read_text())
    assert exported_config["policy"]["type"] == "flow_matching"
    assert exported_config["action"]["offset_steps"] == 0
    assert exported_config["joint_order"] == DEFAULT_JOINT_ORDER
    assert (bundle / "data_split.json").exists()
    manifest = yaml.safe_load((bundle / "manifest.json").read_text())
    assert manifest["policy_type"] == "flow_matching"
    assert manifest["source_checkpoint_selection"] == "val_sample_action_mse"
    assert manifest["weights_variant"] == "raw"
    assert len(manifest["source_checkpoint_sha256"]) == 64
    assert "data_split.json" in manifest["files"]
    policy = Policy(bundle, device="cpu")
    obs = {
        "images": {
            "rgb_head": np.zeros((24, 40, 3), dtype=np.uint8),
            "rgb_right_hand": np.zeros((24, 40, 3), dtype=np.uint8),
        },
        "proprio": np.zeros(13, dtype=np.float32),
    }
    first = policy.infer(obs)
    policy.reset()
    second = policy.infer(obs)
    assert first.shape == (4, 13)
    np.testing.assert_array_equal(first, second)
    policy.action_range_margin_ratio = 0.1
    try:
        policy._validate_action_range(np.full((4, 13), 2.0, dtype=np.float32))
    except ValueError as exc:
        assert "exceeds train range guard" in str(exc)
    else:
        raise AssertionError("out-of-range policy action was accepted")


def test_bundle_can_override_cfm_inference_steps(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.ckpt"
    bundle = tmp_path / "bundle"
    _checkpoint(ckpt)

    export_eval_bundle(ckpt, bundle, num_inference_steps=7)

    exported_config = yaml.safe_load((bundle / "config.yaml").read_text())
    assert exported_config["model"]["cfm"]["num_inference_steps"] == 7
    policy = Policy(bundle, device="cpu")
    assert policy.model.num_inference_steps == 7


def test_bundle_can_export_ema_weights_with_explicit_provenance(tmp_path: Path) -> None:
    ckpt = tmp_path / "best_action_mse.ckpt"
    bundle = tmp_path / "bundle"
    _checkpoint(ckpt, include_ema=True)

    export_eval_bundle(ckpt, bundle, weights_variant="ema")

    manifest = yaml.safe_load((bundle / "manifest.json").read_text())
    exported_state = torch.load(bundle / "ckpt.pt", map_location="cpu", weights_only=True)
    assert manifest["source_checkpoint_selection"] == "val_sample_action_mse"
    assert manifest["weights_variant"] == "ema"
    assert all(torch.count_nonzero(value) == 0 for value in exported_state.values())


def test_bundle_can_override_legacy_checkpoint_selection(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.ckpt"
    bundle = tmp_path / "bundle"
    _checkpoint(ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload.pop("selection_criterion")
    torch.save(payload, ckpt)

    export_eval_bundle(
        ckpt,
        bundle,
        checkpoint_selection="historical_val_loss",
    )

    manifest = yaml.safe_load((bundle / "manifest.json").read_text())
    assert manifest["source_checkpoint_selection"] == "historical_val_loss"


@pytest.mark.parametrize(
    "policy_cfg",
    [
        {"type": "imle", "n_samples_per_condition": 2, "rs_imle_epsilon": 0.0},
        {"type": "diffusion", "diffusion_train_steps": 10, "diffusion_inference_steps": 3},
    ],
)
def test_alternate_policy_bundle_round_trip(tmp_path: Path, policy_cfg: dict) -> None:
    ckpt = tmp_path / "best.ckpt"
    bundle = tmp_path / "bundle"
    _checkpoint(ckpt, policy_cfg=policy_cfg)
    export_eval_bundle(ckpt, bundle, execute_horizon=2)
    exported_config = yaml.safe_load((bundle / "config.yaml").read_text())
    if policy_cfg["type"] == "imle":
        assert exported_config["policy"]["bidirection_enabled"] is True
        assert exported_config["policy"]["bidirection_num_candidates"] == 32
        assert exported_config["policy"]["bidirection_arm_weight"] == 0.0
        assert exported_config["policy"]["bidirection_hand_weight"] == 1.0
    policy = Policy(bundle, device="cpu")
    policy.action_range_margin_ratio = 1.0e6  # random, untrained weights are intentionally unconstrained
    action = policy.infer(
        {
            "images": {
                "rgb_head": np.zeros((24, 40, 3), dtype=np.uint8),
                "rgb_right_hand": np.zeros((24, 40, 3), dtype=np.uint8),
            },
            "proprio": np.zeros(13, dtype=np.float32),
        }
    )
    assert action.shape == (4, 13)
    assert np.isfinite(action).all()
    if policy_cfg["type"] == "imle":
        second = policy.infer(
            {
                "images": {
                    "rgb_head": np.zeros((24, 40, 3), dtype=np.uint8),
                    "rgb_right_hand": np.zeros((24, 40, 3), dtype=np.uint8),
                },
                "proprio": np.zeros(13, dtype=np.float32),
            },
            execute_horizon=2,
        )
        assert second.shape == (4, 13)
        assert np.isfinite(second).all()


def test_success_checker_and_report() -> None:
    checker = ThreePhaseChecker(
        {
            "approach_distance_m": 0.1,
            "close_contact_count": 2,
            "close_hold_steps": 2,
            "lift_height_m": 0.02,
            "lift_hold_steps": 2,
        }
    )
    checker.update({"object_position": [0, 0, 0], "eef_position": [0, 0, 0.05], "contact_count": 2})
    checker.update({"object_position": [0, 0, 0], "eef_position": [0, 0, 0.05], "contact_count": 2})
    checker.update({"object_position": [0, 0, 0.03], "eef_position": [0, 0, 0.04], "contact_count": 2})
    checker.update({"object_position": [0, 0, 0.03], "eef_position": [0, 0, 0.04], "contact_count": 2})
    result = {"metric_group": "primary", **checker.summary()}
    assert checker.done()
    provenance = {
        "weights_variant": "ema",
        "checkpoint_selection": "ema_val_sample_action_mse",
        "source_checkpoint_sha256": "a" * 64,
        "bundle_checkpoint_sha256": "b" * 64,
    }
    report = build_report([result], run_metadata=provenance)
    assert report["overall"]["success_rate"] == 1.0
    assert report["provenance"] == provenance


def test_bundle_provenance_maps_manifest_fields() -> None:
    provenance = _bundle_provenance(
        {
            "weights_variant": "ema",
            "source_checkpoint_selection": "ema_val_sample_action_mse",
            "source_checkpoint_sha256": "a" * 64,
            "files": {"ckpt.pt": {"sha256": "b" * 64}},
        }
    )
    assert provenance == {
        "weights_variant": "ema",
        "checkpoint_selection": "ema_val_sample_action_mse",
        "source_checkpoint_sha256": "a" * 64,
        "bundle_checkpoint_sha256": "b" * 64,
    }


def test_grid_expansion() -> None:
    grid = yaml.safe_load(Path("rollout/eval_grid.yaml").read_text())
    trials = load_trials(grid)
    assert len(trials) == 182
    assert len({trial["id"] for trial in trials}) == 182
