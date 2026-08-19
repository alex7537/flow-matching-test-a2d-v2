from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from flow_matching_test.export_bundle import DEFAULT_JOINT_ORDER, _git_sha, export_eval_bundle
from flow_matching_test.action_contract import HYBRID_ACTION_SEMANTICS
from flow_matching_test.policies.factory import build_policy, resolve_policy_type
from rollout.policy_wrapper import Policy
from rollout.report import build_report
from rollout.retry_controller import GraspRetryConfig, GraspRetryController
from rollout.run_rollout import _bundle_provenance, _execute_trial, load_trials
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
    use_proprio: bool = True,
    action_semantics: str = "executed_joint_position",
) -> None:
    policy_cfg = policy_cfg or {"type": "flow_matching"}
    config = {
        "data": {
            "image_keys": ["rgb_head", "rgb_right_hand"],
            "image_size": 32,
            "history_steps": 1,
            "action_horizon": 4,
            "action_semantics": action_semantics,
        },
        "model": {
            "encoder_type": "cnn",
            "use_proprio": use_proprio,
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
        "action_semantics": action_semantics,
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


def test_hybrid_hand_target_bundle_preserves_action_semantics(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.ckpt"
    bundle = tmp_path / "bundle"
    _checkpoint(ckpt, action_semantics=HYBRID_ACTION_SEMANTICS)

    export_eval_bundle(ckpt, bundle, execute_horizon=2)

    exported_config = yaml.safe_load((bundle / "config.yaml").read_text())
    assert exported_config["action"]["target"] == HYBRID_ACTION_SEMANTICS
    assert exported_config["action"]["layout"][1]["name"] == "hand2_pos_target"
    policy = Policy(bundle, device="cpu")
    assert policy.stats["action_semantics"] == HYBRID_ACTION_SEMANTICS


def test_rgb_only_bundle_rollout_does_not_require_proprio(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.ckpt"
    bundle = tmp_path / "bundle"
    _checkpoint(ckpt, use_proprio=False)

    export_eval_bundle(ckpt, bundle, execute_horizon=2)

    exported_config = yaml.safe_load((bundle / "config.yaml").read_text())
    assert exported_config["model"]["use_proprio"] is False
    assert exported_config["obs"]["proprio_dim"] == 0
    policy = Policy(bundle, device="cpu")
    action = policy.infer(
        {
            "images": {
                "rgb_head": np.zeros((24, 40, 3), dtype=np.uint8),
                "rgb_right_hand": np.zeros((24, 40, 3), dtype=np.uint8),
            }
        }
    )
    assert action.shape == (4, 13)


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
    assert report["overall"]["first_attempt_success_rate"] == 1.0
    assert report["overall"]["recovered_success_rate"] == 0.0
    assert report["overall"]["retry_rate"] == 0.0
    assert report["provenance"] == provenance


class _RetryPolicy:
    def __init__(self) -> None:
        self.seed = 0
        self.reset_seeds: list[int] = []

    def reset(self) -> None:
        self.reset_seeds.append(self.seed)

    def infer(self, obs: dict, *, execute_horizon: int) -> np.ndarray:
        return np.zeros((execute_horizon, 13), dtype=np.float32)


class _RetryEnv:
    def __init__(self) -> None:
        self.attempt_index = 0
        self.step_index = 0
        self.recovery_calls: list[tuple[int, tuple[float, ...] | None]] = []
        self.current = self._state(contact_count=0, object_height=0.0)
        self.sequences = [
            [
                self._state(contact_count=0, object_height=0.0),
                self._state(contact_count=0, object_height=0.0),
            ],
            [
                self._state(contact_count=2, object_height=0.0),
                self._state(contact_count=2, object_height=0.03),
            ],
        ]

    @staticmethod
    def _state(*, contact_count: int, object_height: float) -> dict:
        return {
            "object_position": [0.0, 0.0, object_height],
            "eef_position": [0.0, 0.0, 0.05],
            "contact_count": contact_count,
        }

    def state(self) -> dict:
        return self.current

    def step(self, action: np.ndarray) -> dict:
        sequence = self.sequences[self.attempt_index]
        self.current = sequence[min(self.step_index, len(sequence) - 1)]
        self.step_index += 1
        return {"images": {}}

    def recover(
        self,
        *,
        steps: int,
        joint_positions: tuple[float, ...] | None,
    ) -> dict:
        self.recovery_calls.append((steps, joint_positions))
        self.attempt_index += 1
        self.step_index = 0
        self.current = self._state(contact_count=0, object_height=0.0)
        return {"images": {}}


def test_task_grasp_retry_recovers_after_unconfirmed_contact() -> None:
    policy = _RetryPolicy()
    env = _RetryEnv()
    checker = ThreePhaseChecker(
        {
            "approach_distance_m": 0.1,
            "close_contact_count": 2,
            "close_hold_steps": 1,
            "lift_height_m": 0.02,
            "lift_hold_steps": 1,
        }
    )
    retry = GraspRetryConfig(
        enabled=True,
        max_attempts=2,
        approach_timeout_steps=4,
        close_timeout_steps=1,
        recovery_steps=3,
        seed_stride=100,
    )

    result = _execute_trial(
        policy=policy,
        env=env,
        checker=checker,
        obs={"images": {}},
        horizon=2,
        max_chunks=2,
        base_seed=7,
        retry_config=retry,
    )

    assert result["success"] is True
    assert result["attempt_count"] == 2
    assert result["retry_count"] == 1
    assert result["first_attempt_success"] is False
    assert result["recovered_success"] is True
    assert result["recovery_steps"] == 3
    assert result["attempts"][0]["retry_reason"] == "close_timeout"
    assert policy.reset_seeds == [7, 107]
    assert env.recovery_calls == [(3, None)]


def test_task_grasp_retry_requires_safe_configuration() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        GraspRetryConfig(enabled=True, max_attempts=1).validate()
    with pytest.raises(ValueError, match="13 finite values"):
        GraspRetryConfig.from_execution(
            {
                "task_grasp_retry": {
                    "enabled": True,
                    "max_attempts": 2,
                    "recovery_joint_positions": [0.0] * 12,
                }
            }
        )


def test_task_grasp_retry_detects_approach_timeout() -> None:
    checker = ThreePhaseChecker({"approach_distance_m": 0.1})
    controller = GraspRetryController(
        GraspRetryConfig(
            enabled=True,
            max_attempts=2,
            approach_timeout_steps=2,
            close_timeout_steps=2,
        )
    )
    controller.begin_attempt(0)
    far_state = {
        "object_position": [0.0, 0.0, 0.0],
        "eef_position": [1.0, 0.0, 0.0],
        "contact_count": 0,
    }
    checker.update(far_state)
    controller.observe(checker)
    assert controller.retry_reason(checker) is None
    checker.update(far_state)
    controller.observe(checker)
    assert controller.retry_reason(checker) == "approach_timeout"


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
