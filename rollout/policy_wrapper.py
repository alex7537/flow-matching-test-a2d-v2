from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from flow_matching_test.action_contract import (
    EXECUTED_ACTION_SEMANTICS,
    HYBRID_ACTION_SEMANTICS,
    validate_action_semantics,
)
from flow_matching_test.policies.base import ActionPolicy
from flow_matching_test.policies.factory import build_policy
from flow_matching_test.policies.imle import ImlePolicy
from flow_matching_test.segmentation import SEGMENTATION_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Policy:
    def __init__(self, bundle_dir: str | Path, device: str = "cuda") -> None:
        self.bundle_dir = Path(bundle_dir)
        self.cfg = yaml.safe_load((self.bundle_dir / "config.yaml").read_text())
        self.stats = json.loads((self.bundle_dir / "norm_stats.json").read_text())
        self.manifest = json.loads((self.bundle_dir / "manifest.json").read_text())
        self._validate_bundle()
        self.device = torch.device(device)
        self.model = self._build_model().to(self.device)
        state_dict = torch.load(self.bundle_dir / "ckpt.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        action_stats = self.stats["action"]
        action_mean = np.asarray(action_stats["min"], dtype=np.float32) + 0.5 * np.asarray(
            action_stats["span"], dtype=np.float32
        )
        action_std = 0.5 * np.asarray(action_stats["span"], dtype=np.float32)
        self.action_min = np.asarray(action_stats["min"], dtype=np.float32)
        self.action_max = np.asarray(action_stats["max"], dtype=np.float32)
        self.action_span = np.maximum(
            self.action_max - self.action_min,
            float(self.stats.get("range_eps", 1.0e-4)),
        )
        self.action_range_margin_ratio = float(
            self.cfg["action"].get("range_guard_margin_ratio", 0.1)
        )
        self.model.set_action_stats(
            action_mean=torch.from_numpy(action_mean).to(self.device),
            action_std=torch.from_numpy(action_std).to(self.device),
        )
        self.model.eval()
        self.use_proprio = bool(self.cfg["model"].get("use_proprio", True))
        self.enhanced_proprio = bool(self.cfg["model"].get("enhanced_proprio", False))
        if self.enhanced_proprio and not self.use_proprio:
            raise ValueError("enhanced_proprio bundle requires proprio observations")
        self.history_steps = int(self.cfg["obs"]["history_steps"])
        self.image_size = int(self.cfg["obs"]["image_size"])
        self.camera_map = {
            str(camera["name"]): str(camera["model_key"])
            for camera in self.cfg["obs"]["cameras"]
        }
        self._history: dict[str, deque[torch.Tensor]] = {
            model_key: deque(maxlen=self.history_steps) for model_key in self.camera_map.values()
        }
        self._state_history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.seed = int(self.cfg.get("sampling", {}).get("seed", 42))
        self.execute_horizon = int(self.cfg["action"]["execute_horizon"])
        self._previous_action_normalized: torch.Tensor | None = None
        self._last_raw_state: np.ndarray | None = None
        self._last_joint_delta = np.zeros(13, dtype=np.float32)
        self._last_action_context: np.ndarray | None = None
        self.calls = 0

    def _validate_bundle(self) -> None:
        required = {"ckpt.pt", "norm_stats.json", "config.yaml", "manifest.json"}
        missing = sorted(name for name in required if not (self.bundle_dir / name).is_file())
        if missing:
            raise FileNotFoundError(f"bundle is missing files: {missing}")
        for name, metadata in self.manifest.get("files", {}).items():
            actual = _sha256(self.bundle_dir / name)
            if actual != metadata["sha256"]:
                raise ValueError(f"bundle checksum mismatch for {name}")
        if int(self.cfg.get("schema_version", -1)) != 2:
            raise ValueError("unsupported config schema_version")
        if int(self.manifest.get("schema_version", -1)) != 2:
            raise ValueError("unsupported bundle manifest schema_version")
        stats_semantics = validate_action_semantics(
            str(self.stats.get("action_semantics", EXECUTED_ACTION_SEMANTICS))
        )
        config_semantics = validate_action_semantics(
            str(self.cfg["action"].get("target", EXECUTED_ACTION_SEMANTICS))
        )
        if stats_semantics != config_semantics:
            raise ValueError("bundle action config and stats semantics differ")
        self.action_semantics = config_semantics
        if int(self.cfg["action"]["dim"]) != 13:
            raise ValueError("this rollout harness requires a 13-dimensional action")
        if len(self.cfg["joint_order"]) != 13:
            raise ValueError("joint_order must contain 13 names")
        stats_digest = self.stats.get("train_episode_digest")
        if not stats_digest or self.manifest.get("stats_digest") != stats_digest:
            raise ValueError("bundle stats_digest does not match norm_stats.json")
        provenance = self.manifest.get("data_provenance", {})
        if provenance.get("stats_digest") != stats_digest:
            raise ValueError("bundle data provenance does not match norm_stats.json")
        if int(provenance.get("segmentation_version", -1)) != SEGMENTATION_VERSION:
            raise ValueError("bundle segmentation_version is unsupported")
        self._warn_environment_mismatch()

    def _warn_environment_mismatch(self) -> None:
        try:
            import timm
            timm_version = timm.__version__
        except Exception:
            timm_version = None
        current = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "timm": timm_version,
        }
        training = self.manifest.get("training_environment", {})
        mismatches = {
            key: {"training": training.get(key), "rollout": current.get(key)}
            for key in ("python", "torch", "cuda_build", "timm")
            if training.get(key) != current.get(key)
        }
        if mismatches:
            warnings.warn(f"training/rollout environment differs: {mismatches}", stacklevel=2)

    def _validate_action_range(self, action: np.ndarray) -> None:
        if not np.isfinite(action).all():
            raise ValueError("policy produced non-finite action values")
        margin = self.action_span * self.action_range_margin_ratio
        lower = self.action_min - margin
        upper = self.action_max + margin
        bad = np.flatnonzero(np.any((action < lower) | (action > upper), axis=0))
        if bad.size:
            details = {
                int(index): {
                    "pred_min": float(action[:, index].min()),
                    "pred_max": float(action[:, index].max()),
                    "allowed_min": float(lower[index]),
                    "allowed_max": float(upper[index]),
                }
                for index in bad
            }
            raise ValueError(f"policy action exceeds train range guard: {details}")

    def _build_model(self) -> ActionPolicy:
        model_cfg = dict(self.cfg["model"])
        model_cfg.update(model_cfg.pop("cfm", {}))
        model_cfg["timm_pretrained"] = False
        return build_policy(
            policy_cfg=self.cfg.get("policy", {"type": "flow_matching"}),
            model_cfg=model_cfg,
            image_keys=tuple(camera["model_key"] for camera in self.cfg["obs"]["cameras"]),
            history_steps=int(self.cfg["obs"]["history_steps"]),
            action_dim=int(self.cfg["action"]["dim"]),
            action_horizon=int(self.cfg["action"]["chunk_size"]),
        )

    def reset(self) -> None:
        for history in self._history.values():
            history.clear()
        self._state_history.clear()
        self._previous_action_normalized = None
        self._last_raw_state = None
        self._last_joint_delta = np.zeros(13, dtype=np.float32)
        self._last_action_context = None
        self.calls = 0

    def _prepare_image(self, image: Any) -> torch.Tensor:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(f"camera image must have shape [H,W,3/4], got {array.shape}")
        array = np.ascontiguousarray(array[..., :3])
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        if float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return tensor * 2.0 - 1.0

    def _normalize_state(self, state: Any) -> np.ndarray:
        value = np.asarray(state, dtype=np.float32)
        if value.shape != (13,):
            raise ValueError(f"proprio must have shape (13,), got {value.shape}")
        stats = self.stats["state"]
        lo = np.asarray(stats["min"], dtype=np.float32)
        span = np.maximum(np.asarray(stats["span"], dtype=np.float32), 1.0e-4)
        return np.clip(2.0 * (value - lo) / span - 1.0, -3.0, 3.0)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        stats = self.stats["action"]
        lo = np.asarray(stats["min"], dtype=np.float32)
        span = np.maximum(np.asarray(stats["span"], dtype=np.float32), 1.0e-4)
        return np.clip(2.0 * (action - lo) / span - 1.0, -3.0, 3.0)

    def record_executed_action(self, action: Any) -> None:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (13,) or not np.isfinite(value).all():
            raise ValueError("executed action must contain 13 finite values")
        self._last_action_context = value.copy()

    def record_execution_feedback(self, action: Any, proprio: Any) -> None:
        state = np.asarray(proprio, dtype=np.float32)
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError("execution feedback proprio must contain 13 finite values")
        self.record_executed_action(action)
        assert self._last_action_context is not None
        if self.action_semantics == HYBRID_ACTION_SEMANTICS:
            self._last_action_context[:7] = state[:7]
        elif self.action_semantics == EXECUTED_ACTION_SEMANTICS:
            self._last_action_context[:] = state
        self._last_joint_delta = (
            np.zeros(13, dtype=np.float32)
            if self._last_raw_state is None
            else state - self._last_raw_state
        )
        self._last_raw_state = state.copy()

    @torch.inference_mode()
    def infer(self, obs: dict[str, Any], *, execute_horizon: int | None = None) -> np.ndarray:
        images = obs.get("images")
        if not isinstance(images, dict):
            raise TypeError("obs['images'] must be a camera-name to RGB image mapping")
        for sim_name, model_key in self.camera_map.items():
            if sim_name not in images:
                raise KeyError(f"missing rollout camera image: {sim_name}")
            self._history[model_key].append(self._prepare_image(images[sim_name]))
        current_raw_state: np.ndarray | None = None
        if self.use_proprio:
            if "proprio" not in obs:
                raise KeyError("missing proprio observation for a proprio-conditioned policy")
            current_raw_state = np.asarray(obs["proprio"], dtype=np.float32)
            if current_raw_state.shape != (13,) or not np.isfinite(current_raw_state).all():
                raise ValueError("proprio must contain 13 finite values")
            self._state_history.append(self._normalize_state(current_raw_state))

        model_obs: dict[str, torch.Tensor] = {}
        for model_key, history in self._history.items():
            while len(history) < self.history_steps:
                history.appendleft(history[0].clone())
            model_obs[model_key] = torch.stack(list(history), dim=0).unsqueeze(0).to(self.device)
        if self.use_proprio:
            while len(self._state_history) < self.history_steps:
                self._state_history.appendleft(self._state_history[0].copy())
            model_obs["proprio"] = torch.from_numpy(
                np.concatenate(list(self._state_history), axis=0)
            ).unsqueeze(0).to(self.device)
        if self.enhanced_proprio:
            assert current_raw_state is not None
            if self._last_raw_state is None:
                joint_delta = np.zeros(13, dtype=np.float32)
            elif np.array_equal(current_raw_state, self._last_raw_state):
                joint_delta = self._last_joint_delta
            else:
                joint_delta = current_raw_state - self._last_raw_state
            previous_action = (
                current_raw_state
                if self._last_action_context is None
                else self._last_action_context
            )
            state_span = np.maximum(
                np.asarray(self.stats["state"]["span"], dtype=np.float32), 1.0e-4
            )
            action_span = np.maximum(
                np.asarray(self.stats["action"]["span"], dtype=np.float32), 1.0e-4
            )
            hand_scale = np.maximum(state_span[7:], action_span[7:])
            model_obs["joint_delta"] = torch.from_numpy(
                np.clip(2.0 * joint_delta / state_span, -3.0, 3.0)
            ).unsqueeze(0).to(self.device)
            model_obs["previous_action"] = torch.from_numpy(
                self._normalize_action(previous_action)
            ).unsqueeze(0).to(self.device)
            model_obs["hand_tracking_error"] = torch.from_numpy(
                np.clip(
                    2.0 * (previous_action[7:] - current_raw_state[7:]) / hand_scale,
                    -3.0,
                    3.0,
                )
            ).unsqueeze(0).to(self.device)
            self._last_raw_state = current_raw_state.copy()

        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + self.calls)
            horizon = self.execute_horizon if execute_horizon is None else int(execute_horizon)
            if not 1 <= horizon <= int(self.cfg["action"]["chunk_size"]):
                raise ValueError("execute_horizon is outside the action chunk")
            if isinstance(self.model, ImlePolicy) and self._previous_action_normalized is not None:
                result = self.model.sample_actions_bidirectional(
                    model_obs,
                    previous_action_normalized=self._previous_action_normalized,
                    execute_horizon=horizon,
                )
            else:
                result = self.model.sample_actions(model_obs)
            self._previous_action_normalized = result.action_normalized.detach().clone()
            sampled = result.action[0]
        self.calls += 1
        action = sampled.cpu().numpy()
        self._validate_action_range(action)
        return action
