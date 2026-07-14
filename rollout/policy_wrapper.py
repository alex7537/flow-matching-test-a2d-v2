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

from flow_matching_test.policies.base import ActionPolicy
from flow_matching_test.policies.factory import build_policy
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
        if self.stats.get("action_semantics") != "executed_joint_position":
            raise ValueError("bundle action stats do not use executed_joint_position semantics")
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

    @torch.inference_mode()
    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        images = obs.get("images")
        if not isinstance(images, dict):
            raise TypeError("obs['images'] must be a camera-name to RGB image mapping")
        for sim_name, model_key in self.camera_map.items():
            if sim_name not in images:
                raise KeyError(f"missing rollout camera image: {sim_name}")
            self._history[model_key].append(self._prepare_image(images[sim_name]))
        self._state_history.append(self._normalize_state(obs["proprio"]))

        model_obs: dict[str, torch.Tensor] = {}
        for model_key, history in self._history.items():
            while len(history) < self.history_steps:
                history.appendleft(history[0].clone())
            model_obs[model_key] = torch.stack(list(history), dim=0).unsqueeze(0).to(self.device)
        while len(self._state_history) < self.history_steps:
            self._state_history.appendleft(self._state_history[0].copy())
        model_obs["proprio"] = torch.from_numpy(
            np.concatenate(list(self._state_history), axis=0)
        ).unsqueeze(0).to(self.device)

        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + self.calls)
            sampled = self.model.sample_actions(model_obs).action[0]
        self.calls += 1
        action = sampled.cpu().numpy()
        self._validate_action_range(action)
        return action
