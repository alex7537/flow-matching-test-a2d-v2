from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

try:
    import rerun as rr
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    rr = None


logger = logging.getLogger(__name__)


def _rgb_chw_to_uint8(frame: torch.Tensor) -> np.ndarray:
    image = frame.detach().cpu().float().numpy()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected RGB frame with shape [3,H,W], got {tuple(image.shape)}")
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    return image


def _to_numpy(array: torch.Tensor) -> np.ndarray:
    return array.detach().cpu().float().numpy()


def _set_time_sequence(name: str, value: int) -> None:
    if rr is None:
        return
    rr.set_time(name, sequence=int(value))


class RerunTrainVisualizer:
    def __init__(
        self,
        *,
        enabled: bool,
        app_name: str,
        spawn: bool = False,
        save_path: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.save_path = str(save_path) if save_path else None

        if not self.enabled:
            return
        if rr is None:
            logger.warning("rerun visualization was enabled but rerun-sdk is not installed; disable visualization.")
            self.enabled = False
            return

        rr.init(app_name, spawn=bool(spawn))
        if self.save_path:
            save_file = Path(self.save_path).expanduser().resolve()
            save_file.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(save_file))
            logger.info("Rerun recording will be saved to %s", save_file)

    def log_epoch_metrics(
        self,
        *,
        epoch: int,
        global_step: int,
        metrics: Mapping[str, float | None],
    ) -> None:
        if not self.enabled:
            return
        _set_time_sequence("epoch", int(epoch))
        _set_time_sequence("global_step", int(global_step))
        for key, value in metrics.items():
            if value is None:
                continue
            rr.log(f"metrics/{key}", rr.Scalars(float(value)))

    def log_sample(
        self,
        *,
        sample_index: int,
        split: str,
        obs: Mapping[str, torch.Tensor],
        gt_action_normalized: torch.Tensor,
        pred_action_normalized: torch.Tensor,
        gt_action: torch.Tensor,
        pred_action: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        _set_time_sequence("sample", int(sample_index))
        self._log_obs_images(split=split, obs=obs)
        self._log_action_chunk(
            root=f"samples/{split}/action_normalized",
            gt_action=gt_action_normalized,
            pred_action=pred_action_normalized,
        )
        self._log_action_chunk(
            root=f"samples/{split}/action",
            gt_action=gt_action,
            pred_action=pred_action,
        )

    def _log_obs_images(self, *, split: str, obs: Mapping[str, torch.Tensor]) -> None:
        for key, value in obs.items():
            tensor = value.detach().cpu()
            if key == "proprio":
                continue
            if tensor.ndim == 5:
                tensor = tensor[0]
            if tensor.ndim != 4:
                raise ValueError(f"Expected observation tensor [T,3,H,W], got {tuple(tensor.shape)}")
            for frame_idx, frame in enumerate(tensor):
                _set_time_sequence("history_step", int(frame_idx))
                rr.log(f"samples/{split}/images/{key}", rr.Image(_rgb_chw_to_uint8(frame)))

    def _log_action_chunk(
        self,
        *,
        root: str,
        gt_action: torch.Tensor,
        pred_action: torch.Tensor,
    ) -> None:
        gt = _to_numpy(gt_action[0] if gt_action.ndim == 3 else gt_action)
        pred = _to_numpy(pred_action[0] if pred_action.ndim == 3 else pred_action)
        if gt.shape != pred.shape:
            raise ValueError(f"GT/pred shape mismatch: {gt.shape} vs {pred.shape}")

        per_step_mse = ((gt - pred) ** 2).mean(axis=-1)
        for step_idx in range(gt.shape[0]):
            _set_time_sequence("action_step", int(step_idx))
            rr.log(f"{root}/gt", rr.Scalars(gt[step_idx].astype(np.float64, copy=False)))
            rr.log(f"{root}/pred", rr.Scalars(pred[step_idx].astype(np.float64, copy=False)))
            rr.log(f"{root}/mse", rr.Scalars(float(per_step_mse[step_idx])))
