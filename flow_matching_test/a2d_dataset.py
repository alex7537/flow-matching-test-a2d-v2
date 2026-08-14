"""A2D HDF5 data pipeline for flow matching policy training.

Assumed HDF5 layout per episode file (adjust keys in DataConfig if different):

    episode_0001.hdf5
    ├── observations/
    │   ├── qpos            (T, D_state)  float
    │   ├── rgb_head        (T,) encoded jpeg bytes  OR  (T, H, W, 3) uint8
    │   ├── rgb_left_hand   ...
    │   └── rgb_right_hand  ...
    └── action              (T, D_action) float

Usage:
    # 1. one-off: build index cache + normalization stats
    python a2d_dataset.py --data-dir /path/to/success --compute-stats

    # 2. in training code:
    from a2d_dataset import A2DConfig, build_dataloaders
    cfg = A2DConfig(data_dir=..., ...)
    train_loader, val_loader, norm_stats = build_dataloaders(cfg)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from flow_matching_test.segmentation import (
    DEFAULT_KEYFRAME_THRESHOLD,
    DEFAULT_MOTION_THRESHOLD,
    SEGMENTATION_VERSION,
)

try:
    from turbojpeg import TurboJPEG  # pip install PyTurboJPEG (needs libturbojpeg)
    _JPEG = TurboJPEG()
except Exception:
    _JPEG = None
    import cv2


STATE_DIM = 13
ACTION_DIM = 13
STATE_LAYOUT = "arm2_pos(7)+hand2_pos(6)"
ACTION_LAYOUT = "arm2_pos(7)+hand2_pos(6)"
STATS_SCHEMA_VERSION = 2


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class A2DConfig:
    data_dir: str = ""
    image_keys: tuple = ("rgb_head", "rgb_left_hand", "rgb_right_hand")
    obs_group: str = "observations"         # h5 group holding qpos + images
    state_key: str = "qpos"
    action_key: str = "action"              # at file root; use "observations/action" style if nested
    image_size: int = 224
    history_steps: int = 1
    action_horizon: int = 16
    action_offset_steps: int = 1
    include_tail_padded_windows: bool = False
    transition_oversample_factor: int = 1
    lift_oversample_factor: int = 1
    transition_threshold: float = DEFAULT_KEYFRAME_THRESHOLD
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    allow_duplicate_episodes: bool = False
    range_eps: float = 1.0e-4
    val_ratio: float = 0.1
    seed: int = 42
    # dataloader
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    max_open_hdf5_files: int = 64
    # augmentation (train only)
    aug_brightness: float = 0.2
    aug_contrast: float = 0.4
    aug_saturation: float = 0.2
    aug_hue: float = 0.05
    aug_random_crop_pad: int = 8            # pixels of pad-then-crop jitter, 0 disables
    # cache/stats files (written inside data_dir's parent)
    index_cache: str = "index_cache.json"
    norm_stats: str = "norm_stats.json"
    dataset_manifest: str | None = None
    split_manifest: str | None = None


# --------------------------------------------------------------------------- #
# Index: scan episodes once, cache lengths
# --------------------------------------------------------------------------- #
def build_index(cfg: A2DConfig, force: bool = False) -> list[dict]:
    data_dir = Path(cfg.data_dir)
    cache_path = data_dir / cfg.index_cache
    files = sorted(data_dir.glob("*.hdf5")) + sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .hdf5/.h5 files under {data_dir}")
    inventory = [
        {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in files
    ]
    if cache_path.exists() and not force:
        with open(cache_path) as f:
            cached = json.load(f)
        if (
            isinstance(cached, dict)
            and cached.get("version") == 4
            and cached.get("image_keys") == list(cfg.image_keys)
            and cached.get("file_inventory") == inventory
            and cached.get("segmentation_version") == SEGMENTATION_VERSION
            and np.isclose(cached.get("motion_threshold", np.nan), cfg.motion_threshold)
            and np.isclose(cached.get("keyframe_threshold", np.nan), cfg.transition_threshold)
        ):
            episodes = cached["episodes"]
            for episode in episodes:
                episode["path"] = str(data_dir / episode["file_name"])
            duplicates = [item for item in cached.get("duplicate_groups", []) if len(item) > 1]
            if duplicates and not cfg.allow_duplicate_episodes:
                raise ValueError(f"duplicate episode content detected: {duplicates}")
            return episodes

    episodes = []
    for p in files:
        with h5py.File(p, "r", libver="latest") as f:
            if cfg.obs_group not in f or cfg.state_key not in f[cfg.obs_group]:
                raise ValueError(f"{p}: missing {cfg.obs_group}/{cfg.state_key}")
            state = f[cfg.obs_group][cfg.state_key]
            action = f[cfg.action_key]
            if state.ndim != 2 or state.shape[1] != STATE_DIM:
                raise ValueError(f"{p}: state must have shape (T,{STATE_DIM}), got {state.shape}")
            if action.ndim != 2 or action.shape[1] != ACTION_DIM:
                raise ValueError(f"{p}: action must have shape (T,{ACTION_DIM}), got {action.shape}")
            T = int(action.shape[0])
            if state.shape[0] != T:
                raise ValueError(f"{p}: state/action length mismatch {state.shape[0]} != {T}")
            if str(f.attrs.get("qpos_layout", "")) != STATE_LAYOUT:
                raise ValueError(f"{p}: qpos_layout must be {STATE_LAYOUT!r}")
            if str(f.attrs.get("action_layout", "")) != ACTION_LAYOUT:
                raise ValueError(f"{p}: action_layout must be {ACTION_LAYOUT!r}")
            if str(f.attrs.get("action_semantics", "")) != "executed_joint_position":
                raise ValueError(f"{p}: action_semantics must be 'executed_joint_position'")
            if int(f.attrs.get("segmentation_version", -1)) != SEGMENTATION_VERSION:
                raise ValueError(
                    f"{p}: segmentation_version must be {SEGMENTATION_VERSION}; reprocess the episode"
                )
            if str(f.attrs.get("segmentation_source", "")) != "executed_joint_position":
                raise ValueError(f"{p}: segmentation_source must be 'executed_joint_position'")
            if not np.isclose(
                float(f.attrs.get("segmentation_motion_threshold", np.nan)), cfg.motion_threshold
            ):
                raise ValueError(f"{p}: segmentation motion threshold does not match config")
            if not np.isclose(
                float(f.attrs.get("segmentation_keyframe_threshold", np.nan)), cfg.transition_threshold
            ):
                raise ValueError(f"{p}: segmentation keyframe threshold does not match config")
            for key in ("segment_type", "arm_keyframe"):
                if key not in f or f[key].shape != (T,):
                    raise ValueError(f"{p}: {key} must have shape ({T},)")
            segment_values = np.asarray(f["segment_type"][:], dtype=np.uint8)
            if not np.isin(segment_values, (0, 1, 2)).all():
                raise ValueError(f"{p}: segment_type contains values outside 0/1/2")
            for key in cfg.image_keys:
                if key not in f[cfg.obs_group]:
                    raise ValueError(f"{p}: missing selected image key {cfg.obs_group}/{key}")
                if f[cfg.obs_group][key].shape[0] != T:
                    raise ValueError(f"{p}: {key} length does not match action length {T}")
            digest = hashlib.sha256()
            digest.update(np.asarray(state[:], dtype=np.float32).tobytes())
            digest.update(np.asarray(action[:], dtype=np.float32).tobytes())
            for key in cfg.image_keys:
                image_dataset = f[cfg.obs_group][key]
                for frame_index in range(T):
                    digest.update(np.asarray(image_dataset[frame_index], dtype=np.uint8).tobytes())
        episodes.append({
            "path": str(p),
            "file_name": p.name,
            "length": T,
            "content_hash": digest.hexdigest(),
        })

    paths_by_hash: dict[str, list[str]] = {}
    for episode in episodes:
        paths_by_hash.setdefault(episode["content_hash"], []).append(episode["path"])
    duplicate_groups = [paths for paths in paths_by_hash.values() if len(paths) > 1]
    if duplicate_groups and not cfg.allow_duplicate_episodes:
        raise ValueError(f"duplicate episode content detected: {duplicate_groups}")

    with open(cache_path, "w") as f:
        json.dump(
            {
                "version": 4,
                "image_keys": list(cfg.image_keys),
                "file_inventory": inventory,
                "segmentation_version": SEGMENTATION_VERSION,
                "motion_threshold": float(cfg.motion_threshold),
                "keyframe_threshold": float(cfg.transition_threshold),
                "duplicate_groups": duplicate_groups,
                "episodes": episodes,
            },
            f,
            indent=2,
        )
    print(f"[index] {len(episodes)} episodes, "
          f"{sum(e['length'] for e in episodes)} frames -> {cache_path}")
    return episodes


def split_episodes(episodes: list[dict], val_ratio: float, seed: int):
    """Episode-level split. Sample-level splits leak temporally-adjacent frames."""
    idx = list(range(len(episodes)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_ids = set(idx[:n_val])
    train = [e for i, e in enumerate(episodes) if i not in val_ids]
    val = [e for i, e in enumerate(episodes) if i in val_ids]
    return train, val


def split_episodes_from_manifest(
    cfg: A2DConfig,
    episodes: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    if not cfg.dataset_manifest or not cfg.split_manifest:
        train, val = split_episodes(episodes, cfg.val_ratio, cfg.seed)
        return train, val, {}

    data_dir = Path(cfg.data_dir)
    dataset_path = data_dir / cfg.dataset_manifest
    split_path = data_dir / cfg.split_manifest
    dataset_bytes = dataset_path.read_bytes()
    dataset_manifest = json.loads(dataset_bytes)
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))

    actual_dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    expected_dataset_sha = str(split_manifest.get("dataset_manifest_sha256", ""))
    if actual_dataset_sha != expected_dataset_sha:
        raise ValueError(
            "split manifest does not match the current dataset manifest: "
            f"expected {expected_dataset_sha}, actual {actual_dataset_sha}"
        )
    if int(split_manifest.get("seed", -1)) != int(cfg.seed):
        raise ValueError("split manifest seed does not match the data config")
    if not np.isclose(split_manifest.get("val_ratio", np.nan), cfg.val_ratio):
        raise ValueError("split manifest val_ratio does not match the data config")

    episodes_by_name = {str(episode["file_name"]): episode for episode in episodes}
    manifest_episodes = dataset_manifest.get("episodes", [])
    manifest_by_name = {
        str(item["file_name"]): item
        for item in manifest_episodes
        if isinstance(item, dict) and "file_name" in item
    }
    if set(manifest_by_name) != set(episodes_by_name):
        raise ValueError("dataset manifest episode membership does not match the current dataset")
    for name, episode in episodes_by_name.items():
        item = manifest_by_name[name]
        if str(item.get("content_hash", "")) != str(episode["content_hash"]):
            raise ValueError(f"dataset manifest content hash mismatch for {name}")

    train_names = [str(name) for name in split_manifest.get("train_episodes", [])]
    val_names = [str(name) for name in split_manifest.get("val_episodes", [])]
    if len(train_names) != len(set(train_names)) or len(val_names) != len(set(val_names)):
        raise ValueError("split manifest contains duplicate episode names")
    if set(train_names).intersection(val_names):
        raise ValueError("split manifest train/val membership overlaps")
    if set(train_names).union(val_names) != set(episodes_by_name):
        raise ValueError("split manifest membership does not match the current dataset")

    split_manifest["split_manifest_sha256"] = hashlib.sha256(
        split_path.read_bytes()
    ).hexdigest()
    train = [episodes_by_name[name] for name in train_names]
    val = [episodes_by_name[name] for name in val_names]
    return train, val, split_manifest


def train_episode_binding(episodes: list[dict]) -> tuple[list[str], str]:
    hashes = sorted(str(episode["content_hash"]) for episode in episodes)
    digest = hashlib.sha256(
        json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return hashes, digest


def validate_stats_binding(stats: dict, train_episodes: list[dict]) -> None:
    hashes, digest = train_episode_binding(train_episodes)
    if stats.get("schema_version") != STATS_SCHEMA_VERSION:
        raise ValueError(
            f"norm stats schema_version must be {STATS_SCHEMA_VERSION}; recompute stats"
        )
    if stats.get("train_episode_hashes") != hashes or stats.get("train_episode_digest") != digest:
        raise ValueError(
            "norm stats do not match the current train episode split; "
            "create a new dataset version and recompute stats"
        )
    if stats.get("action_semantics") != "executed_joint_position":
        raise ValueError("norm stats action semantics do not match the dataset contract")


# --------------------------------------------------------------------------- #
# Absolute joint-target normalization stats
# --------------------------------------------------------------------------- #
def compute_norm_stats(cfg: A2DConfig, episodes: list[dict],
                       max_frames_per_ep: int = 500) -> dict:
    """Subsampled pass over training episodes. Saved once, reused at deploy time."""
    states, actions = [], []
    for e in episodes:
        with h5py.File(e["path"], "r", libver="latest") as f:
            T = e["length"]
            sel = np.linspace(0, T - 1, min(T, max_frames_per_ep)).astype(int)
            states.append(f[cfg.obs_group][cfg.state_key][:][sel])
            actions.append(f[cfg.action_key][:][sel])
    states = np.concatenate(states, 0).astype(np.float64)
    actions = np.concatenate(actions, 0).astype(np.float64)

    def minmax(x):
        lo, hi = x.min(axis=0), x.max(axis=0)
        span = np.maximum(hi - lo, float(cfg.range_eps))
        unique = [int(np.unique(np.round(x[:, index], 6)).size) for index in range(x.shape[1])]
        return {
            "min": lo.tolist(),
            "max": hi.tolist(),
            "span": span.tolist(),
            "raw_range": (hi - lo).tolist(),
            "unique_count_1e-6": unique,
            "warnings": [
                {"dim": index, "range": float(hi[index] - lo[index]), "unique_count": unique[index]}
                for index in range(x.shape[1])
                if hi[index] - lo[index] < 1e-3 or unique[index] <= 2
            ],
        }

    train_hashes, train_digest = train_episode_binding(episodes)
    stats = {
        "schema_version": STATS_SCHEMA_VERSION,
        "train_episode_hashes": train_hashes,
        "train_episode_digest": train_digest,
        "train_episode_count": len(train_hashes),
        "split_metadata": {"seed": int(cfg.seed), "val_ratio": float(cfg.val_ratio)},
        "action_semantics": "executed_joint_position",
        "normalization": "train_minmax",
        "range_eps": float(cfg.range_eps),
        "state": minmax(states),
        "action": minmax(actions),
    }
    out = Path(cfg.data_dir) / cfg.norm_stats
    if out.exists():
        with open(out) as f:
            existing = json.load(f)
        if (
            existing.get("schema_version") == STATS_SCHEMA_VERSION
            and existing.get("train_episode_digest") == train_digest
            and np.isclose(existing.get("range_eps", np.nan), cfg.range_eps)
        ):
            print(f"[stats] unchanged train split -> {out}")
            return existing
        raise FileExistsError(
            f"refusing to overwrite {out} with stats for a different train split; "
            "use a new dataset directory or norm_stats filename"
        )
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[stats] state dim={states.shape[1]}, action dim={actions.shape[1]} -> {out}")
    return stats


def load_norm_stats(cfg: A2DConfig, train_episodes: list[dict]) -> dict:
    with open(Path(cfg.data_dir) / cfg.norm_stats) as f:
        stats = json.load(f)
    validate_stats_binding(stats, train_episodes)
    if not np.isclose(stats.get("range_eps", np.nan), cfg.range_eps):
        raise ValueError("norm stats range_eps does not match the data config")
    for field in ("state", "action"):
        raw_range = np.asarray(stats[field].get("raw_range", stats[field]["span"]), dtype=np.float32)
        dims = np.flatnonzero(raw_range < cfg.range_eps).tolist()
        if dims:
            warnings.warn(
                f"{field} dims {dims} have range < range_eps={cfg.range_eps:g}; "
                "normalization will use the protected range",
                stacklevel=2,
            )
    return stats


def normalize(x: np.ndarray, s: dict) -> np.ndarray:
    """Map training min..max -> [-1, 1]."""
    lo = np.asarray(s["min"], dtype=np.float32)
    span = np.maximum(np.asarray(s["span"], dtype=np.float32), 1.0e-4)
    return np.clip(2.0 * (x - lo) / span - 1.0, -3.0, 3.0)


def denormalize(x: np.ndarray, s: dict) -> np.ndarray:
    lo = np.asarray(s["min"], dtype=np.float32)
    span = np.maximum(np.asarray(s["span"], dtype=np.float32), 1.0e-4)
    return (x + 1.0) * 0.5 * span + lo


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class A2DFlowDataset(Dataset):
    """Returns per sample:
        images:      (n_cams * history, 3, S, S) float32 in [0, 1]
        state:       (history * D_state,)        normalized
        action:      (horizon, 13)               normalized future executed joint position
        action_mask: (horizon,)                  1 = real, 0 = tail padding
    """

    def __init__(self, cfg: A2DConfig, episodes: list[dict],
                 norm_stats: dict, train: bool):
        if cfg.action_offset_steps < 0:
            raise ValueError("action_offset_steps must be >= 0")
        self.cfg = cfg
        self.episodes = episodes
        self.stats = norm_stats
        self.train = train
        self.epoch = 0
        # flat sample index: (ep_idx, t); every t with a full history is valid
        self.samples: list[tuple[int, int]] = []
        h = cfg.history_steps - 1
        for i, e in enumerate(episodes):
            for t in range(h, e["length"] - cfg.action_offset_steps):
                self.samples.append((i, t))
        if cfg.max_open_hdf5_files < 1:
            raise ValueError("max_open_hdf5_files must be >= 1")
        # Per-worker lazy LRU handles — NEVER open h5py.File before fork. Keep
        # this bounded because large datasets can exceed the worker fd limit.
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    # ---- h5 handle management ------------------------------------------- #
    def _file(self, ep_idx: int) -> h5py.File:
        f = self._handles.get(ep_idx)
        if f is not None:
            self._handles.move_to_end(ep_idx)
            return f
        if len(self._handles) >= self.cfg.max_open_hdf5_files:
            _, oldest = self._handles.popitem(last=False)
            oldest.close()
        f = h5py.File(self.episodes[ep_idx]["path"], "r",
                      libver="latest", swmr=True)
        self._handles[ep_idx] = f
        return f

    def close(self) -> None:
        handles = getattr(self, "_handles", None)
        while handles:
            _, handle = handles.popitem(last=False)
            handle.close()

    def __del__(self) -> None:
        self.close()

    # ---- image loading --------------------------------------------------- #
    def _decode(self, buf: np.ndarray) -> np.ndarray:
        raw = buf.tobytes() if isinstance(buf, np.ndarray) else bytes(buf)
        if _JPEG is not None:
            return _JPEG.decode(raw)                       # BGR
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)          # BGR

    def _load_frame(self, ep_idx: int, cam: str, t: int) -> np.ndarray:
        cfg = self.cfg
        ds = self._file(ep_idx)[cfg.obs_group][cam]
        item = ds[t]
        img = item if (item.ndim == 3) else self._decode(item)
        return img[..., ::-1]                               # -> RGB

    def _sample_aug_params(self, rng: np.random.Generator) -> dict[str, float | int]:
        cfg = self.cfg
        pad = cfg.aug_random_crop_pad
        return {
            "crop_x": int(rng.integers(0, 2 * pad + 1)) if pad > 0 else 0,
            "crop_y": int(rng.integers(0, 2 * pad + 1)) if pad > 0 else 0,
            "brightness": float(rng.uniform(1 - cfg.aug_brightness, 1 + cfg.aug_brightness)),
            "contrast": float(rng.uniform(1 - cfg.aug_contrast, 1 + cfg.aug_contrast)),
            "saturation": float(rng.uniform(1 - cfg.aug_saturation, 1 + cfg.aug_saturation)),
            "hue": float(rng.uniform(-cfg.aug_hue, cfg.aug_hue)),
        }

    def _resize_aug(self, img: np.ndarray, params: dict[str, float | int]) -> np.ndarray:
        import cv2 as _cv2
        cfg, S = self.cfg, self.cfg.image_size
        img = _cv2.resize(img, (S, S), interpolation=_cv2.INTER_AREA)
        if self.train and cfg.aug_random_crop_pad > 0:
            p = cfg.aug_random_crop_pad
            img = _cv2.copyMakeBorder(img, p, p, p, p, _cv2.BORDER_REFLECT)
            x, y = int(params["crop_x"]), int(params["crop_y"])
            img = img[y:y + S, x:x + S]
        img = img.astype(np.float32) / 255.0
        if self.train:
            img *= float(params["brightness"])
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = (img - mean) * float(params["contrast"]) + mean
            hsv = _cv2.cvtColor(np.clip(img, 0.0, 1.0), _cv2.COLOR_RGB2HSV)
            hsv[..., 1] *= float(params["saturation"])
            hsv[..., 0] = (hsv[..., 0] + float(params["hue"]) * 360.0) % 360.0
            img = _cv2.cvtColor(hsv, _cv2.COLOR_HSV2RGB)
            img = np.clip(img, 0.0, 1.0)
        return img

    # ---- main ------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> dict:
        cfg = self.cfg
        ep_idx, t = self.samples[idx]
        ep = self.episodes[ep_idx]
        f = self._file(ep_idx)
        augmentation_seed = (
            int(cfg.seed) * 1_000_003 + self.epoch * 100_003 + int(idx)
        ) % (2**63 - 1)
        rng = np.random.default_rng(augmentation_seed)
        aug_params = {cam: self._sample_aug_params(rng) for cam in cfg.image_keys}

        # images: history frames per camera, shared aug params per sample is
        # optional; here each frame is augmented consistently enough for IL
        frames = []
        for dt in range(cfg.history_steps - 1, -1, -1):     # oldest -> newest
            for cam in cfg.image_keys:
                img = self._load_frame(ep_idx, cam, t - dt)
                img = self._resize_aug(img, aug_params[cam])
                frames.append(torch.from_numpy(img).permute(2, 0, 1))
        images = torch.stack(frames, 0)                     # (cams*hist, 3, S, S)

        # state (history, D) -> flat, normalized
        s0 = t - (cfg.history_steps - 1)
        state = f[cfg.obs_group][cfg.state_key][s0:t + 1].astype(np.float32)
        state = normalize(state, self.stats["state"]).reshape(-1)

        # Future action chunk [t+offset, t+offset+H) with tail padding + mask.
        T, H = ep["length"], cfg.action_horizon
        start = t + cfg.action_offset_steps
        end = min(start + H, T)
        chunk = f[cfg.action_key][start:end].astype(np.float32)
        n_pad = H - chunk.shape[0]
        mask = np.ones(H, dtype=np.float32)
        if n_pad > 0:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], n_pad, 0)], 0)
            mask[H - n_pad:] = 0.0

        action = normalize(chunk, self.stats["action"])

        return {
            "images": images,
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(mask),
        }


class A2DProcessedWindowDataset(A2DFlowDataset):
    """Adapter from processed A2D HDF5s to the model's ``obs/action`` contract."""

    def __init__(self, *, cfg: A2DConfig, split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"unsupported split: {split}")
        episodes = build_index(cfg)
        train_eps, val_eps, split_manifest = split_episodes_from_manifest(cfg, episodes)
        stats = load_norm_stats(cfg, train_eps)
        stats.setdefault("range_eps", float(cfg.range_eps))
        for field in ("state", "action"):
            stats[field]["span"] = np.maximum(
                np.asarray(stats[field]["span"], dtype=np.float32), cfg.range_eps
            ).tolist()
        if stats.get("action_semantics") != "executed_joint_position":
            raise ValueError("norm_stats.json must use executed_joint_position semantics")
        selected = train_eps if split == "train" else val_eps
        super().__init__(cfg, selected, stats, train=split == "train")
        self.split_manifest = split_manifest

        complete_samples = [
            (ep_idx, t)
            for ep_idx, t in self.samples
            if t + cfg.action_offset_steps + cfg.action_horizon
            <= self.episodes[ep_idx]["length"]
        ]
        self.complete_sample_count = len(complete_samples)
        if not cfg.include_tail_padded_windows:
            self.samples = complete_samples
        self.tail_padded_sample_count = len(self.samples) - self.complete_sample_count
        if not self.samples:
            raise ValueError(f"split {split} has no action windows")
        self.base_sample_count = len(self.samples)
        arm_keyframe_samples: list[tuple[int, int]] = []
        lift_samples: list[tuple[int, int]] = []
        self.segment_by_sample: dict[tuple[int, int], int] = {}
        self.lift_by_sample: dict[tuple[int, int], bool] = {}
        samples_by_episode: dict[int, list[tuple[int, int]]] = {}
        for sample in self.samples:
            samples_by_episode.setdefault(sample[0], []).append(sample)
        for ep_idx, episode in enumerate(self.episodes):
            with h5py.File(episode["path"], "r") as file:
                segment_type = np.asarray(file["segment_type"][:], dtype=np.uint8)
                arm_keyframe = np.asarray(file["arm_keyframe"][:], dtype=bool)
                phase = [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in file["phase"][:]
                ]
            for sample in samples_by_episode.get(ep_idx, []):
                _, start = sample
                start += cfg.action_offset_steps
                stop = min(start + cfg.action_horizon, episode["length"])
                has_arm_keyframe = bool(np.any(arm_keyframe[start:stop]))
                has_lift = any("lift" in value.lower() for value in phase[start:stop])
                self.segment_by_sample[sample] = int(np.max(segment_type[start:stop]))
                self.lift_by_sample[sample] = has_lift
                if has_arm_keyframe:
                    arm_keyframe_samples.append(sample)
                if has_lift:
                    lift_samples.append(sample)
        self.transition_sample_count = len(arm_keyframe_samples)
        self.transition_sample_set = set(arm_keyframe_samples)
        transition_factor = int(cfg.transition_oversample_factor)
        if transition_factor < 1:
            raise ValueError("transition_oversample_factor must be >= 1")
        lift_factor = int(cfg.lift_oversample_factor)
        if lift_factor < 1:
            raise ValueError("lift_oversample_factor must be >= 1")
        if split == "train" and transition_factor > 1:
            self.samples.extend(arm_keyframe_samples * (transition_factor - 1))
        if split == "train" and lift_factor > 1:
            self.samples.extend(lift_samples * (lift_factor - 1))
        self.effective_transition_sample_count = self.transition_sample_count * (
            transition_factor if split == "train" else 1
        )
        self.lift_sample_count = len(lift_samples)
        self.effective_lift_sample_count = self.lift_sample_count * (
            lift_factor if split == "train" else 1
        )
        print(
            f"[sampling:{split}] transition windows {self.transition_sample_count}/"
            f"{self.base_sample_count}; factor="
            f"{transition_factor if split == 'train' else 1}; "
            f"lift windows {self.lift_sample_count}/{self.base_sample_count}; factor="
            f"{lift_factor if split == 'train' else 1}; "
            f"tail padded={self.tail_padded_sample_count}; "
            f"effective samples={len(self.samples)}"
        )

        lo = np.asarray(stats["action"]["min"], dtype=np.float32)
        span = np.asarray(stats["action"]["span"], dtype=np.float32)
        self.action_mean = lo + 0.5 * span
        self.action_std = 0.5 * span
        self.action_dim = int(self.action_mean.shape[0])
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"action stats must have {ACTION_DIM} dims, got {self.action_dim}")
        self.history_steps = int(cfg.history_steps)
        self.action_horizon = int(cfg.action_horizon)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        sample_key = self.samples[idx]
        sample = super().__getitem__(idx)
        stacked = sample["images"]
        num_views = len(self.cfg.image_keys)
        obs = {
            key: torch.stack(
                [stacked[step * num_views + view_idx] for step in range(self.history_steps)],
                dim=0,
            )
            * 2.0
            - 1.0
            for view_idx, key in enumerate(self.cfg.image_keys)
        }
        obs["proprio"] = sample["state"]
        return {
            "obs": obs,
            "action": sample["action"],
            "action_mask": sample["action_mask"],
            "segment_type": torch.tensor(self.segment_by_sample[sample_key], dtype=torch.long),
            "is_lift": torch.tensor(self.lift_by_sample[sample_key], dtype=torch.bool),
            "sample_index": torch.tensor(idx, dtype=torch.long),
        }

    def export_stats(self) -> dict[str, torch.Tensor]:
        return {
            "action_mean": torch.from_numpy(self.action_mean.copy()),
            "action_std": torch.from_numpy(self.action_std.copy()),
        }


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def build_dataloaders(cfg: A2DConfig):
    episodes = build_index(cfg)
    train_eps, val_eps = split_episodes(episodes, cfg.val_ratio, cfg.seed)
    stats = load_norm_stats(cfg, train_eps)

    train_ds = A2DFlowDataset(cfg, train_eps, stats, train=True)
    val_ds = A2DFlowDataset(cfg, val_eps, stats, train=False)
    print(f"[data] train {len(train_ds)} samples / {len(train_eps)} eps | "
          f"val {len(val_ds)} samples / {len(val_eps)} eps")

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers > 0:
        common["prefetch_factor"] = cfg.prefetch_factor
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--compute-stats", action="store_true")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--action-key", default="action")
    ap.add_argument("--state-key", default="qpos")
    ap.add_argument("--obs-group", default="observations")
    ap.add_argument(
        "--image-keys",
        nargs="+",
        default=["rgb_head", "rgb_left_hand", "rgb_right_hand"],
    )
    ap.add_argument("--allow-duplicates", action="store_true")
    ap.add_argument("--norm-stats", default="norm_stats.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD)
    ap.add_argument("--keyframe-threshold", type=float, default=0.1)
    args = ap.parse_args()

    cfg = A2DConfig(data_dir=args.data_dir, action_key=args.action_key,
                    state_key=args.state_key, obs_group=args.obs_group,
                    image_keys=tuple(args.image_keys),
                    allow_duplicate_episodes=args.allow_duplicates,
                    norm_stats=args.norm_stats,
                    seed=args.seed,
                    val_ratio=args.val_ratio,
                    motion_threshold=args.motion_threshold,
                    transition_threshold=args.keyframe_threshold)
    eps = build_index(cfg, force=args.rebuild_index)
    if args.compute_stats:
        train_eps, _ = split_episodes(eps, cfg.val_ratio, cfg.seed)
        compute_norm_stats(cfg, train_eps)
