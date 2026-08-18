"""Preprocess raw A2D grasp episodes into compact training HDF5s.

Raw schema (per episode, ~546 MB):
    trajectory/arm2_pos          (T, 7)   measured arm joints
    trajectory/hand2_pos         (T, 6)   measured hand joints
    trajectory/waist_pos         (T, 2)   (optional state)
    trajectory/phase             (T,)     phase strings (kept for filtering)
    trajectory/cameras/rgb_*     (T, 480, 640, 3) uint8, LZF,
                                 chunks=(17,60,80,1)  <- 17x read amplification

Output schema (per episode, ~10 MB), matches a2d_dataset.py defaults:
    observations/qpos        (T, 13) float32   arm2_pos ++ hand2_pos
    observations/<selected RGB key> (T,) vlen uint8 (jpeg bytes, 224x224)
    action                   (T, 13) float32   arm2_pos ++ hand2_pos
    phase                    (T,)    vlen str
    segment_type             (T,)    uint8     0=static,1=continuous,2=keyframe
    arm_keyframe             (T,)    uint8     executed arm keyframe mask
    attrs: success, object_name, num_steps, source_file

Usage:
    python preprocess_a2d.py \
        --src /home/psibot/Downloads/flow-matching-test/a2d_curobo_collect_run1/success \
        --dst /home/psibot/data/a2d_processed \
        --image-keys rgb_head rgb_right_hand \
        --image-size 224 --jpeg-quality 92 --workers 4

Then point data.data_dir at --dst and run:
    python a2d_dataset.py --data-dir /home/psibot/data/a2d_processed --compute-stats
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flow_matching_test.action_contract import (
    ACTION_LAYOUTS,
    EXECUTED_ACTION_SEMANTICS,
    HYBRID_ACTION_SEMANTICS,
    action_layout,
    validate_action_semantics,
)
from flow_matching_test.segmentation import (
    DEFAULT_KEYFRAME_THRESHOLD,
    DEFAULT_MOTION_THRESHOLD,
    SEGMENTATION_VERSION,
    compute_executed_action_segments,
)

DEFAULT_IMAGE_KEYS = ("rgb_head", "rgb_left_hand", "rgb_right_hand")


def encode_frames(frames: np.ndarray, size: int, quality: int) -> list[np.ndarray]:
    """(T, H, W, 3) RGB uint8 -> list of jpeg byte arrays at size x size."""
    out = []
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    for img in frames:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img[..., ::-1], params)  # expects BGR
        if not ok:
            raise RuntimeError("jpeg encode failed")
        out.append(np.frombuffer(buf.tobytes(), dtype=np.uint8))
    return out


def process_episode(src_path: str, dst_dir: str, size: int, quality: int,
                    include_waist: bool, image_keys: tuple[str, ...],
                    motion_threshold: float, keyframe_threshold: float,
                    action_semantics: str = EXECUTED_ACTION_SEMANTICS) -> str:
    src_path, dst_dir = Path(src_path), Path(dst_dir)
    action_semantics = validate_action_semantics(action_semantics)
    dst_path = dst_dir / src_path.name
    if dst_path.exists():
        with h5py.File(dst_path, "r") as existing:
            stored = tuple(str(x) for x in existing.attrs.get("image_keys", ()))
            stored_action_semantics = str(existing.attrs.get("action_semantics", ""))
            stored_segmentation_version = int(existing.attrs.get("segmentation_version", -1))
            stored_motion_threshold = float(
                existing.attrs.get("segmentation_motion_threshold", np.nan)
            )
            stored_keyframe_threshold = float(
                existing.attrs.get("segmentation_keyframe_threshold", np.nan)
            )
        if stored != image_keys:
            raise ValueError(
                f"existing output {dst_path} uses image_keys={stored}, "
                f"requested {image_keys}; use a different --dst or remove it"
            )
        if stored_action_semantics != action_semantics:
            raise ValueError(
                f"existing output {dst_path} uses action_semantics={stored_action_semantics!r}; "
                "use a different --dst or remove it"
            )
        if stored_segmentation_version != SEGMENTATION_VERSION:
            raise ValueError(
                f"existing output {dst_path} uses segmentation_version="
                f"{stored_segmentation_version}; reprocess or run materialize_segmentation.py"
            )
        if not np.isclose(stored_motion_threshold, motion_threshold) or not np.isclose(
            stored_keyframe_threshold, keyframe_threshold
        ):
            raise ValueError(
                f"existing output {dst_path} uses different segmentation thresholds; "
                "use a new --dst or rematerialize segmentation"
            )
        return f"skip (exists): {dst_path.name}"

    with h5py.File(src_path, "r", libver="latest") as f:
        traj = f["trajectory"]
        arm_pos = traj["arm2_pos"][:].astype(np.float32)
        hand_pos = traj["hand2_pos"][:].astype(np.float32)
        hand_target = (
            traj["hand2_pos_target"][:].astype(np.float32)
            if action_semantics == HYBRID_ACTION_SEMANTICS
            else None
        )
        phase = [p.decode() if isinstance(p, bytes) else str(p)
                 for p in traj["phase"][:]]
        T = arm_pos.shape[0]
        expected_shapes = {
            "arm2_pos": (T, 7),
            "hand2_pos": (T, 6),
        }
        arrays = {
            "arm2_pos": arm_pos,
            "hand2_pos": hand_pos,
        }
        if hand_target is not None:
            expected_shapes["hand2_pos_target"] = (T, 6)
            arrays["hand2_pos_target"] = hand_target
        for key, expected in expected_shapes.items():
            if arrays[key].shape != expected:
                raise ValueError(f"{src_path.name}: {key} must have shape {expected}, got {arrays[key].shape}")
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{src_path.name}: {key} contains non-finite values")
        if hand_target is not None and np.any(np.all(hand_target == 0.0, axis=1)):
            raise ValueError(
                f"{src_path.name}: hand2_pos_target contains an all-zero row; "
                "refusing to treat a possible placeholder as a commanded target"
            )
        qpos_parts = [arm_pos, hand_pos]
        if include_waist:
            qpos_parts.append(traj["waist_pos"][:].astype(np.float32))
        qpos = np.concatenate(qpos_parts, axis=1)
        action_hand = hand_target if hand_target is not None else hand_pos
        action = np.concatenate([arm_pos, action_hand], axis=1)
        segment_type, arm_keyframe = compute_executed_action_segments(
            np.concatenate([arm_pos, hand_pos], axis=1),
            motion_threshold=motion_threshold,
            keyframe_threshold=keyframe_threshold,
        )

        # Sequential full-dataset reads are chunk-friendly; do them once here
        # so training never touches the pathological (17,60,80,1) layout.
        jpegs = {}
        for cam in image_keys:
            frames = traj["cameras"][cam][:]          # (T, 480, 640, 3)
            if frames.shape[0] != T:
                raise ValueError(f"{src_path.name}: {cam} length {frames.shape[0]} != trajectory length {T}")
            jpegs[cam] = encode_frames(frames, size, quality)

        meta = f.get("meta")
        success = bool(meta.attrs.get("success", True)) if meta is not None else True
        object_name = str(meta.attrs.get("object_name", "")) if meta is not None else ""

    tmp = dst_path.with_suffix(".tmp")
    with h5py.File(tmp, "w", libver="latest") as g:
        obs = g.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        vlen_u8 = h5py.vlen_dtype(np.uint8)
        for cam in image_keys:
            ds = obs.create_dataset(cam, shape=(T,), dtype=vlen_u8)
            for t, buf in enumerate(jpegs[cam]):
                ds[t] = buf
        g.create_dataset("action", data=action)
        g.create_dataset("phase", data=np.array(phase, dtype=object),
                         dtype=h5py.string_dtype())
        g.create_dataset("segment_type", data=segment_type)
        g.create_dataset("arm_keyframe", data=arm_keyframe)
        g.attrs["success"] = success
        g.attrs["object_name"] = object_name
        g.attrs["num_steps"] = T
        g.attrs["source_file"] = str(src_path)
        g.attrs["image_size"] = size
        g.attrs["image_keys"] = np.asarray(image_keys, dtype=h5py.string_dtype())
        g.attrs["qpos_layout"] = ("arm2_pos(7)+hand2_pos(6)"
                                  + ("+waist(2)" if include_waist else ""))
        g.attrs["action_layout"] = action_layout(action_semantics)
        g.attrs["action_semantics"] = action_semantics
        g.attrs["segmentation_version"] = SEGMENTATION_VERSION
        g.attrs["segmentation_source"] = "executed_joint_position"
        g.attrs["segmentation_motion_threshold"] = motion_threshold
        g.attrs["segmentation_keyframe_threshold"] = keyframe_threshold
    tmp.rename(dst_path)
    return f"ok: {dst_path.name} (T={T})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument(
        "--image-keys",
        nargs="+",
        default=list(DEFAULT_IMAGE_KEYS),
        choices=DEFAULT_IMAGE_KEYS,
        help="RGB views to export; unselected views are not required or read",
    )
    ap.add_argument("--include-waist", action="store_true")
    ap.add_argument(
        "--action-semantics",
        choices=sorted(ACTION_LAYOUTS),
        default=EXECUTED_ACTION_SEMANTICS,
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD)
    ap.add_argument("--keyframe-threshold", type=float, default=DEFAULT_KEYFRAME_THRESHOLD)
    args = ap.parse_args()
    image_keys = tuple(dict.fromkeys(args.image_keys))

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.hdf5")) + sorted(src.glob("*.h5"))
    print(f"{len(files)} episodes: {src} -> {dst}")

    if args.workers <= 1:
        for p in files:
            print(process_episode(str(p), str(dst), args.image_size,
                                  args.jpeg_quality, args.include_waist,
                                  image_keys, args.motion_threshold,
                                  args.keyframe_threshold, args.action_semantics))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_episode, str(p), str(dst),
                              args.image_size, args.jpeg_quality,
                              args.include_waist, image_keys,
                              args.motion_threshold, args.keyframe_threshold,
                              args.action_semantics): p for p in files}
            for fut in as_completed(futs):
                print(fut.result())

    total = sum(p.stat().st_size for p in dst.glob("*.hdf5")) / 1e9
    print(f"done. processed set size: {total:.2f} GB")


if __name__ == "__main__":
    main()
