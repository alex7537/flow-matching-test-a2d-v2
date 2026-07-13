from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flow_matching_test.segmentation import (
    DEFAULT_KEYFRAME_THRESHOLD,
    DEFAULT_MOTION_THRESHOLD,
    SEGMENTATION_VERSION,
    compute_executed_action_segments,
)


def materialize(path: Path, *, motion_threshold: float, keyframe_threshold: float) -> None:
    with h5py.File(path, "r+") as file:
        action = np.asarray(file["action"][:], dtype=np.float32)
        expected_segment, expected_arm = compute_executed_action_segments(
            action,
            motion_threshold=motion_threshold,
            keyframe_threshold=keyframe_threshold,
        )
        for name, expected in (
            ("segment_type", expected_segment),
            ("arm_keyframe", expected_arm),
        ):
            if name in file:
                actual = np.asarray(file[name][:], dtype=np.uint8)
                if not np.array_equal(actual, expected):
                    raise ValueError(f"{path}: existing {name} differs from recomputed labels")
            else:
                file.create_dataset(name, data=expected)
        file.attrs["segmentation_version"] = SEGMENTATION_VERSION
        file.attrs["segmentation_source"] = "executed_joint_position"
        file.attrs["segmentation_motion_threshold"] = motion_threshold
        file.attrs["segmentation_keyframe_threshold"] = keyframe_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize deterministic segmentation in processed HDF5s")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD)
    parser.add_argument("--keyframe-threshold", type=float, default=DEFAULT_KEYFRAME_THRESHOLD)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.hdf5")) + sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no processed HDF5 files under {data_dir}")
    for path in files:
        materialize(
            path,
            motion_threshold=args.motion_threshold,
            keyframe_threshold=args.keyframe_threshold,
        )
        print(f"ok: {path.name}")


if __name__ == "__main__":
    main()
