"""Create a compact inspection copy of an HDF5 file.

Usage: python3 slim_h5.py input.h5 slim_output.h5
"""

import sys

import h5py
import numpy as np


src_path, dst_path = sys.argv[1], sys.argv[2]
MAX_BYTES = 5 * 1024 * 1024


def copy_slim(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value

    for name, item in src.items():
        if isinstance(item, h5py.Group):
            copy_slim(item, dst.create_group(name))
            continue

        nbytes = item.dtype.itemsize * int(np.prod(item.shape))
        if nbytes <= MAX_BYTES:
            data = item[...]
            kwargs = {} if item.shape == () else {
                "compression": "gzip",
                "compression_opts": 4,
            }
            copied = dst.create_dataset(name, data=data, **kwargs)
        else:
            data = item[:2] if item.ndim > 0 and item.shape[0] >= 2 else item[...]
            kwargs = {} if np.asarray(data).shape == () else {
                "compression": "gzip",
                "compression_opts": 4,
            }
            copied = dst.create_dataset(name + "__sample2", data=data, **kwargs)
            copied.attrs["original_shape"] = item.shape
            copied.attrs["original_dtype"] = str(item.dtype)

        for key, value in item.attrs.items():
            copied.attrs[key] = value


with h5py.File(src_path, "r") as source, h5py.File(dst_path, "w") as target:
    copy_slim(source, target)

print("done:", dst_path)
