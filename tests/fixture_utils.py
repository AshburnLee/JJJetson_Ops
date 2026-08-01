"""Write weight fixture directory for weight_loader_load_fixture."""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np


def write_weight_fixture(
    dir_path: str,
    config: Mapping[str, float | int],
    tensors: Mapping[str, np.ndarray],
) -> None:
    """config.txt + manifest.txt + row-major float32 *.f32 files."""
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, "config.txt"), "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

    manifest_lines: list[str] = []
    for name in sorted(tensors.keys()):
        arr = np.asarray(tensors[name], dtype=np.float32)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        fname = name.replace(".", "_") + ".f32"
        arr.tofile(os.path.join(dir_path, fname))
        dims = " ".join(str(int(d)) for d in arr.shape)
        manifest_lines.append(f"{name} {arr.ndim} {dims} {fname}")

    with open(os.path.join(dir_path, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))
        f.write("\n")
