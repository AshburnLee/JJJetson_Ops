"""Write weight fixture directory for weight_loader_load_fixture."""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Mapping

import numpy as np


# 测试 helper：把 [名字: ndarray] 字典写成单个 F32 .safetensors 文件（不依赖 HF / safetensors 库）。
#
# Big picture 里它在哪？
#   test_weight_loader 先调本函数写出 .safetensors，再调 weight_loader_me.load_safetensors 读回来对比。
#   C++ 侧走 weight_loader_load_safetensors / safetensors_read_file。
#   与 write_weight_fixture 对照：fixture 写一整个目录（config + manifest + 多个 .f32）；本函数只写**一个** safetensors 文件。
#   图纸：doc/guide/fixture_structure.md；phase2 safetensors 步骤 1。
#
# 函数内部顺序（逐步）：
#   例：tensors = {"embed": [2,2] 的 f32 数组, "layer0.w_q": [4] 的 f32 数组}
#   step 1. 按名字排序，逐个 tensor：先保证是内存连续的 float32，再转成原始字节；
#           同时在 header 字典里登记这张 tensor 叫啥、几维、各维多大、字节落在文件 data 段的哪一段；
#           这些字节按顺序追加进 data。
#   step 2. 把 header 字典序列化成 JSON 文本，再编码成 UTF-8 字节串 header_bytes。
#   step 3. 写文件：先写 8 字节小端整数（JSON 有多长），再写 JSON 本身，最后写 step 1 攒好的 data。
#
# 调用契约：只写 F32；JSON key 用内部名（embed、layer0.w_q），不做 HF 映射；不写 __metadata__
def write_safetensors_file(path: str, tensors: Mapping[str, np.ndarray]) -> None:
    # step 1：逐个 tensor 登记元数据，并把 float32 原始字节追加到 data
    header: dict[str, object] = {}
    offset = 0
    data = bytearray()
    for name in sorted(tensors.keys()):
        arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
        blob = arr.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": [int(d) for d in arr.shape],
            "data_offsets": [offset, offset + len(blob)],
        }
        data.extend(blob)
        offset += len(blob)

    # step 2：header 字典变成 JSON 的 UTF-8 字节
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # step 3：按 safetensors 布局写盘（头长 + JSON + data）
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


# 测试 helper：在 dir_path 下写出 weight_loader_load_fixture 能读的 fixture 目录。
#
# Big picture 里它在哪？
#   多数 Phase 2 单测（Engine、GenerateLoop、TransformerModel load 等）先调本函数造目录，
#   再 load_fixture 或 load_weights_from_fixture 把权重搬进 GPU。
#   与 write_safetensors_file 对照：本函数写**目录**（config + manifest + 多个 .f32）；safetensors 写单文件。
#   图纸：doc/guide/fixture_structure.md。
#
# 函数内部顺序（逐步）：
#   例：dir_path=/tmp/jj_fixture，config 含 hidden_size=128 等 11 项，
#       tensors 含 embed shape [512,128]、layer0.w_q shape [128,128] 的 f32 数组
#   step 1. 创建 dir_path（已存在则沿用）
#   step 2. 写 config.txt：config 里每个 key=value 占一行
#   step 3. 按 tensor 名字排序：每个写成 dir_path 下的 .f32（名字里 . 换成 _），
#           并攒一行 manifest（名字、ndim、各维、文件名）
#   step 4. 写 manifest.txt，每行对应一张 .f32
#
# 调用契约：tensors 可为空（只写 config，供 safetensors 可选 config 单测）；权重须 f32 row-major
def write_weight_fixture(
    dir_path: str,
    config: Mapping[str, float | int],
    tensors: Mapping[str, np.ndarray],
) -> None:
    # step 1：确保目录存在
    os.makedirs(dir_path, exist_ok=True)

    # step 2：ModelConfig 写进 config.txt
    with open(os.path.join(dir_path, "config.txt"), "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

    # step 3：逐 tensor 写 .f32，并收集 manifest 行
    manifest_lines: list[str] = []
    for name in sorted(tensors.keys()):
        arr = np.asarray(tensors[name], dtype=np.float32)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        fname = name.replace(".", "_") + ".f32"
        arr.tofile(os.path.join(dir_path, fname))
        dims = " ".join(str(int(d)) for d in arr.shape)
        manifest_lines.append(f"{name} {arr.ndim} {dims} {fname}")

    # step 4：manifest 清单写盘
    with open(os.path.join(dir_path, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))
        f.write("\n")
