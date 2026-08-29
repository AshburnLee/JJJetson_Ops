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


# 测试 helper：从已写好的 fixture 目录导出 .safetensors（safetensors 步骤 2 roundtrip）。
#
# Big picture 里它在哪？
#   write_weight_fixture 写出目录后，本函数用 load_fixture 读回 host tensor，再 write_safetensors_file 写单文件。
#   单测再 load_safetensors 读回，与 load_fixture 结果逐 tensor 对比。
#   图纸：doc/guide/fixture_structure.md；phase2 safetensors 步骤 2。
#
# 函数内部顺序（逐步）：
#   例：fixture_dir 含 config.txt + manifest.txt + layer0_w_q.f32 等
#   step 1. weight_loader_me.load_fixture(fixture_dir) 按 manifest 读 .f32
#   step 2. 把 loaded["tensors"] 转成 numpy dict
#   step 3. write_safetensors_file(out_path, tensors)
#
# 调用契约：fixture_dir 须能被 load_fixture 成功读取；out_path 可为 fixture_dir 内或外路径
def export_fixture_dir_to_safetensors(fixture_dir: str, out_path: str) -> None:
    import weight_loader_me

    # step 1：从 fixture 目录读出 host 权重
    loaded = weight_loader_me.load_fixture(fixture_dir)
    tensors = {name: np.asarray(arr, dtype=np.float32) for name, arr in loaded["tensors"].items()}

    # step 2：写成 safetensors 单文件
    write_safetensors_file(out_path, tensors)


# HF Llama checkpoint key（PyTorch Linear [out,in]）-> 写 safetensors 单测用（步骤 3）。
# Linear 内部已经是 PyTorch [out, in]，和 HF 一样。只有 lm_head 内部是 [hidden, vocab]。


def internal_name_to_hf_llama_key(name: str) -> str:
    if name == "embed":
        return "model.embed_tokens.weight"
    if name == "final_norm":
        return "model.norm.weight"
    if name == "lm_head":
        return "lm_head.weight"
    if not name.startswith("layer") or "." not in name:
        raise KeyError(f"unsupported internal tensor name: {name}")
    layer_idx, suffix = name.split(".", 1)
    layer_num = layer_idx.removeprefix("layer")
    layer_keys = {
        "w_q": f"model.layers.{layer_num}.self_attn.q_proj.weight",
        "w_k": f"model.layers.{layer_num}.self_attn.k_proj.weight",
        "w_v": f"model.layers.{layer_num}.self_attn.v_proj.weight",
        "w_o": f"model.layers.{layer_num}.self_attn.o_proj.weight",
        "w_gate": f"model.layers.{layer_num}.mlp.gate_proj.weight",
        "w_up": f"model.layers.{layer_num}.mlp.up_proj.weight",
        "w_down": f"model.layers.{layer_num}.mlp.down_proj.weight",
        "w_input_layernorm": f"model.layers.{layer_num}.input_layernorm.weight",
        "w_post_attention_layernorm": (f"model.layers.{layer_num}.post_attention_layernorm.weight"),
    }
    if suffix not in layer_keys:
        raise KeyError(f"unsupported layer suffix: {suffix}")
    return layer_keys[suffix]


def internal_tensors_to_hf_llama_layout(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """内部 fixture layout -> HF safetensors JSON key + PyTorch Linear 矩阵方向。"""
    hf: dict[str, np.ndarray] = {}
    for name, arr in tensors.items():
        key = internal_name_to_hf_llama_key(name)
        data = np.asarray(arr, dtype=np.float32)
        if name == "lm_head":
            hf[key] = np.ascontiguousarray(data.T)
        else:
            hf[key] = np.ascontiguousarray(data)
    return hf


# 测试 helper：把内部 ModelConfig dict 写成同目录 config.txt（11 项 key=value）。
def write_model_config_txt(dir_path: str, config: Mapping[str, float | int]) -> None:
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "config.txt"), "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")


# 测试 helper：内部 ModelConfig dict -> HF Llama config.json 字段名。
#
# 例：config num_layers=1, num_q_heads=4, head_dim=32, hidden_size=128
#     写出 num_hidden_layers=1, num_attention_heads=4, head_dim=32, hidden_size=128
def write_hf_llama_config_json(path: str, config: Mapping[str, float | int]) -> None:
    hf = {
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "num_hidden_layers": int(config["num_layers"]),
        "num_attention_heads": int(config["num_q_heads"]),
        "num_key_value_heads": int(config["num_kv_heads"]),
        "head_dim": int(config["head_dim"]),
        "vocab_size": int(config["vocab_size"]),
        "max_position_embeddings": int(config["max_seq_len"]),
        "rope_theta": float(config["freq_base"]),
        "rms_norm_eps": float(config["rms_norm_epsilon"]),
        "tie_word_embeddings": bool(int(config["tie_word_embeddings"])),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hf, f, indent=2)
        f.write("\n")
