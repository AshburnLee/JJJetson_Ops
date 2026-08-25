"""从本地 HF Llama checkpoint 导出 1~2 layer F32 safetensors 切片（safetensors 步骤 4）。

本机 Orin 内存/显存大约 4GB，不能把 TinyLlama 全量读进 RAM。本脚本只做切片，不是单测：
  1. --src 必须是已经下好的本地 HF 目录（含 config.json + .safetensors）
  2. 按 tensor 流式读出前 N 层，转 F32 后立刻写入切片文件（峰值大约一张 embed，约 256MB）
  3. GPU 只加载切片；下载请自己完成，脚本不联网

在 JJJetson_Ops 目录执行：

    python scripts/export_hf_llama_slice.py \\
        --src models/hf_src/TinyLlama__TinyLlama-1.1B-Chat-v1.0 \\
        --out-dir models/tinyllama_2layer \\
        --num-layers 2 \\
        --max-seq-len 256

--out-dir 写出三份文件（不覆盖 --src）：
    model.safetensors   # 只含前 N 层 + embed / norm / lm_head（若 untied），F32
    config.txt          # 引擎 11 项 ModelConfig；num_layers = N
    config.json         # HF 字段名；num_hidden_layers = N
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections.abc import Iterator

import numpy as np

# 本文件在 scripts/，config 写出函数在 tests/fixture_utils.py。
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR = os.path.join(_ROOT_DIR, "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from fixture_utils import write_hf_llama_config_json, write_model_config_txt  # noqa: E402

_KEEP_EXACT = frozenset(
    {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
)


def _keep_hf_key(name: str, num_layers: int, keep_lm_head: bool) -> bool:
    if name == "lm_head.weight":
        return keep_lm_head
    if name in _KEEP_EXACT:
        return True
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        if name.startswith(prefix) and name.endswith(".weight"):
            return True
    return False


def _require_local_src_dir(src: str) -> str:
    # 例：src 是 models/hf_src/TinyLlama...，里面已有 config.json 和 model.safetensors。
    if not os.path.isdir(src):
        raise SystemExit(f"--src must be an existing local HF directory, got {src!r}")
    return os.path.abspath(src)


def _safetensors_files(src_dir: str) -> list[str]:
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        names = sorted(set(index["weight_map"].values()))
        return [os.path.join(src_dir, name) for name in names]
    single = os.path.join(src_dir, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    found = sorted(
        os.path.join(src_dir, name) for name in os.listdir(src_dir) if name.endswith(".safetensors")
    )
    if not found:
        raise SystemExit(f"no .safetensors under {src_dir}")
    return found


def _read_safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def _dtype_to_f32(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    # 例：BF16 的 1.0 是 uint16 0x3F80，左移 16 位变成 F32 的 0x3F800000。
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32, copy=False).reshape(shape)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        bits = (u16.astype(np.uint32) << 16).view(np.float32)
        return bits.reshape(shape).copy()
    raise SystemExit(f"unsupported safetensors dtype {dtype}")


def _iter_kept_tensors(
    src_dir: str, num_layers: int, keep_lm_head: bool
) -> Iterator[tuple[str, str, int, dict]]:
    # 产出 (hf_name, file_path, data_start, info)
    for path in _safetensors_files(src_dir):
        header, data_start = _read_safetensors_header(path)
        for name, info in header.items():
            if _keep_hf_key(name, num_layers, keep_lm_head):
                yield name, path, data_start, info


def _product(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _export_f32_safetensors_streaming(
    src_dir: str, out_path: str, num_layers: int, keep_lm_head: bool
) -> list[str]:
    # 先扫 header 登记要保留的 tensor，再按名字排序写 F32。内存里同时只放一张。
    entries = list(_iter_kept_tensors(src_dir, num_layers, keep_lm_head))
    by_name: dict[str, tuple[str, int, dict]] = {}
    for name, path, data_start, info in entries:
        if name in by_name:
            raise SystemExit(f"duplicate tensor {name}")
        by_name[name] = (path, data_start, info)

    if "model.embed_tokens.weight" not in by_name:
        raise SystemExit("missing model.embed_tokens.weight")
    if "model.norm.weight" not in by_name:
        raise SystemExit("missing model.norm.weight")
    for i in range(num_layers):
        q_key = f"model.layers.{i}.self_attn.q_proj.weight"
        if q_key not in by_name:
            raise SystemExit(f"missing {q_key}")

    names = sorted(by_name.keys())
    header: dict[str, object] = {}
    offset = 0
    for name in names:
        _path, _data_start, info = by_name[name]
        shape = [int(d) for d in info["shape"]]
        nbytes = _product(shape) * 4
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as out_f:
        out_f.write(struct.pack("<Q", len(header_bytes)))
        out_f.write(header_bytes)
        for name in names:
            path, data_start, info = by_name[name]
            start, end = info["data_offsets"]
            print(f"  slice {name} shape={info['shape']} dtype={info['dtype']}")
            with open(path, "rb") as in_f:
                in_f.seek(data_start + int(start))
                raw = in_f.read(int(end) - int(start))
            arr = _dtype_to_f32(raw, str(info["dtype"]), [int(d) for d in info["shape"]])
            del raw
            blob = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
            del arr
            if len(blob) != header[name]["data_offsets"][1] - header[name]["data_offsets"][0]:
                raise SystemExit(f"byte size mismatch for {name}")
            out_f.write(blob)
            del blob
    return names


def _internal_config_from_hf_json(
    hf: dict, num_layers: int, max_seq_len: int | None
) -> dict[str, float | int]:
    hidden = int(hf["hidden_size"])
    q_heads = int(hf["num_attention_heads"])
    head_dim = int(hf["head_dim"]) if "head_dim" in hf else hidden // q_heads
    kv_heads = int(hf.get("num_key_value_heads", q_heads))
    tied = bool(hf.get("tie_word_embeddings", False))
    seq = int(max_seq_len) if max_seq_len is not None else int(hf["max_position_embeddings"])
    return {
        "hidden_size": hidden,
        "intermediate_size": int(hf["intermediate_size"]),
        "num_layers": int(num_layers),
        "num_q_heads": q_heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": int(hf["vocab_size"]),
        "max_seq_len": seq,
        "freq_base": float(hf.get("rope_theta", 10000.0)),
        "rms_norm_epsilon": float(hf["rms_norm_eps"]),
        "tie_word_embeddings": 1 if tied else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 1-2 layer F32 HF Llama safetensors slice")
    parser.add_argument(
        "--src", required=True, help="local HF directory (config.json + safetensors)"
    )
    parser.add_argument("--out-dir", required=True, help="output directory (not for git)")
    parser.add_argument("--num-layers", type=int, default=1, help="keep layers [0, N)")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="override max_seq_len in exported config (Jetson KV); default: HF max_position_embeddings",
    )
    args = parser.parse_args()
    if args.num_layers < 1:
        raise SystemExit("--num-layers must be >= 1")

    src_dir = _require_local_src_dir(args.src)
    cfg_path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"missing {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        hf_cfg = json.load(f)

    src_layers = int(hf_cfg["num_hidden_layers"])
    if args.num_layers > src_layers:
        raise SystemExit(f"--num-layers {args.num_layers} > source num_hidden_layers {src_layers}")

    keep_lm_head = not bool(hf_cfg.get("tie_word_embeddings", False))
    if not keep_lm_head:
        print("tied embeddings: skip lm_head.weight")

    os.makedirs(args.out_dir, exist_ok=True)
    st_path = os.path.join(args.out_dir, "model.safetensors")
    names = _export_f32_safetensors_streaming(src_dir, st_path, args.num_layers, keep_lm_head)
    internal_cfg = _internal_config_from_hf_json(hf_cfg, args.num_layers, args.max_seq_len)
    write_model_config_txt(args.out_dir, internal_cfg)
    write_hf_llama_config_json(os.path.join(args.out_dir, "config.json"), internal_cfg)

    nbytes = os.path.getsize(st_path)
    print(
        f"kept {len(names)} tensors, wrote {st_path} ({nbytes} bytes), num_layers={args.num_layers}"
    )
    print(f"set JJ_HF_LLAMA_SLICE_DIR={os.path.abspath(args.out_dir)} to run real-slice smoke test")


if __name__ == "__main__":
    main()
