"""HF Tokenizer 正式入口（CPU，引擎外）。文本 <-> token_ids；不进 C++ Engine。

对齐 TinyLlama / Llama 系 HF 词表。Engine / GenerateLoop 边界仍是 token_ids。

调用约定：
  1. 先 create Model + Engine（与现有一致）
  2. generate_text(engine, prompt, max_new) 文本进、文本出
  或手动 encode -> generate_loop_me.generate -> decode

例（TinyLlama）：
  encode("Hello") -> [1, 15043]   # 默认带 BOS=1
  decode([15043]) -> "Hello"
"""

from __future__ import annotations

import os
from typing import Any

import generate_loop_me
import numpy as np

_DEFAULT_TOKENIZER_REL = os.path.join("models", "hf_src", "TinyLlama__TinyLlama-1.1B-Chat-v1.0")
_TOKENIZER_MARKERS = ("tokenizer.json", "tokenizer.model")


def _ops_root() -> str:
    # 本文件在 JJJetson_Ops/py/；上一级即工程根（python/ 仅放 .so，已 gitignore）
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _has_tokenizer_files(directory: str) -> bool:
    return all(os.path.isfile(os.path.join(directory, name)) for name in _TOKENIZER_MARKERS)


def resolve_tokenizer_dir(*, slice_dir: str | None = None) -> str:
    """解析 tokenizer 目录。

    优先级：
      1. 环境变量 JJ_HF_TOKENIZER_DIR
      2. slice_dir 内已有 tokenizer.json / tokenizer.model
      3. 默认 models/hf_src/TinyLlama__TinyLlama-1.1B-Chat-v1.0
    """
    env_dir = os.environ.get("JJ_HF_TOKENIZER_DIR", "").strip()
    if env_dir:
        if not _has_tokenizer_files(env_dir):
            raise FileNotFoundError(f"missing tokenizer files under JJ_HF_TOKENIZER_DIR={env_dir}")
        return env_dir

    if slice_dir and _has_tokenizer_files(slice_dir):
        return slice_dir

    default_dir = os.path.join(_ops_root(), _DEFAULT_TOKENIZER_REL)
    if _has_tokenizer_files(default_dir):
        return default_dir

    raise FileNotFoundError(
        "tokenizer not found: set JJ_HF_TOKENIZER_DIR or place tokenizer.json under slice / hf_src"
    )


def load_tokenizer(tokenizer_dir: str | None = None, *, slice_dir: str | None = None) -> Any:
    """加载 HF AutoTokenizer。依赖 transformers（仅 CPU）。"""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers required for HF tokenizer") from exc

    directory = tokenizer_dir or resolve_tokenizer_dir(slice_dir=slice_dir)
    # AutoTokenizer.from_pretrained(目录) 并不“猜”分词算法，而是读该目录里已写好的配置，
    # 再实例化对应类。大致步骤：
    #   1. 读 tokenizer_config.json -> 知道 tokenizer_class / 特殊 token 名等
    #   2. 读 tokenizer.json（或 tokenizer.model）-> 词表、BPE merges、归一化规则
    #   3. 读 special_tokens_map.json -> bos/eos/pad/unk 映射
    #   4. 按 config 选出具体实现（TinyLlama 是 LlamaTokenizer / LlamaTokenizerFast），装好词表
    # 之后 encode/decode 都走这份对象，不再联网。
    #
    # 例（TinyLlama-1.1B-Chat 本地目录）：
    #   input:  directory = ".../TinyLlama__TinyLlama-1.1B-Chat-v1.0"
    #           目录内有 tokenizer.json + tokenizer.model + tokenizer_config.json ...
    #   output: tokenizer 对象；tokenizer.encode("Hello") -> [1, 15043]
    #           tokenizer.bos_token_id == 1，tokenizer.eos_token_id == 2
    return AutoTokenizer.from_pretrained(directory)


def encode(
    text: str,
    tokenizer: Any | None = None,
    *,
    tokenizer_dir: str | None = None,
    slice_dir: str | None = None,
    add_special_tokens: bool = True,
) -> list[int]:
    """文本 -> token_ids。默认 add_special_tokens=True（TinyLlama 会加 BOS）。"""
    tok = tokenizer or load_tokenizer(tokenizer_dir, slice_dir=slice_dir)
    return tok.encode(text, add_special_tokens=add_special_tokens)


def decode(
    token_ids: list[int] | np.ndarray,
    tokenizer: Any | None = None,
    *,
    tokenizer_dir: str | None = None,
    slice_dir: str | None = None,
    skip_special_tokens: bool = True,
) -> str:
    """token_ids -> 文本。默认跳过特殊 token。"""
    tok = tokenizer or load_tokenizer(tokenizer_dir, slice_dir=slice_dir)
    if isinstance(token_ids, np.ndarray):
        token_ids = token_ids.tolist()
    return tok.decode(token_ids, skip_special_tokens=skip_special_tokens)


def eos_token_id(
    tokenizer: Any | None = None,
    *,
    tokenizer_dir: str | None = None,
    slice_dir: str | None = None,
) -> int:
    """从 tokenizer 读 EOS id（TinyLlama-Chat 一般为 2）。不手写硬编码。"""
    tok = tokenizer or load_tokenizer(tokenizer_dir, slice_dir=slice_dir)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos_token_id")
    return int(eos_id)


def bos_token_id(
    tokenizer: Any | None = None,
    *,
    tokenizer_dir: str | None = None,
    slice_dir: str | None = None,
) -> int | None:
    """从 tokenizer 读 BOS id（TinyLlama 一般为 1；无则 None）。"""
    tok = tokenizer or load_tokenizer(tokenizer_dir, slice_dir=slice_dir)
    bos_id = tok.bos_token_id
    return None if bos_id is None else int(bos_id)


def generate_text(
    engine: int,
    prompt: str,
    max_new_tokens: int,
    *,
    tokenizer: Any | None = None,
    tokenizer_dir: str | None = None,
    slice_dir: str | None = None,
    top_k: int = 1,
    seed: int = 0,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """正式文本入口：encode -> generate_loop_me.generate -> decode 新生成 token。

    Engine 仍只吃 id；本函数不改 Engine API。返回的是**新生成**文本，不含 prompt。
    EOS 使用 tokenizer.eos_token_id。
    """
    tok = tokenizer or load_tokenizer(tokenizer_dir, slice_dir=slice_dir)
    prompt_ids = encode(prompt, tok)
    if not prompt_ids:
        raise ValueError("encode produced empty token list")

    prompt_np = np.array(prompt_ids, dtype=np.int32)
    new_ids = generate_loop_me.generate(
        engine,
        prompt_np,
        max_new_tokens,
        eos_token_id=eos_token_id(tok),
        top_k=top_k,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
    )
    return decode(new_ids, tok)
