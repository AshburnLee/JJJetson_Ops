"""TinyLlama HF tokenizer 薄封装（CPU）。Engine 外：文本 <-> token_ids。

例：
  encode("Hello") -> [15043]
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
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _has_tokenizer_files(directory: str) -> bool:
    return all(os.path.isfile(os.path.join(directory, name)) for name in _TOKENIZER_MARKERS)


def resolve_tokenizer_dir(*, slice_dir: str | None = None) -> str:
    """Tokenizer 目录：JJ_HF_TOKENIZER_DIR > slice_dir 内文件 > 默认 hf_src。"""
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
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers required for HF tokenizer") from exc

    directory = tokenizer_dir or resolve_tokenizer_dir(slice_dir=slice_dir)
    return AutoTokenizer.from_pretrained(directory)


def encode(
    text: str,
    tokenizer: Any | None = None,
    *,
    tokenizer_dir: str | None = None,
    slice_dir: str | None = None,
    add_special_tokens: bool = True,
) -> list[int]:
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
    tok = tokenizer or load_tokenizer(tokenizer_dir, slice_dir=slice_dir)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos_token_id")
    return int(eos_id)


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
    """文本进、文本出：encode -> GenerateLoop -> detokenize 新生成 token。"""
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
