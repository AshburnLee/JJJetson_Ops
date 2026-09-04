"""文本 e2e：encode (CPU) -> GenerateLoop (GPU) -> detokenize (CPU)。

正式入口：py/hf_tokenizer.py（契约 doc/guide/tokenizer_api.md）。

依赖：
  - JJ_HF_LLAMA_SLICE_DIR 指向 2 层 TinyLlama 切片（或本机默认 models/tinyllama_2layer）
  - tokenizer 文件（JJ_HF_TOKENIZER_DIR 或 models/hf_src/TinyLlama__TinyLlama-1.1B-Chat-v1.0）
  - transformers（仅 CPU 分词，不进 Engine）

缺切片 / tokenizer / transformers 时跳过，不 fail 全量 run_tests。
"""

from __future__ import annotations

import os

import generate_loop_me
import hf_tokenizer
import inference_engine_me
import numpy as np
import transformer_model_me

from test_hf_llama_real_slice_smoke import _parse_config_txt

decode = hf_tokenizer.decode
encode = hf_tokenizer.encode
eos_token_id = hf_tokenizer.eos_token_id
generate_text = hf_tokenizer.generate_text
load_tokenizer = hf_tokenizer.load_tokenizer
resolve_tokenizer_dir = hf_tokenizer.resolve_tokenizer_dir

_DEFAULT_SLICE_REL = os.path.join("models", "tinyllama_2layer")


def _ops_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _slice_dir() -> str:
    env_dir = os.environ.get("JJ_HF_LLAMA_SLICE_DIR", "").strip()
    if env_dir:
        return env_dir
    default_dir = os.path.join(_ops_root(), _DEFAULT_SLICE_REL)
    if os.path.isdir(default_dir):
        return default_dir
    return ""


def _skip_reason() -> str | None:
    slice_dir = _slice_dir()
    if not slice_dir:
        return "no slice dir (set JJ_HF_LLAMA_SLICE_DIR)"
    st_path = os.path.join(slice_dir, "model.safetensors")
    cfg_path = os.path.join(slice_dir, "config.txt")
    if not os.path.isfile(st_path):
        return f"missing {st_path}"
    if not os.path.isfile(cfg_path):
        return f"missing {cfg_path}"
    try:
        resolve_tokenizer_dir(slice_dir=slice_dir)
    except FileNotFoundError as exc:
        return str(exc)
    try:
        import transformers  # noqa: F401
    except ImportError:
        return "transformers not installed"
    return None


def test_text_generate_e2e_tinyllama_slice() -> None:
    reason = _skip_reason()
    if reason:
        print(f"Passed test_text_generate_e2e_tinyllama_slice skipped ({reason})")
        return

    slice_dir = _slice_dir()
    tokenizer_dir = resolve_tokenizer_dir(slice_dir=slice_dir)
    cfg = _parse_config_txt(os.path.join(slice_dir, "config.txt"))
    st_path = os.path.join(slice_dir, "model.safetensors")

    prompt = "Hello"
    max_new = 4

    tokenizer = load_tokenizer(tokenizer_dir)
    prompt_ids = encode(prompt, tokenizer)
    assert prompt_ids, "encode returned empty ids"
    assert all(0 <= tid < int(cfg["vocab_size"]) for tid in prompt_ids)

    # roundtrip：分词再反分词应回到原 prompt（不加特殊 token）
    roundtrip = decode(prompt_ids, tokenizer, skip_special_tokens=True)
    assert roundtrip == prompt, f"roundtrip {roundtrip!r} != prompt {prompt!r}"

    model = transformer_model_me.create_model(**cfg)
    engine = None
    try:
        transformer_model_me.load_weights_from_safetensors_hf_llama(model, st_path)
        engine = inference_engine_me.create_engine(model)

        prompt_np = np.array(prompt_ids, dtype=np.int32)
        new_ids = generate_loop_me.generate(
            engine,
            prompt_np,
            max_new,
            eos_token_id=eos_token_id(tokenizer),
            top_k=1,
            seed=0,
        )
        assert len(new_ids) == max_new
        assert all(isinstance(tid, int) for tid in new_ids)
        assert all(0 <= tid < int(cfg["vocab_size"]) for tid in new_ids)

        completion = decode(new_ids, tokenizer)
        assert isinstance(completion, str)
        assert len(completion) > 0

        assert inference_engine_me.kv_cache_len(engine) == len(prompt_ids) + max_new - 1

        inference_engine_me.reset_engine(engine)
        completion_via_helper = generate_text(
            engine,
            prompt,
            max_new,
            tokenizer=tokenizer,
            top_k=1,
            seed=0,
        )
        assert completion_via_helper == completion

        print(
            "Passed test_text_generate_e2e_tinyllama_slice",
            f"prompt={prompt!r}",
            f"ids={prompt_ids}",
            f"new_ids={new_ids}",
            f"completion={completion!r}",
            f"tokenizer_dir={tokenizer_dir}",
        )
    finally:
        if engine is not None:
            inference_engine_me.destroy_engine(engine)
        transformer_model_me.destroy_model(model)


if __name__ == "__main__":
    test_text_generate_e2e_tinyllama_slice()
