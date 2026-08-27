"""可选数值单测：真实 HF Llama 切片 prefill logits vs 4.1 dump 的 npy。

未设置 JJ_HF_LLAMA_SLICE_DIR、或切片目录里没有 ref npy 时跳过。
不要 import transformers（dump 已在 scripts/export_hf_llama_slice_logits.py 做完）。
不要在 Python 里再 load 一份完整权重 numpy：只读 config.txt，H2D 由 C++ Loader 完成。
不要和 fixture tensor ref 再比一遍；唯一基准是那张 logits npy。

例：token_ids=[1,2,3,4]，npy 与 Engine 都是 [32000, 4]、列主序。
    logits[v, t] 是第 t 个输入位置、词表第 v 维。只比这两张表。
T=1 探针：没有未来 token，causal 与双向应几乎一样；只打印 max_abs，不断言对齐。
"""

import os

import inference_engine_me
import numpy as np
import transformer_model_me

from test_hf_llama_real_slice_smoke import _parse_config_txt, _slice_dir

_REF_NPY = "ref_prefill_logits_t1234.npy"
_REF_NPY_T1 = "ref_prefill_logits_t1.npy"
_TOKEN_IDS = np.array([1, 2, 3, 4], dtype=np.int32)
_TOKEN_IDS_T1 = np.array([1], dtype=np.int32)

# FA 的 Q/K/V staging 是 IEEE fp16，HF dump 是全 F32 attention。
_ATOL = 1e-4
_RTOL = 1e-4


def _compare_engine_to_npy(token_ids: np.ndarray, ref_name: str, *, assert_close: bool) -> None:
    slice_dir = _slice_dir()
    t_len = int(token_ids.shape[0])
    case = f"t{''.join(str(int(x)) for x in token_ids)}"
    if not slice_dir:
        print(f"Passed test_hf_llama_real_slice_logits_{case} skipped")
        return

    st_path = os.path.join(slice_dir, "model.safetensors")
    cfg_path = os.path.join(slice_dir, "config.txt")
    ref_path = os.path.join(slice_dir, ref_name)
    if not os.path.isfile(ref_path):
        print(f"Passed test_hf_llama_real_slice_logits_{case} skipped (no ref npy)")
        return
    if not os.path.isfile(st_path):
        raise AssertionError(f"missing {st_path}")
    if not os.path.isfile(cfg_path):
        raise AssertionError(f"missing {cfg_path}")

    cfg = _parse_config_txt(cfg_path)
    vocab = int(cfg["vocab_size"])
    ref = np.asfortranarray(np.load(ref_path).astype(np.float32, copy=False))
    if ref.shape != (vocab, t_len):
        raise AssertionError(f"ref shape {ref.shape} != ({vocab}, {t_len})")

    model = transformer_model_me.create_model(**cfg)
    engine = None
    try:
        transformer_model_me.load_weights_from_safetensors_hf_llama(model, st_path)
        if not transformer_model_me.is_weights_loaded(model):
            raise AssertionError("weights not loaded")
        engine = inference_engine_me.create_engine(model)

        logits = np.zeros((vocab, t_len), dtype=np.float32, order="F")
        inference_engine_me.forward_token_host(engine, t_len, 0, token_ids, logits)
        assert inference_engine_me.kv_cache_len(engine) == t_len
        if not np.isfinite(logits).all():
            raise AssertionError("engine prefill logits not finite")

        engine_argmax = int(np.argmax(logits[:, -1]))
        ref_argmax = int(np.argmax(ref[:, -1]))
        col_max_abs = np.max(np.abs(logits - ref), axis=0)
        max_abs = float(np.max(col_max_abs))
        print(
            f"  {case} engine_argmax={engine_argmax} ref_argmax={ref_argmax} "
            f"max_abs={max_abs:.6g} "
            f"per_token_max_abs={np.array2string(col_max_abs, precision=4)}"
        )
        if not assert_close:
            print(f"Passed test_hf_llama_real_slice_logits_{case} probe")
            return
        if engine_argmax != ref_argmax:
            raise AssertionError(
                f"last-token argmax mismatch: engine={engine_argmax} hf={ref_argmax} "
                f"max_abs={max_abs:.6g}"
            )
        if not np.allclose(logits, ref, atol=_ATOL, rtol=_RTOL):
            raise AssertionError(
                f"logits not close to HF npy: max_abs={max_abs:.6g} atol={_ATOL} rtol={_RTOL}"
            )
        print(f"Passed test_hf_llama_real_slice_logits_{case}")
    finally:
        if engine is not None:
            inference_engine_me.destroy_engine(engine)
        transformer_model_me.destroy_model(model)


def test_hf_llama_real_slice_logits_t1() -> None:
    # T=1 探针：不断言对齐，只打印 max_abs / argmax。
    _compare_engine_to_npy(_TOKEN_IDS_T1, _REF_NPY_T1, assert_close=False)


def test_hf_llama_real_slice_logits() -> None:
    _compare_engine_to_npy(_TOKEN_IDS, _REF_NPY, assert_close=True)


def test_hf_llama_real_slice_embed_t1() -> None:
    # embed 探针：只比查表，不进 layer。T=1，npy 是 dump 脚本顺带写出的 ref_embed_t1.npy。
    slice_dir = _slice_dir()
    if not slice_dir:
        print("Passed test_hf_llama_real_slice_embed_t1 skipped")
        return
    st_path = os.path.join(slice_dir, "model.safetensors")
    cfg_path = os.path.join(slice_dir, "config.txt")
    ref_path = os.path.join(slice_dir, "ref_embed_t1.npy")
    if not os.path.isfile(ref_path):
        print("Passed test_hf_llama_real_slice_embed_t1 skipped (no ref npy)")
        return
    if not os.path.isfile(st_path) or not os.path.isfile(cfg_path):
        raise AssertionError("missing slice safetensors or config.txt")

    cfg = _parse_config_txt(cfg_path)
    hidden = int(cfg["hidden_size"])
    ref = np.asfortranarray(np.load(ref_path).astype(np.float32, copy=False))
    if ref.shape != (hidden, 1):
        raise AssertionError(f"embed ref shape {ref.shape} != ({hidden}, 1)")

    model = transformer_model_me.create_model(**cfg)
    try:
        transformer_model_me.load_weights_from_safetensors_hf_llama(model, st_path)
        got = transformer_model_me.embed_forward_host(model, _TOKEN_IDS_T1, 1)
        got = np.asfortranarray(np.asarray(got, dtype=np.float32))
        if got.shape != ref.shape:
            raise AssertionError(f"embed engine shape {got.shape} != {ref.shape}")
        max_abs = float(np.max(np.abs(got - ref)))
        print(f"  embed_t1 max_abs={max_abs:.6g}")
        print("Passed test_hf_llama_real_slice_embed_t1 probe")
    finally:
        transformer_model_me.destroy_model(model)


def test_hf_llama_real_slice_hidden_t1() -> None:
    # 从已对齐的 embed 进 Engine 两层+final_norm，和 HF final hidden 比。
    slice_dir = _slice_dir()
    if not slice_dir:
        print("Passed test_hf_llama_real_slice_hidden_t1 skipped")
        return
    st_path = os.path.join(slice_dir, "model.safetensors")
    cfg_path = os.path.join(slice_dir, "config.txt")
    embed_path = os.path.join(slice_dir, "ref_embed_t1.npy")
    final_path = os.path.join(slice_dir, "ref_hidden_final_t1.npy")
    if not os.path.isfile(embed_path) or not os.path.isfile(final_path):
        print("Passed test_hf_llama_real_slice_hidden_t1 skipped (no ref npy)")
        return

    cfg = _parse_config_txt(cfg_path)
    hidden = int(cfg["hidden_size"])
    embed = np.asfortranarray(np.load(embed_path).astype(np.float32, copy=False))
    ref = np.asfortranarray(np.load(final_path).astype(np.float32, copy=False))
    if embed.shape != (hidden, 1) or ref.shape != (hidden, 1):
        raise AssertionError(f"hidden ref shapes embed={embed.shape} final={ref.shape}")

    model = transformer_model_me.create_model(**cfg)
    engine = None
    try:
        transformer_model_me.load_weights_from_safetensors_hf_llama(model, st_path)
        engine = inference_engine_me.create_engine(model)
        out = np.zeros((hidden, 1), dtype=np.float32, order="F")
        inference_engine_me.forward_hidden_host(engine, 1, 0, embed, out)
        max_abs = float(np.max(np.abs(out - ref)))
        finite = bool(np.isfinite(out).all())
        print(f"  hidden_final_t1 max_abs={max_abs:.6g} finite={finite}")
        print("Passed test_hf_llama_real_slice_hidden_t1 probe")
    finally:
        if engine is not None:
            inference_engine_me.destroy_engine(engine)
        transformer_model_me.destroy_model(model)


if __name__ == "__main__":
    test_hf_llama_real_slice_logits_t1()
    test_hf_llama_real_slice_embed_t1()
    test_hf_llama_real_slice_hidden_t1()
    test_hf_llama_real_slice_logits()
