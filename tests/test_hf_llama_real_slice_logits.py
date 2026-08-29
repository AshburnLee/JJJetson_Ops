"""4.2：Engine prefill logits vs 4.1 dump 的 npy（唯一数值基准）。

未设置 JJ_HF_LLAMA_SLICE_DIR、或切片目录里没有 ref npy 时跳过。
不要 import transformers（dump 已在 scripts/export_hf_llama_slice_logits.py 做完）。
不要在 Python 里再 load 一份完整权重 numpy：只读 config.txt，H2D 由 C++ Loader 完成。
不要和 fixture tensor ref 再比一遍。

例：token_ids=[1,2,3,4]，npy 与 Engine 都是 [32000, 4]、列主序。
    logits[v, t] 是第 t 个输入位置、词表第 v 维。只比这两张表。
T=4 是 4.2 的 gate；T=1 是 4.1c.5 留下的回归。argmax 一致，max_abs 是 FA fp16 噪声（约 1e-3），不是旧的 ~13。
"""

import os

import inference_engine_me
import linear_me
import numpy as np
import rms_norm_me
import rope_global_cache_me
import transformer_model_me
import weight_loader_me

from test_hf_llama_real_slice_smoke import _parse_config_txt, _slice_dir

_REF_NPY = "ref_prefill_logits_t1234.npy"
_REF_NPY_T1 = "ref_prefill_logits_t1.npy"
_TOKEN_IDS = np.array([1, 2, 3, 4], dtype=np.int32)
_TOKEN_IDS_T1 = np.array([1], dtype=np.int32)

# FA 的 Q/K/V staging 是 IEEE fp16，HF dump 是全 F32 attention。
# 4.1c 实测：T=1 max_abs≈8.8e-4，T=4 max_abs≈3.6e-3；旧 Linear W^T bug 是 ~13。
_ATOL = 1e-2
_RTOL = 1e-2


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
    # 4.1c.5：T=1 验收，argmax 一致且 max_abs 落在 fp16 噪声量级。
    _compare_engine_to_npy(_TOKEN_IDS_T1, _REF_NPY_T1, assert_close=True)


def test_hf_llama_real_slice_logits() -> None:
    # 4.2：token [1,2,3,4] vs ref_prefill_logits_t1234.npy。
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


def test_hf_llama_real_slice_q_rope_t1() -> None:
    # 4.1c.1：只用现有 host 算子搭到 layer0 RoPE 后的 Q，不进 FA。
    slice_dir = _slice_dir()
    if not slice_dir:
        print("Passed test_hf_llama_real_slice_q_rope_t1 skipped")
        return
    st_path = os.path.join(slice_dir, "model.safetensors")
    cfg_path = os.path.join(slice_dir, "config.txt")
    embed_path = os.path.join(slice_dir, "ref_embed_t1.npy")
    ref_path = os.path.join(slice_dir, "ref_q_rope_t1.npy")
    if not os.path.isfile(ref_path) or not os.path.isfile(embed_path):
        print("Passed test_hf_llama_real_slice_q_rope_t1 skipped (no ref npy)")
        return

    cfg = _parse_config_txt(cfg_path)
    hidden = int(cfg["hidden_size"])
    q_dim = int(cfg["num_q_heads"]) * int(cfg["head_dim"])
    embed = np.asfortranarray(np.load(embed_path).astype(np.float32, copy=False))
    ref = np.asfortranarray(np.load(ref_path).astype(np.float32, copy=False))
    if embed.shape != (hidden, 1) or ref.shape != (q_dim, 1):
        raise AssertionError(f"q_rope shapes embed={embed.shape} ref={ref.shape}")

    loaded = weight_loader_me.load_safetensors_hf_llama(st_path)
    tensors = loaded["tensors"]
    w_in = np.ascontiguousarray(tensors["layer0.w_input_layernorm"])
    w_q = np.ascontiguousarray(tensors["layer0.w_q"])
    eps = float(cfg["rms_norm_epsilon"])
    h_norm = np.zeros((hidden, 1), dtype=np.float32, order="F")
    rms_norm_me.forward_host(embed, w_in, h_norm, hidden, 1, eps)
    q_lin = np.zeros((q_dim, 1), dtype=np.float32, order="F")
    linear_me.forward_host(h_norm, w_q, q_lin, hidden, 1, q_dim)
    cache = rope_global_cache_me.create_cossin_cache(
        int(cfg["max_seq_len"]), int(cfg["head_dim"]), float(cfg["freq_base"])
    )
    q_out = np.zeros((q_dim, 1), dtype=np.float32, order="F")
    pos = np.array([0], dtype=np.int32)
    try:
        rope_global_cache_me.forward_host(
            cache, q_lin, pos, q_out, int(cfg["head_dim"]), int(cfg["num_q_heads"]), 1, 1
        )
    finally:
        rope_global_cache_me.destroy_cossin_cache(cache)
    max_abs = float(np.max(np.abs(q_out - ref)))
    print(f"  q_rope_t1 max_abs={max_abs:.6g}")
    if max_abs > 1e-4:
        raise AssertionError(f"layer0 RoPE Q vs HF dump max_abs={max_abs:.6g}")

    import fa_dst_unpack_me
    import fa_me
    import qkv_pack_fp16_me

    kv_dim = int(cfg["num_kv_heads"]) * int(cfg["head_dim"])
    w_k = np.ascontiguousarray(tensors["layer0.w_k"])
    w_v = np.ascontiguousarray(tensors["layer0.w_v"])
    w_o = np.ascontiguousarray(tensors["layer0.w_o"])
    k_lin = np.zeros((kv_dim, 1), dtype=np.float32, order="F")
    v_lin = np.zeros((kv_dim, 1), dtype=np.float32, order="F")
    linear_me.forward_host(h_norm, w_k, k_lin, hidden, 1, kv_dim)
    linear_me.forward_host(h_norm, w_v, v_lin, hidden, 1, kv_dim)
    k_out = np.zeros((kv_dim, 1), dtype=np.float32, order="F")
    cache_k = rope_global_cache_me.create_cossin_cache(
        int(cfg["max_seq_len"]), int(cfg["head_dim"]), float(cfg["freq_base"])
    )
    try:
        rope_global_cache_me.forward_host(
            cache_k, k_lin, pos, k_out, int(cfg["head_dim"]), int(cfg["num_kv_heads"]), 1, 1
        )
    finally:
        rope_global_cache_me.destroy_cossin_cache(cache_k)
    k_ref_path = os.path.join(slice_dir, "ref_k_rope_t1.npy")
    if os.path.isfile(k_ref_path):
        k_ref = np.asfortranarray(np.load(k_ref_path).astype(np.float32, copy=False))
        k_abs = float(np.max(np.abs(k_out - k_ref)))
        print(f"  k_rope_t1 max_abs={k_abs:.6g}")
        if k_abs > 1e-4:
            raise AssertionError(f"layer0 RoPE K vs HF dump max_abs={k_abs:.6g}")
    v_ref_path = os.path.join(slice_dir, "ref_v_t1.npy")
    if os.path.isfile(v_ref_path):
        v_ref = np.asfortranarray(np.load(v_ref_path).astype(np.float32, copy=False))
        v_abs = float(np.max(np.abs(v_lin - v_ref)))
        print(f"  v_t1 max_abs={v_abs:.6g}")
        if v_abs > 1e-4:
            raise AssertionError(f"layer0 V vs HF dump max_abs={v_abs:.6g}")

    head_dim = int(cfg["head_dim"])
    nq = int(cfg["num_q_heads"])
    nkv = int(cfg["num_kv_heads"])
    q_fp16 = np.zeros((head_dim, 1, nq, 1), dtype=np.uint16, order="F")
    k_fp16 = np.zeros((head_dim, 1, nkv, 1), dtype=np.uint16, order="F")
    v_fp16 = np.zeros((head_dim, 1, nkv, 1), dtype=np.uint16, order="F")
    qkv_pack_fp16_me.forward_host(q_out, q_fp16, head_dim, 1, nq)
    qkv_pack_fp16_me.forward_host(k_out, k_fp16, head_dim, 1, nkv)
    qkv_pack_fp16_me.forward_host(v_lin, v_fp16, head_dim, 1, nkv)
    fa_out = np.zeros((head_dim, 1, nq, 1), dtype=np.float32, order="F")
    fa_scale = 1.0 / (head_dim**0.5)
    fa_me.forward_host_shape(q_fp16, k_fp16, v_fp16, fa_out, fa_scale, 1, 0)
    flat = np.zeros((q_dim, 1), dtype=np.float32, order="F")
    fa_dst_unpack_me.forward_host(fa_out, flat, head_dim, 1, nq)
    attn = np.zeros((hidden, 1), dtype=np.float32, order="F")
    linear_me.forward_host(flat, w_o, attn, q_dim, 1, hidden)
    attn_ref_path = os.path.join(slice_dir, "ref_attn_out_t1.npy")
    if os.path.isfile(attn_ref_path):
        attn_ref = np.asfortranarray(np.load(attn_ref_path).astype(np.float32, copy=False))
        attn_abs = float(np.max(np.abs(attn - attn_ref)))
        print(f"  attn_out_t1 max_abs={attn_abs:.6g}")
    print("Passed test_hf_llama_real_slice_q_rope_t1")


if __name__ == "__main__":
    test_hf_llama_real_slice_logits_t1()
    test_hf_llama_real_slice_embed_t1()
    test_hf_llama_real_slice_q_rope_t1()
    test_hf_llama_real_slice_hidden_t1()
    test_hf_llama_real_slice_logits()
