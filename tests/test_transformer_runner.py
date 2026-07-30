import elementwise_me
import fa_me
import linear_me
import numpy as np
import qkv_pack_fp16_me
import rms_norm_fused_add_me
import rope_global_cache_me
import torch
import transformer_runner_me

import utils

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 32
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
NUM_TOKENS = 13
BATCH = 1
SEED = 24
MAX_SEQ_LEN = 256
ROPE_FREQ_BASE = 10000.0
RMS_NORM_EPS = 1e-6


def _silu_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    pos = x >= 0
    out = np.empty_like(x, dtype=np.float32)
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    x_neg = x[~pos]
    exp_x = np.exp(x_neg)
    out[~pos] = (x_neg * exp_x) / (1.0 + exp_x)
    return out


def _fused_add_norm(
    input_np: np.ndarray, residual_np: np.ndarray, weight_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    input_me = np.array(input_np, copy=True, order="F")
    residual_me = np.array(residual_np, copy=True, order="F")
    rms_norm_fused_add_me.forward_host(
        input_me, residual_me, weight_np, HIDDEN_SIZE, NUM_TOKENS, RMS_NORM_EPS
    )
    return input_me, residual_me


def _rope_qk_ref(q: np.ndarray, k: np.ndarray, pos_offset: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.arange(pos_offset, pos_offset + NUM_TOKENS, dtype=np.int32)
    q_out = np.zeros_like(q, order="F")
    k_out = np.zeros_like(k, order="F")
    cache = rope_global_cache_me.create_cossin_cache(MAX_SEQ_LEN, HEAD_DIM, ROPE_FREQ_BASE)
    try:
        rope_global_cache_me.forward_host(
            cache, q, pos, q_out, HEAD_DIM, NUM_Q_HEADS, NUM_TOKENS, 1
        )
        rope_global_cache_me.forward_host(
            cache, k, pos, k_out, HEAD_DIM, NUM_KV_HEADS, NUM_TOKENS, 1
        )
    finally:
        rope_global_cache_me.destroy_cossin_cache(cache)
    return q_out, k_out


# 手动串联 Pre-LN fused add + Linear + RoPE + FA + O Linear + FFN + Post-FFN residual add
def chain_linear_me_ref(
    hidden_np: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    w_input_layernorm: np.ndarray,
    w_post_attention_layernorm: np.ndarray,
    pos_offset: int = 0,
) -> np.ndarray:
    residual_zero = np.zeros_like(hidden_np, order="F")
    h_norm, residual_h = _fused_add_norm(hidden_np, residual_zero, w_input_layernorm)

    q = np.zeros((Q_DIM, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    k = np.zeros((KV_DIM, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    v = np.zeros((KV_DIM, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    linear_me.forward_host(h_norm, w_q, q, HIDDEN_SIZE, NUM_TOKENS, Q_DIM)
    linear_me.forward_host(h_norm, w_k, k, HIDDEN_SIZE, NUM_TOKENS, KV_DIM)
    linear_me.forward_host(h_norm, w_v, v, HIDDEN_SIZE, NUM_TOKENS, KV_DIM)

    q, k = _rope_qk_ref(q, k, pos_offset)

    q_fp16 = np.zeros((HEAD_DIM, NUM_TOKENS, NUM_Q_HEADS, 1), dtype=np.uint16, order="F")
    k_fp16 = np.zeros((HEAD_DIM, NUM_TOKENS, NUM_KV_HEADS, 1), dtype=np.uint16, order="F")
    v_fp16 = np.zeros((HEAD_DIM, NUM_TOKENS, NUM_KV_HEADS, 1), dtype=np.uint16, order="F")
    qkv_pack_fp16_me.forward_host(q, q_fp16, HEAD_DIM, NUM_TOKENS, NUM_Q_HEADS)
    qkv_pack_fp16_me.forward_host(k, k_fp16, HEAD_DIM, NUM_TOKENS, NUM_KV_HEADS)
    qkv_pack_fp16_me.forward_host(v, v_fp16, HEAD_DIM, NUM_TOKENS, NUM_KV_HEADS)

    fa_out = np.zeros((HEAD_DIM, NUM_TOKENS, NUM_Q_HEADS, 1), dtype=np.float32, order="F")
    fa_scale = 1.0 / (HEAD_DIM**0.5)
    fa_me.forward_host_shape(q_fp16, k_fp16, v_fp16, fa_out, fa_scale)

    fa_q = fa_out.reshape(Q_DIM, NUM_TOKENS, 1, BATCH, order="F")
    attn_out = np.zeros((HIDDEN_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    linear_me.forward_host(fa_q, w_o, attn_out, Q_DIM, NUM_TOKENS, HIDDEN_SIZE)

    h_ffn_in, residual_mid = _fused_add_norm(attn_out, residual_h, w_post_attention_layernorm)

    gate = np.zeros((INTERMEDIATE_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    up = np.zeros((INTERMEDIATE_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    linear_me.forward_host(h_ffn_in, w_gate, gate, HIDDEN_SIZE, NUM_TOKENS, INTERMEDIATE_SIZE)
    linear_me.forward_host(h_ffn_in, w_up, up, HIDDEN_SIZE, NUM_TOKENS, INTERMEDIATE_SIZE)

    ffn_mid = (_silu_np(gate) * up).astype(np.float32, order="F")
    ffn_out = np.zeros((HIDDEN_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    linear_me.forward_host(ffn_mid, w_down, ffn_out, INTERMEDIATE_SIZE, NUM_TOKENS, HIDDEN_SIZE)

    layer_out = np.zeros((HIDDEN_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    elementwise_me.forward_host("add", residual_mid, ffn_out, layer_out)
    return layer_out


# torch 的 GEMM 路径与 transformer_runner_me 不尽相同，与torch相比， backend 不同
# 7 层串联计算链太长，中间结果已不完全一样，
# silu(gate) * up 将误差非线性放大。所以torch不适合作为ref
# 这里与手动串联的 linear_me.forward_host 构成的同结构 Transformer 比较
def test_transformer_runner():
    np.random.seed(SEED)
    hidden_np = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )
    w_q = np.random.randn(Q_DIM, HIDDEN_SIZE).astype(np.float32)
    w_k = np.random.randn(KV_DIM, HIDDEN_SIZE).astype(np.float32)
    w_v = np.random.randn(KV_DIM, HIDDEN_SIZE).astype(np.float32)
    w_o = np.random.randn(HIDDEN_SIZE, Q_DIM).astype(np.float32)
    w_gate = np.random.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32)
    w_up = np.random.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32)
    w_down = np.random.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE).astype(np.float32)
    w_input_layernorm = np.random.randn(HIDDEN_SIZE).astype(np.float32)
    w_post_attention_layernorm = np.random.randn(HIDDEN_SIZE).astype(np.float32)

    # H2D * 9 (linear + norm weights; forward 仍含 norm，已接入)
    runner = transformer_runner_me.create_runner(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        MAX_SEQ_LEN,
        ROPE_FREQ_BASE,
        w_q,
        w_k,
        w_v,
        w_o,
        w_gate,
        w_up,
        w_down,
        w_input_layernorm,
        w_post_attention_layernorm,
    )

    output_me = np.zeros((HIDDEN_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    transformer_runner_me.forward_host(runner, NUM_TOKENS, 0, hidden_np, output_me)
    transformer_runner_me.destroy_runner(runner)

    ref_np = chain_linear_me_ref(
        hidden_np,
        w_q,
        w_k,
        w_v,
        w_o,
        w_gate,
        w_up,
        w_down,
        w_input_layernorm,
        w_post_attention_layernorm,
        pos_offset=0,
    )
    ok = utils.compare_np_torch(output_me, torch.from_numpy(ref_np), atol=1e-4, rtol=1e-4)
    assert ok, "transformer_runner output differs from chained Pre-LN + residual ref"
    print("Passed")


if __name__ == "__main__":
    test_transformer_runner()
