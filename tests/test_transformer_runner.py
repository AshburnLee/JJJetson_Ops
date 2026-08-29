import elementwise_me
import fa_dst_unpack_me
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
    input_np: np.ndarray,
    residual_np: np.ndarray,
    weight_np: np.ndarray,
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_me = np.array(input_np, copy=True, order="F")
    residual_me = np.array(residual_np, copy=True, order="F")
    rms_norm_fused_add_me.forward_host(
        input_me, residual_me, weight_np, HIDDEN_SIZE, num_tokens, RMS_NORM_EPS
    )
    return input_me, residual_me


def _rope_qk_ref(
    q: np.ndarray, k: np.ndarray, pos_offset: int, num_tokens: int
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.arange(pos_offset, pos_offset + num_tokens, dtype=np.int32)
    q_out = np.zeros_like(q, order="F")
    k_out = np.zeros_like(k, order="F")
    cache = rope_global_cache_me.create_cossin_cache(MAX_SEQ_LEN, HEAD_DIM, ROPE_FREQ_BASE)
    try:
        rope_global_cache_me.forward_host(
            cache, q, pos, q_out, HEAD_DIM, NUM_Q_HEADS, num_tokens, 1
        )
        rope_global_cache_me.forward_host(
            cache, k, pos, k_out, HEAD_DIM, NUM_KV_HEADS, num_tokens, 1
        )
    finally:
        rope_global_cache_me.destroy_cossin_cache(cache)
    return q_out, k_out


class KvCacheRef:
    """Host 侧 KV cache ref，语义对齐 device：append 不改 len，advance 在 step 末。"""

    def __init__(self, head_dim: int, max_seq: int, num_kv_heads: int) -> None:
        self.head_dim = head_dim
        self.max_seq = max_seq
        self.num_kv_heads = num_kv_heads
        self.kv_dim = head_dim * num_kv_heads
        self.k_cache = np.zeros((head_dim, max_seq, num_kv_heads, 1), dtype=np.float32, order="F")
        self.v_cache = np.zeros((head_dim, max_seq, num_kv_heads, 1), dtype=np.float32, order="F")
        self.cache_len = 0

    def append(self, k_flat: np.ndarray, v_flat: np.ndarray, n_tokens: int) -> None:
        offset = self.cache_len
        k_2d = np.asarray(k_flat, dtype=np.float32, order="F").reshape(self.kv_dim, n_tokens)
        v_2d = np.asarray(v_flat, dtype=np.float32, order="F").reshape(self.kv_dim, n_tokens)
        for t in range(n_tokens):
            dst_t = offset + t
            for h in range(self.num_kv_heads):
                for d in range(self.head_dim):
                    row = d + self.head_dim * h
                    self.k_cache[d, dst_t, h, 0] = k_2d[row, t]
                    self.v_cache[d, dst_t, h, 0] = v_2d[row, t]

    def cast_fp16(self, num_kv_tokens: int) -> tuple[np.ndarray, np.ndarray]:
        k_flat = self._cache_to_flat(self.k_cache, num_kv_tokens)
        v_flat = self._cache_to_flat(self.v_cache, num_kv_tokens)
        k_fp16 = np.zeros(
            (self.head_dim, num_kv_tokens, self.num_kv_heads, 1), dtype=np.uint16, order="F"
        )
        v_fp16 = np.zeros(
            (self.head_dim, num_kv_tokens, self.num_kv_heads, 1), dtype=np.uint16, order="F"
        )
        qkv_pack_fp16_me.forward_host(
            k_flat, k_fp16, self.head_dim, num_kv_tokens, self.num_kv_heads
        )
        qkv_pack_fp16_me.forward_host(
            v_flat, v_fp16, self.head_dim, num_kv_tokens, self.num_kv_heads
        )
        return k_fp16, v_fp16

    def advance_len(self, n_tokens: int) -> None:
        self.cache_len += n_tokens

    def _cache_to_flat(self, cache: np.ndarray, num_kv_tokens: int) -> np.ndarray:
        flat = np.zeros((self.kv_dim, num_kv_tokens), dtype=np.float32, order="F")
        for t in range(num_kv_tokens):
            for h in range(self.num_kv_heads):
                for d in range(self.head_dim):
                    row = d + self.head_dim * h
                    flat[row, t] = cache[d, t, h, 0]
        return flat


def chain_linear_me_ref_step(
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
    pos_offset: int,
    kv_cache: KvCacheRef,
) -> np.ndarray:
    num_tokens = hidden_np.shape[1]
    batch = hidden_np.shape[3]

    residual_zero = np.zeros_like(hidden_np, order="F")
    h_norm, residual_h = _fused_add_norm(hidden_np, residual_zero, w_input_layernorm, num_tokens)

    q = np.zeros((Q_DIM, num_tokens, 1, batch), dtype=np.float32, order="F")
    k = np.zeros((KV_DIM, num_tokens, 1, batch), dtype=np.float32, order="F")
    v = np.zeros((KV_DIM, num_tokens, 1, batch), dtype=np.float32, order="F")
    linear_me.forward_host(h_norm, w_q, q, HIDDEN_SIZE, num_tokens, Q_DIM)
    linear_me.forward_host(h_norm, w_k, k, HIDDEN_SIZE, num_tokens, KV_DIM)
    linear_me.forward_host(h_norm, w_v, v, HIDDEN_SIZE, num_tokens, KV_DIM)

    q, k = _rope_qk_ref(q, k, pos_offset, num_tokens)

    q_fp16 = np.zeros((HEAD_DIM, num_tokens, NUM_Q_HEADS, 1), dtype=np.uint16, order="F")
    qkv_pack_fp16_me.forward_host(q, q_fp16, HEAD_DIM, num_tokens, NUM_Q_HEADS)

    cache_len_before = kv_cache.cache_len
    num_kv_tokens = cache_len_before + num_tokens
    kv_cache.append(k, v, num_tokens)
    k_fp16, v_fp16 = kv_cache.cast_fp16(num_kv_tokens)

    fa_out = np.zeros((HEAD_DIM, num_tokens, NUM_Q_HEADS, 1), dtype=np.float32, order="F")
    fa_scale = 1.0 / (HEAD_DIM**0.5)
    fa_me.forward_host_shape(q_fp16, k_fp16, v_fp16, fa_out, fa_scale, 1, cache_len_before)
    flat = np.zeros((Q_DIM, num_tokens), dtype=np.float32, order="F")
    fa_dst_unpack_me.forward_host(fa_out, flat, HEAD_DIM, num_tokens, NUM_Q_HEADS)
    fa_q = flat.reshape(Q_DIM, num_tokens, 1, batch, order="F")
    attn_out = np.zeros((HIDDEN_SIZE, num_tokens, 1, batch), dtype=np.float32, order="F")
    linear_me.forward_host(fa_q, w_o, attn_out, Q_DIM, num_tokens, HIDDEN_SIZE)

    h_ffn_in, residual_mid = _fused_add_norm(
        attn_out, residual_h, w_post_attention_layernorm, num_tokens
    )

    gate = np.zeros((INTERMEDIATE_SIZE, num_tokens, 1, batch), dtype=np.float32, order="F")
    up = np.zeros((INTERMEDIATE_SIZE, num_tokens, 1, batch), dtype=np.float32, order="F")
    linear_me.forward_host(h_ffn_in, w_gate, gate, HIDDEN_SIZE, num_tokens, INTERMEDIATE_SIZE)
    linear_me.forward_host(h_ffn_in, w_up, up, HIDDEN_SIZE, num_tokens, INTERMEDIATE_SIZE)

    ffn_mid = (_silu_np(gate) * up).astype(np.float32, order="F")
    ffn_out = np.zeros((HIDDEN_SIZE, num_tokens, 1, batch), dtype=np.float32, order="F")
    linear_me.forward_host(ffn_mid, w_down, ffn_out, INTERMEDIATE_SIZE, num_tokens, HIDDEN_SIZE)

    layer_out = np.zeros((HIDDEN_SIZE, num_tokens, 1, batch), dtype=np.float32, order="F")
    elementwise_me.forward_host("add", residual_mid, ffn_out, layer_out)

    kv_cache.advance_len(num_tokens)
    return layer_out


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
    kv = KvCacheRef(HEAD_DIM, MAX_SEQ_LEN, NUM_KV_HEADS)
    return chain_linear_me_ref_step(
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
        pos_offset,
        kv,
    )


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
    print("Passed test_transformer_runner")


if __name__ == "__main__":
    test_transformer_runner()
