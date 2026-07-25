"""
fa 系列 kernel 测试共用：参考实现、随机输入、数值断言。
与 fa_me / fa_tc_me 解耦，仅依赖 numpy。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# 与 CUDA 侧形状一致（列主序 Q/K/V）
HEAD_DIM = 128
TOK_Q = 13
Q_HEADS = 16
TOK_KV = 256
KV_HEADS = 8
KV_TILE = 32
LOOP_KV = TOK_KV // KV_TILE
DST_SHAPE = (HEAD_DIM, TOK_Q, Q_HEADS, 1)


def to_5dp_strings(arr: np.ndarray) -> list:
    return [f"{float(x):.5f}" for x in np.asarray(arr, dtype=np.float32).ravel()]


def warp_reduce_down_max_ref(row32: np.ndarray) -> np.float32:
    """模拟 CUDA warp_reduce_down_max，row32 长度 32。"""
    v = np.asarray(row32, dtype=np.float32).copy()
    for off in (16, 8, 4, 2, 1):
        v[:-off] = np.maximum(v[:-off], v[off:])
    return np.float32(v[0])


def warp_reduce_down_sum_ref(row32: np.ndarray) -> np.float32:
    """模拟 CUDA warp_reduce_down_sum，row32 长度 32。"""
    v = np.asarray(row32, dtype=np.float32).copy()
    for off in (16, 8, 4, 2, 1):
        v[:-off] = v[:-off] + v[off:]
    return np.float32(v[0])


def fa_ref(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float = 1.0) -> tuple[Any, ...]:
    """
    与 fa kernel 一致的两遍式参考（col-major）；shape 由 Q/K/V 数组推断。
    返回 (dst, m_all, l_all, S_all, row_sum_all, scale_old_all, scale_new_all, exp_val_all)。
    """
    Q = np.asarray(Q, dtype=np.float16)
    K = np.asarray(K, dtype=np.float16)
    V = np.asarray(V, dtype=np.float16)

    head_dim = Q.shape[0]
    tok_q = Q.shape[1]
    q_heads = Q.shape[2]
    tok_kv = K.shape[1]
    kv_heads = K.shape[2]
    kv_tile = KV_TILE
    loop_kv = (tok_kv + kv_tile - 1) // kv_tile
    rows_two = tok_q * 2
    dst_shape = (head_dim, tok_q, q_heads, 1)

    assert K.shape == (head_dim, tok_kv, kv_heads, 1)
    assert V.shape == (head_dim, tok_kv, kv_heads, 1)
    assert q_heads == kv_heads * 2

    dst = np.zeros(dst_shape, dtype=np.float32)
    m_all = np.zeros((kv_heads, rows_two), dtype=np.float32)
    l_all = np.zeros((kv_heads, rows_two), dtype=np.float32)
    S_all = np.zeros((kv_heads, loop_kv, rows_two, kv_tile), dtype=np.float32)
    row_sum_all = np.zeros((kv_heads, loop_kv, rows_two), dtype=np.float32)
    scale_old_all = np.zeros((kv_heads, loop_kv, rows_two), dtype=np.float32)
    scale_new_all = np.zeros((kv_heads, loop_kv, rows_two), dtype=np.float32)
    exp_val_all = np.zeros((kv_heads, loop_kv, rows_two, kv_tile), dtype=np.float32)

    for kv_head in range(kv_heads):
        q0 = kv_head * 2 + 0
        q1 = kv_head * 2 + 1
        m = np.full((rows_two,), -np.inf, dtype=np.float32)
        ell = np.zeros((rows_two,), dtype=np.float32)

        for tile_id in range(loop_kv):
            t0 = tile_id * kv_tile
            t1 = min(t0 + kv_tile, tok_kv)
            cols = kv_tile
            if q0 < q_heads:
                S0 = Q[:, :tok_q, q0, 0].astype(np.float32).T @ K[:, t0:t1, kv_head, 0].astype(
                    np.float32
                )
                if S0.shape[1] < cols:
                    pad = np.full((tok_q, cols - S0.shape[1]), -np.inf, dtype=np.float32)
                    S0 = np.concatenate([S0, pad], axis=1)
            else:
                S0 = np.full((tok_q, cols), -np.inf, dtype=np.float32)
            if q1 < q_heads:
                S1 = Q[:, :tok_q, q1, 0].astype(np.float32).T @ K[:, t0:t1, kv_head, 0].astype(
                    np.float32
                )
                if S1.shape[1] < cols:
                    pad = np.full((tok_q, cols - S1.shape[1]), -np.inf, dtype=np.float32)
                    S1 = np.concatenate([S1, pad], axis=1)
            else:
                S1 = np.full((tok_q, cols), -np.inf, dtype=np.float32)
            for c in range(cols):
                if t0 + c >= tok_kv:
                    S0[:, c] = -np.inf
                    S1[:, c] = -np.inf
            S = np.concatenate([S0, S1], axis=0) * scale
            S_all[kv_head, tile_id, :, :] = S
            row_max = np.max(S, axis=1)
            exp_mat = np.exp(S - row_max[:, None]).astype(np.float32)
            exp_val_all[kv_head, tile_id, :, :] = exp_mat
            row_sum = np.sum(np.exp(S - row_max[:, None]), axis=1)
            m_new = np.maximum(m, row_max)
            scale_old = np.exp(m - m_new)
            scale_new = np.exp(row_max - m_new)
            row_sum_all[kv_head, tile_id, :] = row_sum
            scale_old_all[kv_head, tile_id, :] = scale_old
            scale_new_all[kv_head, tile_id, :] = scale_new
            ell = ell * scale_old + row_sum * scale_new
            m = m_new

        m_all[kv_head, :] = m
        l_all[kv_head, :] = ell

        out0 = np.zeros((tok_q, head_dim), dtype=np.float32)
        out1 = np.zeros((tok_q, head_dim), dtype=np.float32)
        for tile_id in range(loop_kv):
            t0 = tile_id * kv_tile
            t1 = min(t0 + kv_tile, tok_kv)
            cols = kv_tile
            if q0 < q_heads:
                S0 = Q[:, :tok_q, q0, 0].astype(np.float32).T @ K[:, t0:t1, kv_head, 0].astype(
                    np.float32
                )
                if S0.shape[1] < cols:
                    pad = np.full((tok_q, cols - S0.shape[1]), -np.inf, dtype=np.float32)
                    S0 = np.concatenate([S0, pad], axis=1)
                for c in range(cols):
                    if t0 + c >= tok_kv:
                        S0[:, c] = -np.inf
                P0 = np.exp(S0 * scale - m[0:tok_q, None]) / ell[0:tok_q, None]
                v_tile = V[:, t0:t1, kv_head, 0].astype(np.float32).T
                if v_tile.shape[0] < cols:
                    v_pad = np.zeros((cols - v_tile.shape[0], head_dim), dtype=np.float32)
                    v_tile = np.concatenate([v_tile, v_pad], axis=0)
                out0 += P0 @ v_tile
            if q1 < q_heads:
                S1 = Q[:, :tok_q, q1, 0].astype(np.float32).T @ K[:, t0:t1, kv_head, 0].astype(
                    np.float32
                )
                if S1.shape[1] < cols:
                    pad = np.full((tok_q, cols - S1.shape[1]), -np.inf, dtype=np.float32)
                    S1 = np.concatenate([S1, pad], axis=1)
                for c in range(cols):
                    if t0 + c >= tok_kv:
                        S1[:, c] = -np.inf
                P1 = np.exp(S1 * scale - m[tok_q : 2 * tok_q, None]) / ell[tok_q : 2 * tok_q, None]
                v_tile = V[:, t0:t1, kv_head, 0].astype(np.float32).T
                if v_tile.shape[0] < cols:
                    v_pad = np.zeros((cols - v_tile.shape[0], head_dim), dtype=np.float32)
                    v_tile = np.concatenate([v_tile, v_pad], axis=0)
                out1 += P1 @ v_tile
        if q0 < q_heads:
            dst[:, :, q0, 0] = out0.T
        if q1 < q_heads:
            dst[:, :, q1, 0] = out1.T

    return dst, m_all, l_all, S_all, row_sum_all, scale_old_all, scale_new_all, exp_val_all


def fa_ref_dst_only(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """仅返回 dst，供 test_fa_tc 等与 debug 无关的用例。"""
    return fa_ref(Q, K, V, scale)[0]


def empty_dst_f(shape: tuple[int, ...] | None = None) -> np.ndarray:
    if shape is None:
        shape = DST_SHAPE
    return np.zeros(shape, dtype=np.float32, order="F")


def random_fa_inputs_for_shape(
    head_dim: int,
    tok_q: int,
    q_heads: int,
    tok_kv: int,
    kv_heads: int,
    seed: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """列主序随机 Q/K/V，shape 由参数指定（q_heads 须为 2*kv_heads）。"""
    assert q_heads == kv_heads * 2
    np.random.seed(seed)
    q_shape = (head_dim, tok_q, q_heads, 1)
    kv_shape = (head_dim, tok_kv, kv_heads, 1)
    Q = np.asfortranarray(np.random.randn(*q_shape).astype(np.float16))
    K = np.asfortranarray(np.random.randn(*kv_shape).astype(np.float16))
    V = np.asfortranarray(np.random.randn(*kv_shape).astype(np.float16))
    return Q, K, V


def random_fa_inputs(seed: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """列主序随机 Q/K/V（与历史 np.random.seed + randn 行为一致）。"""
    return random_fa_inputs_for_shape(HEAD_DIM, TOK_Q, Q_HEADS, TOK_KV, KV_HEADS, seed)


def assert_dst_close(
    name: str,
    dst: np.ndarray,
    dst_ref: np.ndarray,
    rtol: float = 0.05,
    atol: float = 0.05,
) -> None:
    """
    默认容差 0.05: 与 fa_ref (全程 FP32) 对比时, TC 版本的 kernel 在 QK 与
    s_scores/softmax 链路上用 FP16 累加/存储(S、P 都是 half), 多 tile online softmax
    会放大与 FP32 参考的偏差; max abs diff 常落在约 0.05 以内。这是个取舍, 非逻辑bug
    """
    diff = np.max(np.abs(dst - dst_ref))
    ok = np.allclose(dst, dst_ref, rtol=rtol, atol=atol)
    print(f"[{name}] max abs diff dst vs ref: {diff}, allclose: {ok}")
    if not ok:
        print(f"[{name}] dst_ref[:, 0, 0]:\n", dst_ref[:, 0, 0])
        print(f"[{name}] dst[:, 0, 0]:\n", dst[:, 0, 0])
        raise SystemExit(1)


def run_launcher(
    launcher,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
) -> np.ndarray:
    dst_shape = (Q.shape[0], Q.shape[1], Q.shape[2], 1)
    dst = empty_dst_f(dst_shape)
    launcher(Q, K, V, dst, scale)
    return np.array(dst, copy=True)


def debug_ml_enabled(module: Any) -> bool:
    return hasattr(module, "launch_fa_debug_ml") and os.environ.get("DEBUG_MY_OPS", "") == "1"
