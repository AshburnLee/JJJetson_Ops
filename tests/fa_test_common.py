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


def fa_ref(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[Any, ...]:
    """
    与 fa kernel 一致的两遍式参考（col-major）。
    返回 (dst, m_all, l_all, S_all, row_sum_all, scale_old_all, scale_new_all, exp_val_all)。
    """
    Q = np.asarray(Q, dtype=np.float16)
    K = np.asarray(K, dtype=np.float16)
    V = np.asarray(V, dtype=np.float16)

    assert Q.shape == (HEAD_DIM, TOK_Q, Q_HEADS, 1)
    assert K.shape == (HEAD_DIM, TOK_KV, KV_HEADS, 1)
    assert V.shape == (HEAD_DIM, TOK_KV, KV_HEADS, 1)

    dst = np.zeros(DST_SHAPE, dtype=np.float32)
    m_all = np.zeros((KV_HEADS, TOK_Q * 2), dtype=np.float32)
    l_all = np.zeros((KV_HEADS, TOK_Q * 2), dtype=np.float32)
    S_all = np.zeros((KV_HEADS, LOOP_KV, TOK_Q * 2, KV_TILE), dtype=np.float32)
    row_sum_all = np.zeros((KV_HEADS, LOOP_KV, TOK_Q * 2), dtype=np.float32)
    scale_old_all = np.zeros((KV_HEADS, LOOP_KV, TOK_Q * 2), dtype=np.float32)
    scale_new_all = np.zeros((KV_HEADS, LOOP_KV, TOK_Q * 2), dtype=np.float32)
    exp_val_all = np.zeros((KV_HEADS, LOOP_KV, TOK_Q * 2, KV_TILE), dtype=np.float32)

    for kv_head in range(KV_HEADS):
        q0 = kv_head * 2 + 0
        q1 = kv_head * 2 + 1
        m = np.full((26,), -np.inf, dtype=np.float32)
        ell = np.zeros((26,), dtype=np.float32)

        for tile_id in range(LOOP_KV):
            t0 = tile_id * KV_TILE
            t1 = t0 + KV_TILE
            if q0 < Q_HEADS:
                S0 = Q[:, :, q0, 0].astype(np.float32).T @ K[:, t0:t1, kv_head, 0].astype(
                    np.float32
                )
            else:
                S0 = np.full((TOK_Q, KV_TILE), -np.inf, dtype=np.float32)
            if q1 < Q_HEADS:
                S1 = Q[:, :, q1, 0].astype(np.float32).T @ K[:, t0:t1, kv_head, 0].astype(
                    np.float32
                )
            else:
                S1 = np.full((TOK_Q, KV_TILE), -np.inf, dtype=np.float32)
            S = np.concatenate([S0, S1], axis=0)
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

        out0 = np.zeros((TOK_Q, HEAD_DIM), dtype=np.float32)
        out1 = np.zeros((TOK_Q, HEAD_DIM), dtype=np.float32)
        for tile_id in range(LOOP_KV):
            t0 = tile_id * KV_TILE
            t1 = t0 + KV_TILE
            if q0 < Q_HEADS:
                S0 = Q[:, :, q0, 0].astype(np.float32).T @ K[:, t0:t1, kv_head, 0].astype(
                    np.float32
                )
                P0 = np.exp(S0 - m[0:TOK_Q, None]) / ell[0:TOK_Q, None]
                out0 += P0 @ V[:, t0:t1, kv_head, 0].astype(np.float32).T
            if q1 < Q_HEADS:
                S1 = Q[:, :, q1, 0].astype(np.float32).T @ K[:, t0:t1, kv_head, 0].astype(
                    np.float32
                )
                P1 = np.exp(S1 - m[TOK_Q : 2 * TOK_Q, None]) / ell[TOK_Q : 2 * TOK_Q, None]
                out1 += P1 @ V[:, t0:t1, kv_head, 0].astype(np.float32).T
        if q0 < Q_HEADS:
            dst[:, :, q0, 0] = out0.T
        if q1 < Q_HEADS:
            dst[:, :, q1, 0] = out1.T

    return dst, m_all, l_all, S_all, row_sum_all, scale_old_all, scale_new_all, exp_val_all


def fa_ref_dst_only(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """仅返回 dst，供 test_fa_tc 等与 debug 无关的用例。"""
    return fa_ref(Q, K, V)[0]


def random_fa_inputs(seed: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """列主序随机 Q/K/V（与历史 np.random.seed + randn 行为一致）。"""
    np.random.seed(seed)
    Q = np.asfortranarray(np.random.randn(*DST_SHAPE).astype(np.float16))
    K = np.asfortranarray(np.random.randn(HEAD_DIM, TOK_KV, KV_HEADS, 1).astype(np.float16))
    V = np.asfortranarray(np.random.randn(HEAD_DIM, TOK_KV, KV_HEADS, 1).astype(np.float16))
    return Q, K, V


def empty_dst_f() -> np.ndarray:
    return np.zeros(DST_SHAPE, dtype=np.float32, order="F")


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
        print(f"[{name}] dst_ref[:, 0, 0, 0]:\n", dst_ref[:, 0, 0, 0])
        print(f"[{name}] dst[:, 0, 0, 0]:\n", dst[:, 0, 0, 0])
        raise SystemExit(1)


def run_launcher(
    launcher,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
) -> np.ndarray:
    dst = empty_dst_f()
    launcher(Q, K, V, dst, scale)
    return np.array(dst, copy=True)


def debug_ml_enabled(module: Any) -> bool:
    return hasattr(module, "launch_fa_debug_ml") and os.environ.get("DEBUG_MY_OPS", "") == "1"
