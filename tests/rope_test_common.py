"""
RoPE 测试共用：形状、随机种子、输入构造
"""

from __future__ import annotations

import numpy as np

# base case col-major [D, H, T, B]
ROPE_D = 128
ROPE_H = 16
ROPE_T = 13
ROPE_B = 1
ROPE_SEED = 24
ROPE_FREQ_BASE = 10000.0

ROPE_DIMS = [ROPE_D, ROPE_H, ROPE_T, ROPE_B]


def make_rope_inputs(seed: int = ROPE_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    np.random.seed(seed)
    input_f = np.random.randn(ROPE_D, ROPE_H, ROPE_T, ROPE_B).astype(np.float32)
    input_np = np.asfortranarray(input_f)
    pos_np = np.arange(ROPE_T, dtype=np.int32)
    output_np = np.zeros_like(input_np, order="F")
    return input_np, pos_np, output_np, ROPE_DIMS.copy()


def rope_ref_single(vec: np.ndarray, pos: float, base: float = ROPE_FREQ_BASE) -> np.ndarray:
    d = vec.shape[0]
    half = d // 2
    x0 = vec[:half]
    x1 = vec[half:d]

    k = np.arange(half, dtype=np.float32)
    theta_scale = base ** (-2.0 / d)
    theta = pos * (theta_scale**k)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    y0 = x0 * cos_theta - x1 * sin_theta
    y1 = x0 * sin_theta + x1 * cos_theta

    out = vec.copy()
    out[:half] = y0
    out[half:d] = y1
    return out


def check_token(
    input_np: np.ndarray,
    output_np: np.ndarray,
    pos_np: np.ndarray,
    head: int,
    token: int,
    *,
    verbose: bool = True,
) -> bool:
    p = float(pos_np[token])
    vec_in = input_np[:, head, token, 0].copy()
    vec_out = output_np[:, head, token, 0].copy()
    ref = rope_ref_single(vec_in, p)
    max_error = float(np.max(np.abs(vec_out - ref)))
    if verbose:
        print(f"head {head} token {token} max abs diff:", max_error)
    ok = max_error <= 1e-5
    if verbose and not ok:
        print(f"head {head} token {token} input:\n", vec_in)
        print(f"head {head} token {token} output:\n", vec_out)
    return ok


def check_all_tokens(
    input_np: np.ndarray, output_np: np.ndarray, pos_np: np.ndarray, *, verbose: bool = True
) -> bool:
    all_ok = True
    for h in range(ROPE_H):
        for t in range(ROPE_T):
            all_ok = check_token(input_np, output_np, pos_np, h, t, verbose=verbose) and all_ok
    return all_ok
