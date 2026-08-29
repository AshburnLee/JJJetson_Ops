"""FA dst unpack：FA layout [head_dim, T, H] -> Linear flat [H*head_dim, T]。"""

import fa_dst_unpack_me
import numpy as np


def _ref_unpack(src: np.ndarray, head_dim: int, num_tokens: int, num_heads: int) -> np.ndarray:
    # 与 kernel 同一套索引。例 D=2,T=2,H=2：FA token0 head0=[1,2] -> 第 0 列前两维。
    dst = np.zeros((head_dim * num_heads, num_tokens), dtype=np.float32, order="F")
    for h in range(num_heads):
        for t in range(num_tokens):
            for d in range(head_dim):
                dst[h * head_dim + d, t] = src[d, t, h, 0]
    return dst


def test_fa_dst_unpack_tok4() -> None:
    head_dim = 64
    num_tokens = 4
    num_heads = 32
    rng = np.random.RandomState(4)
    src = np.asfortranarray(rng.randn(head_dim, num_tokens, num_heads, 1).astype(np.float32))
    dst = np.zeros((head_dim * num_heads, num_tokens), dtype=np.float32, order="F")
    fa_dst_unpack_me.forward_host(src, dst, head_dim, num_tokens, num_heads)
    ref = _ref_unpack(src, head_dim, num_tokens, num_heads)
    max_abs = float(np.max(np.abs(dst - ref)))
    if max_abs != 0.0:
        raise AssertionError(f"unpack tok4 max_abs={max_abs}")
    print("Passed test_fa_dst_unpack_tok4")


def test_fa_dst_unpack_t1_identity() -> None:
    # T=1 时 FA layout 与 Linear flat 碰巧同一块内存解释。
    head_dim = 8
    num_tokens = 1
    num_heads = 4
    rng = np.random.RandomState(1)
    src = np.asfortranarray(rng.randn(head_dim, num_tokens, num_heads, 1).astype(np.float32))
    dst = np.zeros((head_dim * num_heads, num_tokens), dtype=np.float32, order="F")
    fa_dst_unpack_me.forward_host(src, dst, head_dim, num_tokens, num_heads)
    flat_view = np.asarray(src, order="F").reshape((head_dim * num_heads, 1), order="F")
    max_abs = float(np.max(np.abs(dst - flat_view)))
    if max_abs != 0.0:
        raise AssertionError(f"unpack t1 identity max_abs={max_abs}")
    print("Passed test_fa_dst_unpack_t1_identity")


if __name__ == "__main__":
    test_fa_dst_unpack_tok4()
    test_fa_dst_unpack_t1_identity()
