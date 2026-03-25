import numpy as np
import flash_attention_me

def to_5dp_strings(arr):
    return [f"{float(x):.5f}" for x in np.asarray(arr, dtype=np.float32).ravel()]

def warp_reduce_down_max_ref(row32: np.ndarray) -> np.float32:
    """
    模拟 CUDA warp_reduce_down_max:
      val = max(val, shfl_down(val, 16/8/4/2/1))
    约定 row32 长度为 32。
    """
    v = np.asarray(row32, dtype=np.float32).copy()
    for off in (16, 8, 4, 2, 1):
        v[:-off] = np.maximum(v[:-off], v[off:])
    return np.float32(v[0])

def warp_reduce_down_sum_ref(row32: np.ndarray) -> np.float32:
    """
    模拟 CUDA warp_reduce_down_sum:
      val += shfl_down(val, 16/8/4/2/1)
    约定 row32 长度为 32。
    """
    v = np.asarray(row32, dtype=np.float32).copy()
    for off in (16, 8, 4, 2, 1):
        v[:-off] = v[:-off] + v[off:]
    return np.float32(v[0])

def flash_attention_ref(Q, K, V):
    """
    与 flash_attn_tile_kernel 的逻辑一致（col-major 解释）:
    Q: float16, shape (128, 13, 16, 1)
    K: float16, shape (128, 256, 8, 1)
    V: float16, shape (128, 256, 8, 1)
    返回:
      dst: float32, shape (128, 13, 16, 1)
    注意: 与你的 kernel 一样，这里不额外乘 scale（kernel 里目前也没用 scale）。
    """
    Q = np.asarray(Q, dtype=np.float16)
    K = np.asarray(K, dtype=np.float16)
    V = np.asarray(V, dtype=np.float16)

    HEAD_DIM = 128
    TOK_Q = 13
    Q_HEADS = 16
    TOK_KV = 256
    KV_HEADS = 8
    KV_TILE = 32
    LOOP_KV = TOK_KV // KV_TILE  # 8

    assert Q.shape == (HEAD_DIM, TOK_Q, Q_HEADS, 1)
    assert K.shape == (HEAD_DIM, TOK_KV, KV_HEADS, 1)
    assert V.shape == (HEAD_DIM, TOK_KV, KV_HEADS, 1)

    dst = np.zeros((HEAD_DIM, TOK_Q, Q_HEADS, 1), dtype=np.float32)
    m_all = np.zeros((KV_HEADS, TOK_Q * 2), dtype=np.float32)
    l_all = np.zeros((KV_HEADS, TOK_Q * 2), dtype=np.float32)
    S_all = np.zeros((KV_HEADS, LOOP_KV, TOK_Q * 2, KV_TILE), dtype=np.float32)
    row_sum_all = np.zeros((KV_HEADS, LOOP_KV, TOK_Q * 2), dtype=np.float32)
    scale_old_all = np.zeros((KV_HEADS, LOOP_KV, TOK_Q * 2), dtype=np.float32)
    scale_new_all = np.zeros((KV_HEADS, LOOP_KV, TOK_Q * 2), dtype=np.float32)
    exp_val_all = np.zeros((KV_HEADS, LOOP_KV, TOK_Q * 2, KV_TILE), dtype=np.float32)

    for kv_head in range(KV_HEADS):               # 对应 blockIdx.x = 0..7
        # 该 block 负责两个 qhead：q0 与 q1
        q0 = kv_head * 2 + 0
        q1 = kv_head * 2 + 1

        # m/l 需要按 kernel 的 26 行维护：r = qhead_local*13 + token
        m = np.full((26,), -np.inf, dtype=np.float32)
        l = np.zeros((26,), dtype=np.float32)

        # -------- pass1: KV tiles，更新全局 m/l--------
        for tile_id in range(LOOP_KV):
            t0 = tile_id * KV_TILE
            t1 = t0 + KV_TILE

            # S0/S1: [13,32]
            if q0 < Q_HEADS:
                S0 = (Q[:, :, q0, 0].astype(np.float32).T @
                      K[:, t0:t1, kv_head, 0].astype(np.float32))
            else:
                S0 = np.full((TOK_Q, KV_TILE), -np.inf, dtype=np.float32)

            if q1 < Q_HEADS:
                S1 = (Q[:, :, q1, 0].astype(np.float32).T @
                      K[:, t0:t1, kv_head, 0].astype(np.float32))
            else:
                S1 = np.full((TOK_Q, KV_TILE), -np.inf, dtype=np.float32)

            # 拼成 26 行
            S = np.concatenate([S0, S1], axis=0)  # (26,32)
            S_all[kv_head, tile_id, :, :] = S

            # 线性reduce：
            row_max = np.max(S, axis=1)  # (26,)
            # 与 CUDA 一致：每行 32 列按 warp_reduce_down_max 树形归约得到 row_max
            # row_max = np.array([warp_reduce_down_max_ref(S[r, :]) for r in range(26)], dtype=np.float32)
            exp_mat = np.exp(S - row_max[:, None]).astype(np.float32)
            exp_val_all[kv_head, tile_id, :, :] = exp_mat
            row_sum = np.sum(np.exp(S - row_max[:, None]), axis=1)  # (26,)
            # row_sum = np.array([warp_reduce_down_sum_ref(exp_mat[r, :]) for r in range(26)], dtype=np.float32)

            m_new = np.maximum(m, row_max)
            scale_old = np.exp(m - m_new)
            scale_new = np.exp(row_max - m_new)

            row_sum_all[kv_head, tile_id, :] = row_sum
            scale_old_all[kv_head, tile_id, :] = scale_old
            scale_new_all[kv_head, tile_id, :] = scale_new

            l = l * scale_old + row_sum * scale_new
            m = m_new

        # 记录 ref 的 m/l（对应每个 block）
        m_all[kv_head, :] = m
        l_all[kv_head, :] = l

        # -------- pass2: softmax + 乘 V，累加到输出 --------
        out0 = np.zeros((TOK_Q, HEAD_DIM), dtype=np.float32)
        out1 = np.zeros((TOK_Q, HEAD_DIM), dtype=np.float32)

        for tile_id in range(LOOP_KV):
            t0 = tile_id * KV_TILE
            t1 = t0 + KV_TILE

            if q0 < Q_HEADS:
                S0 = (Q[:, :, q0, 0].astype(np.float32).T @
                      K[:, t0:t1, kv_head, 0].astype(np.float32))
                P0 = np.exp(S0 - m[0:TOK_Q, None]) / l[0:TOK_Q, None]
                out0 += P0 @ V[:, t0:t1, kv_head, 0].astype(np.float32).T

            if q1 < Q_HEADS:
                S1 = (Q[:, :, q1, 0].astype(np.float32).T @
                      K[:, t0:t1, kv_head, 0].astype(np.float32))
                P1 = np.exp(S1 - m[TOK_Q:2*TOK_Q, None]) / l[TOK_Q:2*TOK_Q, None]
                out1 += P1 @ V[:, t0:t1, kv_head, 0].astype(np.float32).T

        if q0 < Q_HEADS:
            dst[:, :, q0, 0] = out0.T
        if q1 < Q_HEADS:
            dst[:, :, q1, 0] = out1.T

    return dst, m_all, l_all, S_all, row_sum_all, scale_old_all, scale_new_all, exp_val_all

def test_fa():
    np.random.seed(24)

    # col-major (Fortran order): dim0 (head_dim) contiguous
    Q = np.asfortranarray(np.random.randn(128, 13, 16, 1).astype(np.float16))
    K = np.asfortranarray(np.random.randn(128, 256, 8, 1).astype(np.float16))
    V = np.asfortranarray(np.random.randn(128, 256, 8, 1).astype(np.float16))

    dst = np.zeros((128, 13, 16, 1), dtype=np.float32, order="F")
    scale = 1.0  # kernel 里目前未使用 scale，这里传 1.0 即可
    # 判断模块是否含有 debug 这个接口
    debug_mode = hasattr(flash_attention_me, "launch_flash_attention_debug_ml")

    if debug_mode:
        m_out = np.zeros((8, 26), dtype=np.float32)
        l_out = np.zeros((8, 26), dtype=np.float32)
        S_out = np.zeros((8, 8, 26, 32), dtype=np.float32)
        row_sum_out = np.zeros((8, 8, 26), dtype=np.float32)
        scale_old_out = np.zeros((8, 8, 26), dtype=np.float32)
        scale_new_out = np.zeros((8, 8, 26), dtype=np.float32)
        exp_val_out = np.zeros((8, 8, 26, 32), dtype=np.float32)

        flash_attention_me.launch_flash_attention_debug_ml(
            Q, K, V, dst, scale,
            m_out, l_out, S_out,
            row_sum_out, scale_old_out, scale_new_out,
            exp_val_out,
        )
    else:
        flash_attention_me.launch_flash_attention(Q, K, V, dst, scale)

    dst_ref, m_ref, l_ref, S_ref, row_sum_ref, scale_old_ref, scale_new_ref, exp_val_ref = flash_attention_ref(Q, K, V)

    print("ref dst[:,0,0,0]: \n", dst_ref[:,0,0,0])
    print("me  dst[:,0,0,0]: \n", dst[:,0,0,0])
    
    if debug_mode:
        # 对比 m/l
        ml_diff_m = np.max(np.abs(m_out - m_ref))
        ml_diff_l = np.max(np.abs(l_out - l_ref))
        print("max abs diff m:", ml_diff_m)
        print("max abs diff l:", ml_diff_l)

        # 对比 S_shared（pass1 的 8 tiles）
        s_shared_diff = np.max(np.abs(S_out - S_ref))
        print("max abs diff S_shared:", s_shared_diff)

        # 对比 row_sum / scale_old / scale_new
        row_sum_diff = np.max(np.abs(row_sum_out - row_sum_ref))
        scale_old_diff = np.max(np.abs(scale_old_out - scale_old_ref))
        scale_new_diff = np.max(np.abs(scale_new_out - scale_new_ref))
        exp_diff = np.max(np.abs(exp_val_out - exp_val_ref))
        print("max abs diff row_sum:", row_sum_diff)
        print("max abs diff scale_old:", scale_old_diff)
        print("max abs diff scale_new:", scale_new_diff)
        print("max abs diff exp_val:", exp_diff)

    # 对比最终 dst
    diff = np.max(np.abs(dst - dst_ref))
    print("max abs diff dst:", diff)
    # 对比最终 diff，不close 返回failed，allclose 返回 pass
    if not np.allclose(dst, dst_ref, rtol=1e-3, atol=1e-3):
        print("Failed")
    else:
        print("Passed")

    # 打印全部内容（8 blocks * 8 tiles * 26 rows）
    # for b in range(8):
    #     for t in range(8):
            # print(f"[cu ] block {b} tile {t} row_sum[26]:", to_5dp_strings(row_sum_out[b, t]))
            # print(f"[ref] block {b} tile {t} row_sum[26]:", to_5dp_strings(row_sum_ref[b, t]))
            # print(f"[cu ] block {b} tile {t} scale_old[26]:", to_5dp_strings(scale_old_out[b, t]))
            # print(f"[ref] block {b} tile {t} scale_old[26]:", to_5dp_strings(scale_old_ref[b, t]))
            # print(f"[cu ] block {b} tile {t} scale_new[26]:", to_5dp_strings(scale_new_out[b, t]))
            # print(f"[ref] block {b} tile {t} scale_new[26]:", to_5dp_strings(scale_new_ref[b, t]))
            # print(f"[cu ] block {b} tile {t} exp_val row0[32]:", to_5dp_strings(exp_val_out[b, t, 0]))
            # print(f"[ref] block {b} tile {t} exp_val row0[32]:", to_5dp_strings(exp_val_ref[b, t, 0]))
            # pass


if __name__ == "__main__":
    test_fa()
