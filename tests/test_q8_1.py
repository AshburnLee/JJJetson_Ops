import numpy as np
import q8_1_me

def q8_1_ref(input_np: np.ndarray):
    """
    输入:
    - input_np: float32, shape = (2048, 13, 1, 1), Fortran order (col-major)

    返回:
    - qs_ref:   (n_blocks, 128) int8    # 每 block 128 个量化值
    - d_ref:    (n_blocks, 4)   float32 # 每 block 4 组 scale, 每组 32 个值
    - sum_ref:  (n_blocks, 4)   float32 # 每 block 4 组 sum, 每组 32 个值

    每 32 个值一组，128 里一共 4 组，而且这 32 个值来自 warp 里 8 个线程，每线程 4 个值,
    这个值由 __shfl_xor_sync 的 offset 和 QK8_1 决定的
    """
    assert input_np.dtype == np.float32
    x = np.asfortranarray(input_np)

    ne0, ne1, ne2, ne3 = x.shape      # 2048, 13, 1, 1
    blocks_y = 4
    vals_per_block = 128
    vals_per_scale = 32
    qs_ref = np.zeros((ne1 * 16, 128), dtype=np.int8)
    d_ref  = np.zeros((ne1 * 16, 4),   dtype=np.float32)
    sum_ref= np.zeros((ne1 * 16, 4),   dtype=np.float32)
    for j in range(ne1):          # blockIdx.x
        for by in range(blocks_y):   # blockIdx.y
            for warp in range(4):    # warp = threadIdx.x // 32
                group = 4 * by + warp           # 0..15
                block_id = group * ne1 + j      # 与 id_block 一致
                start = group * 128
                end = min(start + 128, ne0)
                vals = np.zeros(128, dtype=np.float32)
                if start < ne0:
                    vals[: end - start] = x[start:end, j, 0, 0]

                # 在 128 个值内部，再按 4 组，每组 32 个值分别计算 scale/sum
                for g in range(4):
                    s = g * vals_per_scale
                    e = s + vals_per_scale
                    group = vals[s:e]

                    amax = np.max(np.abs(group))
                    if amax > 0.0:
                        d_inv = 127.0 / amax
                        d = 1.0 / d_inv
                    else:
                        d_inv = 0.0
                        d = 0.0

                    q_group = np.round(group * d_inv).astype(np.int8)
                    qs_ref[block_id, s:e] = q_group

                    d_ref[block_id, g] = d
                    sum_ref[block_id, g] = float(group.sum())

                block_id += 1

    return qs_ref, d_ref, sum_ref


def test_all():
    # 1. 准备输入：col-major，shape = (2048, 13, 1, 1)
    # dim0: 2048 (变化最快)
    # dim1: 13   (seq_len)
    # dim2: 1
    # dim3: 1
    np.random.seed(24)
    input_f = np.random.randn(2048, 13, 1, 1).astype(np.float32)
    input_np = np.asfortranarray(input_f)

    n_blocks = 13 * ((2048 + 128 - 1) // 128)  # 13 * 16 = 208

    # 2. 分配输出缓冲
    # sizeof(block_q8_1_mmq) = 144
    q_output = np.zeros(n_blocks * 144, dtype=np.uint8)

    # 3. 调用量化（dims 按 col-major 物理顺序传入）
    q8_1_me.quantize(input_np, q_output, [2048, 13, 1, 1])

    # 4. 解析
    qs_int8, scale_fp32, sum_fp32 = q8_1_me.parse(q_output, n_blocks)

    print("qs_int8 shape:", qs_int8.shape)       # (208, 128)
    print("scale_fp32 shape:", scale_fp32.shape) # (208, 4)
    print("sum_fp32 shape:", sum_fp32.shape)     # (208, 4)
    print("qs_int8 dtype:", qs_int8.dtype)
    print("scale_fp32 dtype:", scale_fp32.dtype)
    print("sum_fp32 dtype:", sum_fp32.dtype)

    qs_ref, scale_ref, sum_ref = q8_1_ref(input_np)

    print("first block qs_int8:\n", qs_int8[0, :])
    print("first block qs_ref: \n", qs_ref[0, :])
    print("first block scales:\n", scale_fp32[0, :])
    print("first block scales ref:\n", scale_ref[0, :])
    print("first block sums:\n", sum_fp32[0, :])
    print("first block sums ref:\n", sum_ref[0, :])

    print("second block qs_int8:\n", qs_int8[1, :])
    print("second block qs_ref: \n", qs_ref[1, :])
    print("second block scales:\n", scale_fp32[1, :])
    print("second block scales ref:\n", scale_ref[1, :])
    print("second block sums:\n", sum_fp32[1, :])
    print("second block sums ref:\n", sum_ref[1, :])

    # 允许极小的浮点误差
    print("max |qs diff|:", np.max(np.abs(qs_int8 - qs_ref)))
    print("max |scale diff|:", np.max(np.abs(scale_fp32 - scale_ref)))
    print("max |sum diff|:", np.max(np.abs(sum_fp32 - sum_ref)))

    assert qs_int8.shape == qs_ref.shape
    assert scale_fp32.shape == scale_ref.shape
    assert sum_fp32.shape == sum_ref.shape

    assert np.all(qs_int8 == qs_ref)
    assert np.allclose(scale_fp32, scale_ref, atol=1e-5)
    assert np.allclose(sum_fp32,   sum_ref,   atol=1e-2)

    print("Passed.")


if __name__ == "__main__":
    test_all()

