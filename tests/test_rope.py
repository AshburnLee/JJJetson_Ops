import numpy as np
import rope_me

def rope_ref_single(vec, pos, base=10000.0):
    """
    vec: shape (d,), d=128
    pos: 标量位置 p
    """
    d = vec.shape[0]
    d_rot = d  # 这里等于 128
    half = d_rot // 2        # 64
    x0 = vec[:half]
    x1 = vec[half:d_rot]

    # k = 0..half-1 对应频率
    k = np.arange(half, dtype=np.float32)
    theta_scale = base ** (-2.0 / d_rot)
    theta = pos * (theta_scale ** k)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    y0 = x0 * cos_theta - x1 * sin_theta
    y1 = x0 * sin_theta + x1 * cos_theta

    out = vec.copy()
    out[:half]  = y0
    out[half:d_rot] = y1
    return out

def test_token(input_np, output_np, pos_np, head, token):
    p = pos_np[token]
    # 取出这一条的 128 维向量（注意 col-major 的取法）
    vec_in  = input_np[:, head, token, 0].copy()   # shape (128,)
    vec_out = output_np[:, head, token, 0].copy()
    ref = rope_ref_single(vec_in, p)
    max_error = np.max(np.abs(vec_out - ref))
    print("max abs diff:", max_error)
    if max_error > 1e-5:
        print(f"token {token} input: \n", vec_in)
        print(f"token {token} output \n:", vec_out)


def test_all():
    # 物理上按列主序 (Fortran order) 存储，shape = (128, 16, 13, 1)
    D, H, T, B = 128, 16, 13, 1
    np.random.seed(24)
    input_f = np.random.randn(D, H, T, B).astype(np.float32)
    input_np = np.asfortranarray(input_f)

    n_elem = input_np.size
    pos_np = np.arange(T, dtype=np.int32)
    output_np = np.zeros_like(input_np, order="F")

    rope_me.RoPE(input_np, pos_np, output_np, [D, H, T, B])

    for _h, _t in zip(range(H), range(T)):
        test_token(input_np,output_np, pos_np, _h,_t)


def test_rope_small():
    # head_dim=4, n_heads=1, seq_len=3, batch=1
    D, H, T, B = 4, 1, 3, 1

    # 构造列主序输入，形状 [D, H, T, B]
    # token0: [1, 0, 0, 0]
    # token1: [0, 1, 0, 0]
    # token2: [0, 0, 1, 0]
    input_f = np.zeros((D, H, T, B), dtype=np.float32)
    input_f[0, 0, 0, 0] = 1.0  # token0，第一维为1
    input_f[1, 0, 1, 0] = 1.0  # token1，第二维为1
    input_f[2, 0, 2, 0] = 1.0  # token2，第三维为1
    input_np = np.asfortranarray(input_f)

    pos_np = np.arange(T, dtype=np.int32)   # [0, 1, 2]

    output_np = np.zeros_like(input_np, order="F")

    # dims 按 col-major 物理顺序 [D, H, T, B]
    rope_me.RoPE(input_np, pos_np, output_np, [D, H, T, B])

    # 把每个 token 的向量打印出来，方便观察旋转前后差异
    print("input (per token):")
    for t in range(T):
        v = input_np[:, 0, t, 0]
        print(f"token {t} input:", v)

    print("\noutput (per token):")
    for t in range(T):
        v = output_np[:, 0, t, 0]
        print(f"token {t} output:", v)

if __name__ == "__main__":
    test_all()
    # test_rope_small()

