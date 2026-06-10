import numpy as np
import rope_me

from rope_test_common import (
    check_all_tokens,
    make_rope_inputs,
)


def test_all():
    input_np, pos_np, output_np, dims = make_rope_inputs()
    rope_me.RoPE(input_np, pos_np, output_np, dims)

    all_ok = check_all_tokens(input_np, output_np, pos_np)
    if all_ok:
        print("Passed")
    else:
        print("Failed")
    assert all_ok, "test_all: one or more (head, token) checks failed"


def test_rope_small():
    D, H, T, B = 4, 1, 3, 1

    input_f = np.zeros((D, H, T, B), dtype=np.float32)
    input_f[0, 0, 0, 0] = 1.0  # token0，第一维为1
    input_f[1, 0, 1, 0] = 1.0  # token1，第二维为1
    input_f[2, 0, 2, 0] = 1.0  # token2，第三维为1
    input_np = np.asfortranarray(input_f)

    pos_np = np.arange(T, dtype=np.int32)
    output_np = np.zeros_like(input_np, order="F")

    rope_me.RoPE(input_np, pos_np, output_np, [D, H, T, B])

    print("input (per token):")
    for t in range(T):
        print(f"token {t} input:", input_np[:, 0, t, 0])

    print("\noutput (per token):")
    for t in range(T):
        print(f"token {t} output:", output_np[:, 0, t, 0])


if __name__ == "__main__":
    test_all()
