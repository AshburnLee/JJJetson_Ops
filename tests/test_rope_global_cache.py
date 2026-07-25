import numpy as np
import rope_global_cache_me

from rope_test_common import (
    ROPE_D,
    ROPE_FREQ_BASE,
    ROPE_H,
    ROPE_T,
    check_all_tokens,
    check_single_token,
    make_rope_inputs,
)


def test_prefill():
    max_len = 156
    input_np, pos_np, output_np, _ = make_rope_inputs()
    cache = rope_global_cache_me.create_cossin_cache(max_len, ROPE_D, ROPE_FREQ_BASE)
    try:
        rope_global_cache_me.forward_device(
            cache, input_np, pos_np, output_np, ROPE_D, ROPE_H, ROPE_T, 1
        )
    finally:
        rope_global_cache_me.destroy_cossin_cache(cache)

    assert check_all_tokens(input_np, output_np, pos_np, verbose=False), (
        "rope forward_device prefill differs from NumPy reference"
    )
    print("Prefill (forward_device) passed")


def test_decode():
    max_len = 156
    decode_pos = ROPE_T

    np.random.seed(24)
    input_np = np.asfortranarray(np.random.randn(ROPE_D, ROPE_H, 1, 1).astype(np.float32))
    pos_np = np.array([decode_pos], dtype=np.int32)
    output_np = np.zeros_like(input_np, order="F")
    cache = rope_global_cache_me.create_cossin_cache(max_len, ROPE_D, ROPE_FREQ_BASE)
    try:
        rope_global_cache_me.forward_device(
            cache, input_np, pos_np, output_np, ROPE_D, ROPE_H, 1, 1
        )
    finally:
        rope_global_cache_me.destroy_cossin_cache(cache)

    all_ok = all(
        check_single_token(input_np, output_np, pos_np, h, 0, verbose=False) for h in range(ROPE_H)
    )
    assert all_ok, "rope forward_device decode differs from NumPy reference"
    print("Decode (forward_device) passed")


if __name__ == "__main__":
    test_prefill()
    test_decode()
