"""泛化 shape 的 fa_double_buffer 测试：head_dim=32（对齐 TransformerRunner）与 legacy 128。"""

import fa_me

import fa_test_common as fc


def test_fa_double_buffer_head32() -> None:
    # 与 test_transformer_runner 一致的 GQA 2:1
    head_dim = 32
    tok_q = 13
    q_heads = 4
    kv_heads = 2
    for tok_kv in (13, 37, 256):
        seed = 100 + tok_kv
        Q, K, V = fc.random_fa_inputs_for_shape(head_dim, tok_q, q_heads, tok_kv, kv_heads, seed)
        scale = 1.0 / (head_dim**0.5)
        dst = fc.run_launcher(fa_me.forward_host_shape, Q, K, V, scale)
        dst_ref = fc.fa_ref_dst_only(Q, K, V, scale)
        fc.assert_dst_close(f"head32 tok_kv={tok_kv}", dst, dst_ref)
    print("Passed test_fa_double_buffer_head32")


def test_fa_double_buffer_legacy128() -> None:
    Q, K, V = fc.random_fa_inputs(24)
    dst = fc.run_launcher(fa_me.forward_host_shape, Q, K, V, 1.0)
    dst_ref = fc.fa_ref_dst_only(Q, K, V)
    fc.assert_dst_close("legacy128 forward_host_shape", dst, dst_ref)
    print("Passed test_fa_double_buffer_legacy128")


if __name__ == "__main__":
    test_fa_double_buffer_head32()
    test_fa_double_buffer_legacy128()
