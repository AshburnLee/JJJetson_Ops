import fa_tc_me

import fa_test_common as fc


def test_fa_double_buffer() -> None:
    seed = 2477
    Q, K, V = fc.random_fa_inputs(seed)
    dst = fc.run_launcher(fa_tc_me.launch_fa_one_pass_parallel_double_buffer, Q, K, V, 1.0)
    dst_ref = fc.fa_ref_dst_only(Q, K, V)
    fc.assert_dst_close(f"double_buffer seed={seed}", dst, dst_ref)
    print("Passed")


if __name__ == "__main__":
    test_fa_double_buffer()
