"""测试 fa_kernel_one_pass_parallel_tc（WMMA QK）与 ref dst 对齐。"""

import fa_tc_me

import fa_test_common as fc


def test_fa_one_pass_parallel_tc():
    Q, K, V = fc.random_fa_inputs(24)
    dst_ref = fc.fa_ref_dst_only(Q, K, V)

    dst = fc.run_launcher(fa_tc_me.launch_fa_one_pass_parallel_tc, Q, K, V, 1.0)
    fc.assert_dst_close("one_pass_parallel_tc", dst, dst_ref)

    print("Passed")


if __name__ == "__main__":
    test_fa_one_pass_parallel_tc()
