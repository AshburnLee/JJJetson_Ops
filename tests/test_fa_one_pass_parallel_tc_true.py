"""fa_kernel_one_pass_parallel_tc_true：多 seed 与 fa_ref_dst_only 数值对齐。"""

import fa_tc_me

import fa_test_common as fc


def test_fa_one_pass_parallel_tc_true():
    for seed in (24, 42, 2477):
        Q, K, V = fc.random_fa_inputs(seed)
        dst_ref = fc.fa_ref_dst_only(Q, K, V)
        dst = fc.run_launcher(fa_tc_me.launch_fa_one_pass_parallel_tc_true, Q, K, V, 1.0)
        fc.assert_dst_close(f"tc_true seed={seed}", dst, dst_ref)
    print("Passed")


if __name__ == "__main__":
    test_fa_one_pass_parallel_tc_true()
