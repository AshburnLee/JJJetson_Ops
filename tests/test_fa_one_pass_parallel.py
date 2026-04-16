"""测试 fa_kernel_one_pass_parallel（16-block）；并验证 launch_fa 与 parallel 等价。"""
import fa_me
import fa_test_common as fc


def test_fa_one_pass_parallel():
    Q, K, V = fc.random_fa_inputs(24)
    dst_ref = fc.fa_ref(Q, K, V)[0]

    dst = fc.run_launcher(fa_me.launch_fa_one_pass_parallel, Q, K, V, 1.0)
    fc.assert_dst_close("one_pass_parallel", dst, dst_ref)

    dst_alias = fc.run_launcher(fa_me.launch_fa, Q, K, V, 1.0)
    fc.assert_dst_close("launch_fa (alias)", dst_alias, dst_ref)

    print("Passed")


if __name__ == "__main__":
    test_fa_one_pass_parallel()
