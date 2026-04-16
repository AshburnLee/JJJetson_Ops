"""测试 fa_kernel_two_pass（两遍 KV）。"""
import fa_me
import fa_test_common as fc


def test_fa_two_pass():
    Q, K, V = fc.random_fa_inputs(24)
    dst_ref = fc.fa_ref(Q, K, V)[0]
    dst = fc.run_launcher(fa_me.launch_fa_two_pass, Q, K, V, 1.0)
    fc.assert_dst_close("two_pass", dst, dst_ref)
    print("Passed")


if __name__ == "__main__":
    test_fa_two_pass()
