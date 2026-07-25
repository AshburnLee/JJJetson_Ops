"""测试 fa_kernel_one_pass_parallel（16-block，优化中间步骤）；生产路径见 launch_fa / forward_device。"""

import fa_me

import fa_test_common as fc


def test_fa_one_pass_parallel():
    Q, K, V = fc.random_fa_inputs(24)
    dst_ref = fc.fa_ref(Q, K, V)[0]

    # 中间优化实验 kernel
    dst = fc.run_launcher(fa_me.launch_fa_one_pass_parallel, Q, K, V, 1.0)
    fc.assert_dst_close("one_pass_parallel (experimental)", dst, dst_ref)

    # 生产路径：fa_double_buffer
    dst_prod = fc.run_launcher(fa_me.launch_fa, Q, K, V, 1.0)
    fc.assert_dst_close("launch_fa (double_buffer)", dst_prod, dst_ref)

    dst_device = fc.run_launcher(fa_me.forward_device, Q, K, V, 1.0)
    fc.assert_dst_close("forward_device (double_buffer)", dst_device, dst_ref)

    print("Passed")


if __name__ == "__main__":
    test_fa_one_pass_parallel()
