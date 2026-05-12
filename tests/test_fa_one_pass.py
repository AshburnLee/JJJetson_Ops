"""测试 fa_kernel_one_pass（单遍 8-block streaming）。"""

import fa_me
import numpy as np

import fa_test_common as fc


def test_fa_one_pass():
    Q, K, V = fc.random_fa_inputs(24)
    dst_ref, m_ref, l_ref, S_ref, row_sum_ref, scale_old_ref, scale_new_ref, exp_val_ref = (
        fc.fa_ref(Q, K, V)
    )
    dst = fc.run_launcher(fa_me.launch_fa_one_pass, Q, K, V, 1.0)
    fc.assert_dst_close("one_pass", dst, dst_ref)

    if fc.debug_ml_enabled(fa_me):
        dst_dbg = fc.empty_dst_f()
        m_out = np.zeros((8, 26), dtype=np.float32)
        l_out = np.zeros((8, 26), dtype=np.float32)
        S_out = np.zeros((8, 8, 26, 32), dtype=np.float32)
        row_sum_out = np.zeros((8, 8, 26), dtype=np.float32)
        scale_old_out = np.zeros((8, 8, 26), dtype=np.float32)
        scale_new_out = np.zeros((8, 8, 26), dtype=np.float32)
        exp_val_out = np.zeros((8, 8, 26, 32), dtype=np.float32)
        fa_me.launch_fa_debug_ml(
            Q,
            K,
            V,
            dst_dbg,
            1.0,
            m_out,
            l_out,
            S_out,
            row_sum_out,
            scale_old_out,
            scale_new_out,
            exp_val_out,
        )
        print("max abs diff m (debug):", np.max(np.abs(m_out - m_ref)))
        print("max abs diff l (debug):", np.max(np.abs(l_out - l_ref)))
        print("max abs diff S_shared (debug):", np.max(np.abs(S_out - S_ref)))
        print("max abs diff row_sum:", np.max(np.abs(row_sum_out - row_sum_ref)))
        print("max abs diff scale_old:", np.max(np.abs(scale_old_out - scale_old_ref)))
        print("max abs diff scale_new:", np.max(np.abs(scale_new_out - scale_new_ref)))
        print("max abs diff exp_val:", np.max(np.abs(exp_val_out - exp_val_ref)))
        fc.assert_dst_close("one_pass_debug_ml", dst_dbg, dst_ref)

    print("Passed")


if __name__ == "__main__":
    test_fa_one_pass()
