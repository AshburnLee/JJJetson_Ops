"""fa_kernel_one_pass_parallel_tc：多 seed 与 fa_ref_dst_only 数值对齐。"""
from __future__ import annotations

import fa_tc_me
import fa_test_common as fc


def test_fa_tc_wmma_multi_seed() -> None:
    for seed in (24, 44, 77):
        Q, K, V = fc.random_fa_inputs(seed)
        dst = fc.run_launcher(fa_tc_me.launch_fa_one_pass_parallel_tc_true, Q, K, V, 1.0)
        dst_ref = fc.fa_ref_dst_only(Q, K, V)
        fc.assert_dst_close(f"paralleltc_true seed={seed}", dst, dst_ref)
    print("Passed")


if __name__ == "__main__":
    test_fa_tc_wmma_multi_seed()
