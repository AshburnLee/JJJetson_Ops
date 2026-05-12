import moe_dispatch_me
import numpy as np


def expert_offsets_ref(expert_ids: np.ndarray, num_experts: int) -> np.ndarray:
    # 以 case (5, 2, 4, 3) 为例:
    #   num_tokens = 5, top_k = 2, num_experts = 3
    #   expert_ids 的 shape = (10,)
    #   expert_ids 可能为: [0, 0, 2, 1, 0, 1, 2, 2, 1, 0]
    #   bincount 统计每个 expert id 的数量: counts = np.bincount(expert_ids, minlength=3) -> [4, 3, 3]
    #   expert_offsets 是 counts 的前缀和（exclusive scan）:
    #     out[0] = 0,
    #     out[1] = 4,
    #     out[2] = 4+3=7,
    #     out[3] = 4+3+3=10
    #   即 expert_offsets = [0, 4, 7, 10]
    counts = np.bincount(expert_ids, minlength=num_experts)[:num_experts]
    out = np.empty(num_experts + 1, dtype=np.int32)
    out[0] = 0
    np.cumsum(counts, dtype=np.int32, out=out[1:])
    return out


def test_moe_dispatch():
    rng = np.random.default_rng(0)
    dtype = np.float32

    cases = [
        (5, 2, 4, 3),
        # (3, 2, 4, 5),
        # (1, 2, 8, 2),
        # (16, 2, 32, 8),
        # (128, 2, 64, 32),
    ]

    for num_tokens, top_k, hidden_size, num_experts in cases:
        num_routes = num_tokens * top_k
        x = rng.standard_normal((num_tokens, hidden_size)).astype(dtype)
        # expert_ids 的生成方式应该来自 top-k 的结果，这里用随机数模拟这个输入
        expert_ids = rng.integers(0, num_experts, size=num_routes, dtype=np.int32)

        permuted = np.empty((num_routes, hidden_size), dtype=dtype)
        source_token = np.empty(num_routes, dtype=np.int32)
        source_k = np.empty(num_routes, dtype=np.int32)
        expert_offsets = np.empty(num_experts + 1, dtype=np.int32)

        moe_dispatch_me.moe_dispatch(
            x=x,
            expert_ids=expert_ids,
            top_k=top_k,
            num_experts=num_experts,
            permuted_x=permuted,
            source_token=source_token,
            source_k=source_k,
            expert_offsets=expert_offsets,
        )

        off_ref = expert_offsets_ref(expert_ids, num_experts)
        assert np.array_equal(expert_offsets, off_ref), "expert_offsets mismatch"

        pairs_cuda = {(int(source_token[p]), int(source_k[p])) for p in range(num_routes)}
        pairs_expected = {(t, k) for t in range(num_tokens) for k in range(top_k)}
        assert pairs_cuda == pairs_expected, "route coverage (src_t, src_k) mismatch"

        for p in range(num_routes):
            # permuted 是 CUDA API 的输出，代表经过 expert 分发后的 token 排列
            # 这里后续检查 permuted 的每一行是否等于原始 x 中 source_token[p] 行
            t = int(source_token[p])
            assert np.allclose(permuted[p], x[t], rtol=1e-5, atol=1e-5)

    print("Passed")


if __name__ == "__main__":
    test_moe_dispatch()
