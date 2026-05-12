import moe_combine_me
import numpy as np


def combine_ref(
    expert_out: np.ndarray,
    source_token: np.ndarray,
    source_k: np.ndarray,
    route_weights: np.ndarray,
    num_tokens: int,
    top_k: int,
) -> np.ndarray:
    num_routes, hidden_size = expert_out.shape
    y = np.zeros((num_tokens, hidden_size), dtype=np.float32)
    for r in range(num_routes):
        t = int(source_token[r])
        k = int(source_k[r])
        y[t] += route_weights[t, k] * expert_out[r].astype(np.float64)
    return y.astype(np.float32)


def test_moe_combine():
    rng = np.random.default_rng(1)
    dtype = np.float32

    cases = [
        (5, 2, 4),
        # (1, 2, 8),
        # (32, 2, 16),
        # (64, 2, 128),
        # (128, 2, 32),
    ]

    for num_tokens, top_k, hidden_size in cases:
        num_routes = num_tokens * top_k
        expert_out = rng.standard_normal((num_routes, hidden_size)).astype(dtype)
        source_token = np.repeat(np.arange(num_tokens, dtype=np.int32), top_k)
        source_k = np.tile(np.arange(top_k, dtype=np.int32), num_tokens)
        route_weights = np.abs(rng.standard_normal((num_tokens, top_k))).astype(dtype)

        y = np.zeros((num_tokens, hidden_size), dtype=dtype)
        y_ref = combine_ref(expert_out, source_token, source_k, route_weights, num_tokens, top_k)

        moe_combine_me.moe_combine(
            expert_out=expert_out,
            source_token=source_token,
            source_k=source_k,
            route_weights=route_weights,
            y=y,
            num_tokens=num_tokens,
            top_k=top_k,
        )

        # 打印 pos, src_t, x 和 y & y_ref 的对应数值，数table展示
        print(
            f"\n{'pos':>4}  {'src_t':>5}  {'src_k':>5}  {'expert_out':>12}  {'route_weight':>13}  {'y':>12}  {'y_ref':>12}"
        )
        for t in range(num_tokens):
            for k in range(top_k):
                pos = t * top_k + k
                eo = np.array2string(expert_out[pos], precision=3, separator=", ")
                y_row = np.array2string(y[t], precision=3, separator=", ")
                y_ref_row = np.array2string(y_ref[t], precision=3, separator=", ")
                print(
                    f"{pos:4d}  {source_token[pos]:5d}  {source_k[pos]:5d}  {eo:>12}  {route_weights[t, k]:13.6f}  {y_row:>12}  {y_ref_row:>12}"
                )

        assert np.allclose(y, y_ref, rtol=1e-5, atol=1e-4)

    print("Passed")


if __name__ == "__main__":
    test_moe_combine()
