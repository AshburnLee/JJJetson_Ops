import moe_combine_sota_me
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
    y = np.zeros((num_tokens, hidden_size), dtype=np.float64)
    for pos in range(num_routes):
        t = int(source_token[pos])
        k = int(source_k[pos])
        y[t] += route_weights[t, k] * expert_out[pos].astype(np.float64)
    return y.astype(np.float32)


def test_moe_combine_sota():
    rng = np.random.default_rng(1)
    dtype = np.float32

    cases = [
        (5, 2, 4),
    ]

    for num_tokens, top_k, hidden_size in cases:
        num_routes = num_tokens * top_k
        expert_out = rng.standard_normal((num_routes, hidden_size)).astype(dtype)
        source_token = np.repeat(np.arange(num_tokens, dtype=np.int32), top_k)
        source_k = np.tile(np.arange(top_k, dtype=np.int32), num_tokens)
        route_weights = np.abs(rng.standard_normal((num_tokens, top_k))).astype(dtype)

        y = np.zeros((num_tokens, hidden_size), dtype=dtype)
        y_ref = combine_ref(expert_out, source_token, source_k, route_weights, num_tokens, top_k)

        moe_combine_sota_me.moe_combine_sota(
            expert_out=expert_out,
            source_token=source_token,
            source_k=source_k,
            route_weights=route_weights,
            y=y,
            num_tokens=num_tokens,
            top_k=top_k,
        )

        assert np.allclose(y, y_ref, rtol=1e-5, atol=1e-4)

    print("Passed")


if __name__ == "__main__":
    test_moe_combine_sota()
