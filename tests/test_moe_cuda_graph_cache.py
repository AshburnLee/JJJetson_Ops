import moe_cuda_graph_cache_me
import moe_pipeline_sota_me
import numpy as np


def test_moe_cuda_graph_cache():
    rng = np.random.default_rng(7)
    dtype = np.float32

    num_tokens = 5
    top_k = 2
    hidden_size = 4
    intermediate_size = 2
    num_experts = 4

    x = rng.standard_normal((num_tokens, hidden_size)).astype(dtype)
    logits = rng.standard_normal((num_tokens, num_experts)).astype(dtype)
    wg_elems = num_experts * intermediate_size * hidden_size
    wd_elems = num_experts * hidden_size * intermediate_size
    w_gate = rng.standard_normal(wg_elems).astype(dtype)
    w_up = rng.standard_normal(wg_elems).astype(dtype)
    w_down = rng.standard_normal(wd_elems).astype(dtype)

    y_eager = np.zeros((num_tokens, hidden_size), dtype=dtype)
    moe_pipeline_sota_me.moe_pipeline_sota_forward(
        x=x,
        logits=logits,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
        y=y_eager,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    )

    cache = moe_cuda_graph_cache_me.create_cache(hidden_size, intermediate_size, num_experts, top_k)
    try:
        assert not moe_cuda_graph_cache_me.has_graph(cache, num_tokens)
        moe_cuda_graph_cache_me.capture(cache, num_tokens)
        assert moe_cuda_graph_cache_me.has_graph(cache, num_tokens)

        y_graph = np.zeros((num_tokens, hidden_size), dtype=dtype)
        moe_cuda_graph_cache_me.run_graph(
            cache,
            num_tokens,
            x,
            logits,
            w_gate,
            w_up,
            w_down,
            y_graph,
            num_experts,
        )

        assert np.allclose(y_graph, y_eager, rtol=1e-4, atol=5e-3), (
            f"max_abs={np.max(np.abs(y_graph - y_eager))}"
        )

        y_replay = np.zeros((num_tokens, hidden_size), dtype=dtype)
        moe_cuda_graph_cache_me.run_graph(
            cache,
            num_tokens,
            x,
            logits,
            w_gate,
            w_up,
            w_down,
            y_replay,
            num_experts,
        )
        assert np.allclose(y_replay, y_eager, rtol=1e-4, atol=5e-3)
    finally:
        moe_cuda_graph_cache_me.destroy_cache(cache)

    print("Passed")


if __name__ == "__main__":
    test_moe_cuda_graph_cache()
