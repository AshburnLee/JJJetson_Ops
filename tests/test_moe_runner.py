import moe_pipeline_sota_me
import moe_runner_me
import numpy as np


def test_moe_runner():
    rng = np.random.default_rng(11)
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

    y_golden = np.zeros((num_tokens, hidden_size), dtype=dtype)
    moe_pipeline_sota_me.moe_pipeline_sota_forward(
        x=x,
        logits=logits,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
        y=y_golden,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    )

    runner = moe_runner_me.create_runner(
        hidden_size, intermediate_size, num_experts, top_k, enable_graph=True
    )
    try:
        y_prefill = np.zeros((num_tokens, hidden_size), dtype=dtype)
        used_graph = moe_runner_me.forward_host(
            runner,
            num_tokens,
            is_decode=False,
            x=x,
            logits=logits,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            y=y_prefill,
        )
        assert not used_graph
        assert np.allclose(y_prefill, y_golden, rtol=1e-4, atol=5e-3)

        y_decode = np.zeros((num_tokens, hidden_size), dtype=dtype)
        used_graph = moe_runner_me.forward_host(
            runner,
            num_tokens,
            is_decode=True,
            x=x,
            logits=logits,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            y=y_decode,
        )
        assert used_graph
        assert moe_runner_me.has_graph(runner, num_tokens)
        assert np.allclose(y_decode, y_golden, rtol=1e-4, atol=5e-3)

        moe_runner_me.set_dispatch(runner, moe_runner_me.Dispatch.EAGER)
        y_eager = np.zeros((num_tokens, hidden_size), dtype=dtype)
        used_graph = moe_runner_me.forward_host(
            runner,
            num_tokens,
            is_decode=True,
            x=x,
            logits=logits,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            y=y_eager,
        )
        assert not used_graph
        assert np.allclose(y_eager, y_golden, rtol=1e-4, atol=5e-3)

        moe_runner_me.set_dispatch(runner, moe_runner_me.Dispatch.GRAPH)
        y_graph = np.zeros((num_tokens, hidden_size), dtype=dtype)
        used_graph = moe_runner_me.forward_host(
            runner,
            num_tokens,
            is_decode=False,
            x=x,
            logits=logits,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            y=y_graph,
        )
        assert used_graph
        assert np.allclose(y_graph, y_golden, rtol=1e-4, atol=5e-3)
    finally:
        moe_runner_me.destroy_runner(runner)

    print("Passed")


if __name__ == "__main__":
    test_moe_runner()
