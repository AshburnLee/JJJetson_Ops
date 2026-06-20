import moe_dispatch_sota_me
import moe_pipeline_sota_me
import moe_top_k_me
import numpy as np


def silu_f(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def combine_ref(expert_out, source_token, source_k, route_weights, num_tokens, top_k):
    num_routes, hidden_size = expert_out.shape
    y = np.zeros((num_tokens, hidden_size), dtype=np.float64)
    for pos in range(num_routes):
        t = int(source_token[pos])
        k = int(source_k[pos])
        y[t] += route_weights[t, k] * expert_out[pos].astype(np.float64)
    return y.astype(np.float32)


def swiglu_experts_ref(permuted_x, expert_offsets, w_gate, w_up, w_down):
    num_routes, hidden_size = permuted_x.shape
    num_experts = expert_offsets.shape[0] - 1
    expert_out = np.empty((num_routes, hidden_size), dtype=np.float32)
    for e in range(num_experts):
        start = int(expert_offsets[e])
        end = int(expert_offsets[e + 1])
        if end <= start:
            continue
        xe = permuted_x[start:end]
        gate = xe @ w_gate[e].T
        up = xe @ w_up[e].T
        mid = silu_f(gate) * up
        expert_out[start:end] = mid @ w_down[e].T
    return expert_out


def pipeline_sota_ref(
    x, logits, w_gate_flat, w_up_flat, w_down_flat, num_experts, top_k, intermediate_size
):
    num_tokens, H = x.shape
    num_routes = num_tokens * top_k

    route_weights = np.zeros((num_tokens, top_k), dtype=np.float32)
    expert_ids_flat = np.zeros(num_routes, dtype=np.int32)
    moe_top_k_me.moe_top_k(logits, top_k, route_weights, expert_ids_flat, [num_tokens, num_experts])

    permuted = np.empty((num_routes, H), dtype=np.float32)
    source_token = np.empty(num_routes, dtype=np.int32)
    source_k = np.empty(num_routes, dtype=np.int32)
    expert_offsets = np.empty(num_experts + 1, dtype=np.int32)
    moe_dispatch_sota_me.moe_dispatch_sota(
        x=x,
        expert_ids=expert_ids_flat,
        top_k=top_k,
        num_experts=num_experts,
        permuted_x=permuted,
        source_token=source_token,
        source_k=source_k,
        expert_offsets=expert_offsets,
    )

    w_gate = w_gate_flat.reshape(num_experts, intermediate_size, H)
    w_up = w_up_flat.reshape(num_experts, intermediate_size, H)
    w_down = w_down_flat.reshape(num_experts, H, intermediate_size)
    expert_out = swiglu_experts_ref(permuted, expert_offsets, w_gate, w_up, w_down)
    return combine_ref(expert_out, source_token, source_k, route_weights, num_tokens, top_k)


def test_moe_pipeline_sota():
    rng = np.random.default_rng(42)
    dtype = np.float32
    cases = [(5, 2, 4, 2, 4)]

    for num_tokens, top_k, hidden_size, intermediate_size, num_experts in cases:
        logits = rng.standard_normal((num_tokens, num_experts)).astype(dtype)
        x = rng.standard_normal((num_tokens, hidden_size)).astype(dtype)
        wg_elems = num_experts * intermediate_size * hidden_size
        wd_elems = num_experts * hidden_size * intermediate_size
        w_gate = rng.standard_normal(wg_elems).astype(dtype)
        w_up = rng.standard_normal(wg_elems).astype(dtype)
        w_down = rng.standard_normal(wd_elems).astype(dtype)

        y_gpu = np.zeros((num_tokens, hidden_size), dtype=dtype)
        moe_pipeline_sota_me.moe_pipeline_sota_forward(
            x=x,
            logits=logits,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            y=y_gpu,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        )

        y_ref = pipeline_sota_ref(
            x, logits, w_gate, w_up, w_down, num_experts, top_k, intermediate_size
        )
        assert np.allclose(y_gpu, y_ref, rtol=1e-4, atol=5e-3), (
            f"max_abs={np.max(np.abs(y_gpu - y_ref))}"
        )

    print("Passed")


if __name__ == "__main__":
    test_moe_pipeline_sota()
