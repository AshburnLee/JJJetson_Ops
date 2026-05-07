import os
import sys

import numpy as np
import moe_dispatch_me
import moe_pipeline_me


def silu_f(x: np.ndarray) -> np.ndarray:
    """silu_f"""
    return x * (1.0 / (1.0 + np.exp(-x)))


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
        y[t] += float(route_weights[t, k]) * expert_out[pos].astype(np.float64)
    return y.astype(np.float32)


def swiglu_experts_ref(
    permuted_x: np.ndarray,
    expert_offsets: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    """
    Same GEMM + silu multiply + down projection as the CUDA path, on permuted rows
    Weight blocks: WG/WU (I, H), WD (H, I) per expert, row-major
    """
    num_routes, hidden_size = permuted_x.shape
    num_experts = expert_offsets.shape[0] - 1
    intermediate_size = w_gate.shape[1]

    expert_out = np.empty((num_routes, hidden_size), dtype=np.float32)

    for e in range(num_experts):
        start = int(expert_offsets[e])
        end = int(expert_offsets[e + 1])
        n = end - start
        if n <= 0:
            continue
        xe = permuted_x[start:end]
        wg = w_gate[e]
        wu = w_up[e]
        wd = w_down[e]
        gate = xe @ wg.T
        up = xe @ wu.T
        mid = silu_f(gate) * up
        out = mid @ wd.T
        expert_out[start:end] = out

    return expert_out


def pipeline_ref_from_gpu_dispatch(
    x: np.ndarray,
    expert_ids: np.ndarray,
    route_weights: np.ndarray,
    w_gate_flat: np.ndarray,
    w_up_flat: np.ndarray,
    w_down_flat: np.ndarray,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
) -> np.ndarray:
    num_tokens, hidden_size = x.shape
    num_routes = num_tokens * top_k

    permuted = np.empty((num_routes, hidden_size), dtype=np.float32)
    source_token = np.empty(num_routes, dtype=np.int32)
    source_k = np.empty(num_routes, dtype=np.int32)
    expert_offsets = np.empty(num_experts + 1, dtype=np.int32)
    # 复用了本repo的 Dispatch， 前提是 moe_dispatch_me 已经通过 correnctness 测试 
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

    w_gate = w_gate_flat.reshape(num_experts, intermediate_size, hidden_size)
    w_up = w_up_flat.reshape(num_experts, intermediate_size, hidden_size)
    w_down = w_down_flat.reshape(num_experts, hidden_size, intermediate_size)

    eo = swiglu_experts_ref(permuted, expert_offsets, w_gate, w_up, w_down)
    return combine_ref(eo, source_token, source_k, route_weights, num_tokens, top_k)


def test_moe_pipeline():
    rng = np.random.default_rng(42)
    dtype = np.float32

    cases = [
        (5, 2, 4, 2, 3),
        # (5, 2, 4, 8, 3),
        # (12, 2, 16, 32, 5),
    ]

    for num_tokens, top_k, hidden_size, intermediate_size, num_experts in cases:
        num_routes = num_tokens * top_k
        x = rng.standard_normal((num_tokens, hidden_size)).astype(dtype)
        expert_ids = rng.integers(0, num_experts, size=num_routes, dtype=np.int32)
        route_weights = np.abs(rng.standard_normal((num_tokens, top_k))).astype(dtype)

        wg_elems = num_experts * intermediate_size * hidden_size
        wd_elems = num_experts * hidden_size * intermediate_size
        w_gate = rng.standard_normal(wg_elems).astype(dtype)
        w_up = rng.standard_normal(wg_elems).astype(dtype)
        w_down = rng.standard_normal(wd_elems).astype(dtype)

        y_gpu = np.zeros((num_tokens, hidden_size), dtype=dtype)
        moe_pipeline_me.moe_swiglu_experts_forward(
            x=x,
            expert_ids=expert_ids,
            route_weights=route_weights,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            y=y_gpu,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        )

        y_ref = pipeline_ref_from_gpu_dispatch(
            x,
            expert_ids,
            route_weights,
            w_gate,
            w_up,
            w_down,
            num_experts,
            top_k,
            intermediate_size,
        )

        assert np.allclose(y_gpu, y_ref, rtol=1e-4, atol=5e-3), (
            f"max_abs={np.max(np.abs(y_gpu - y_ref))} case="
            f"({num_tokens}, {top_k}, {hidden_size}, {intermediate_size}, {num_experts})"
        )

    print("Passed")


if __name__ == "__main__":
    test_moe_pipeline()
