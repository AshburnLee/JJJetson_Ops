import linear_me
import numpy as np
import torch
import transformer_runner_me

import utils

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 32
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
NUM_TOKENS = 13
BATCH = 1
SEED = 24


def _dims(in_features: int) -> list[int]:
    return [in_features, NUM_TOKENS, 1, BATCH]


def _silu_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    pos = x >= 0
    out = np.empty_like(x, dtype=np.float32)
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    x_neg = x[~pos]
    exp_x = np.exp(x_neg)
    out[~pos] = (x_neg * exp_x) / (1.0 + exp_x)
    return out


def chain_linear_me_ref(
    hidden_np: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    q = np.zeros((Q_DIM, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    k = np.zeros((KV_DIM, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    v = np.zeros((KV_DIM, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    linear_me.linear(hidden_np, w_q, q, _dims(HIDDEN_SIZE), Q_DIM)
    linear_me.linear(hidden_np, w_k, k, _dims(HIDDEN_SIZE), KV_DIM)
    linear_me.linear(hidden_np, w_v, v, _dims(HIDDEN_SIZE), KV_DIM)

    h_mid = np.zeros((HIDDEN_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    linear_me.linear(q, w_o, h_mid, _dims(Q_DIM), HIDDEN_SIZE)

    gate = np.zeros((INTERMEDIATE_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    up = np.zeros((INTERMEDIATE_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    linear_me.linear(h_mid, w_gate, gate, _dims(HIDDEN_SIZE), INTERMEDIATE_SIZE)
    linear_me.linear(h_mid, w_up, up, _dims(HIDDEN_SIZE), INTERMEDIATE_SIZE)

    ffn_mid = (_silu_np(gate) * up).astype(np.float32, order="F")
    h_out = np.zeros((HIDDEN_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    linear_me.linear(ffn_mid, w_down, h_out, _dims(INTERMEDIATE_SIZE), HIDDEN_SIZE)
    return h_out


def torch_layer_linears_ref(
    hidden_np: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    in_features = hidden_np.shape[0]
    num_tokens = hidden_np.shape[1] * hidden_np.shape[2] * hidden_np.shape[3]
    x = torch.from_numpy(hidden_np.reshape(in_features, num_tokens, order="F").T.copy())
    q = torch.nn.functional.linear(x, torch.from_numpy(w_q))
    _k = torch.nn.functional.linear(x, torch.from_numpy(w_k))
    _v = torch.nn.functional.linear(x, torch.from_numpy(w_v))
    h_mid = torch.nn.functional.linear(q, torch.from_numpy(w_o))
    gate = torch.nn.functional.linear(h_mid, torch.from_numpy(w_gate))
    up = torch.nn.functional.linear(h_mid, torch.from_numpy(w_up))
    ffn_mid = torch.nn.functional.silu(gate) * up
    h_out = torch.nn.functional.linear(ffn_mid, torch.from_numpy(w_down))
    return np.asfortranarray(h_out.T.reshape(HIDDEN_SIZE, num_tokens, 1, BATCH).numpy())


def test_transformer_runner():
    np.random.seed(SEED)
    hidden_np = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )
    w_q = np.random.randn(Q_DIM, HIDDEN_SIZE).astype(np.float32)
    w_k = np.random.randn(KV_DIM, HIDDEN_SIZE).astype(np.float32)
    w_v = np.random.randn(KV_DIM, HIDDEN_SIZE).astype(np.float32)
    w_o = np.random.randn(HIDDEN_SIZE, Q_DIM).astype(np.float32)
    w_gate = np.random.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32)
    w_up = np.random.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32)
    w_down = np.random.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE).astype(np.float32)

    # H2D * 7
    runner = transformer_runner_me.create_runner(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        w_q,
        w_k,
        w_v,
        w_o,
        w_gate,
        w_up,
        w_down,
    )  # 返回 handle

    output_me = np.zeros((HIDDEN_SIZE, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    # test 入口 返回 output_me
    # 返回handle 表示同一个 runner
    transformer_runner_me.forward_host(runner, NUM_TOKENS, hidden_np, output_me)
    transformer_runner_me.destroy_runner(runner)  # 返回handle 表示同一个runner

    # trs runner vs linear chain
    ref_np = chain_linear_me_ref(hidden_np, w_q, w_k, w_v, w_o, w_gate, w_up, w_down)
    ok = utils.compare_np_torch(output_me, torch.from_numpy(ref_np), atol=1e-4, rtol=1e-4)
    assert ok, "transformer_runner output differs from chained linear_me"

    # trs runner vs torch implementation
    # cuBLAS 链 vs 纯 torch：7 层 GEMM + SwiGLU 后 abs 误差会放大
    torch_ref_np = torch_layer_linears_ref(hidden_np, w_q, w_k, w_v, w_o, w_gate, w_up, w_down)
    ok_torch = utils.compare_np_torch(
        output_me, torch.from_numpy(torch_ref_np), atol=64.0, rtol=1e-2
    )
    assert ok_torch, "transformer_runner output differs from torch sanity reference"
    print("Passed")


if __name__ == "__main__":
    test_transformer_runner()
