import linear_me
import numpy as np
import torch

import utils

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 32
NUM_TOKENS = 13
BATCH = 1
SEED = 24


def torch_linear_ref(input_np: np.ndarray, weight_np: np.ndarray) -> np.ndarray:
    in_features = input_np.shape[0]
    num_tokens = input_np.shape[1] * input_np.shape[2] * input_np.shape[3]
    out_features = weight_np.shape[0]
    x = torch.from_numpy(input_np.reshape(in_features, num_tokens, order="F").T.copy())
    w = torch.from_numpy(weight_np)
    y = torch.nn.functional.linear(x, w)
    return np.asfortranarray(y.T.reshape(out_features, num_tokens, 1, BATCH).numpy())


def run_linear_case(name: str, in_features: int, out_features: int, input_np: np.ndarray):
    weight_np = np.random.randn(out_features, in_features).astype(np.float32)
    output_np = np.zeros((out_features, NUM_TOKENS, 1, BATCH), dtype=np.float32, order="F")
    dims = [in_features, NUM_TOKENS, 1, BATCH]

    linear_me.linear(input_np, weight_np, output_np, dims, out_features)

    torch_np = torch_linear_ref(input_np, weight_np)

    ok = utils.compare_np_torch(output_np, torch.from_numpy(torch_np), atol=1e-4, rtol=1e-4)
    assert ok, f"{name} linear output differs from reference"
    print(f"{name} passed")


def test_linear():
    np.random.seed(SEED)
    hidden_np = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )
    attn_out_np = np.asfortranarray(
        np.random.randn(NUM_Q_HEADS * HEAD_DIM, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )
    ffn_mid_np = np.asfortranarray(
        np.random.randn(INTERMEDIATE_SIZE, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )

    run_linear_case("q_proj", HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM, hidden_np)
    run_linear_case("k_proj", HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, hidden_np)
    run_linear_case("v_proj", HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, hidden_np)
    run_linear_case("o_proj", NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE, attn_out_np)
    run_linear_case("gate_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE, hidden_np)
    run_linear_case("up_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE, hidden_np)
    run_linear_case("down_proj", INTERMEDIATE_SIZE, HIDDEN_SIZE, ffn_mid_np)
    print("Passed")


if __name__ == "__main__":
    test_linear()
