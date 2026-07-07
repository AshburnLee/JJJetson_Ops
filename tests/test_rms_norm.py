import numpy as np
import rms_norm_me
import torch

import utils

HIDDEN_SIZE = 128
NUM_TOKENS = 13
BATCH = 1
EPS = 1e-6
SEED = 24


def rms_norm_ref(input_np: np.ndarray, weight_np: np.ndarray, epsilon: float) -> np.ndarray:
    hidden_size = input_np.shape[0]
    num_tokens = input_np.shape[1] * input_np.shape[2] * input_np.shape[3]
    out = np.zeros_like(input_np)
    flat_in = input_np.reshape(hidden_size, num_tokens, order="F")
    flat_out = out.reshape(hidden_size, num_tokens, order="F")
    for t in range(num_tokens):
        vec = flat_in[:, t]
        inv_rms = float(1.0 / np.sqrt(np.mean(vec * vec) + epsilon))
        flat_out[:, t] = vec * inv_rms * weight_np
    return out


def test_rms_norm():
    np.random.seed(SEED)
    input_np = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )
    weight_np = np.random.randn(HIDDEN_SIZE).astype(np.float32)
    output_np = np.zeros_like(input_np, order="F")
    dims = [HIDDEN_SIZE, NUM_TOKENS, 1, BATCH]

    rms_norm_me.rms_norm(input_np, weight_np, output_np, dims, EPS)

    ref_np = rms_norm_ref(input_np, weight_np, EPS)

    x = torch.from_numpy(input_np.reshape(HIDDEN_SIZE, -1).T.copy())
    w = torch.from_numpy(weight_np)
    torch_ref = torch.nn.functional.rms_norm(x, (HIDDEN_SIZE,), w, eps=EPS)
    torch_out = np.asfortranarray(torch_ref.T.reshape(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).numpy())

    ok_np = utils.compare_np_torch(output_np, torch.from_numpy(ref_np), atol=1e-5, rtol=1e-5)
    ok_torch = utils.compare_np_torch(output_np, torch.from_numpy(torch_out), atol=1e-5, rtol=1e-5)
    assert ok_np and ok_torch, "rms_norm output differs from reference"
    print("Passed")


if __name__ == "__main__":
    test_rms_norm()
