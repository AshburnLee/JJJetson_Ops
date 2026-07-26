import numpy as np
import rms_norm_me
import torch

import utils

HIDDEN_SIZE = 128
NUM_TOKENS = 13
BATCH = 1
EPS = 1e-6
SEED = 24


def test_rms_norm():
    np.random.seed(SEED)
    input_np = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )
    weight_np = np.random.randn(HIDDEN_SIZE).astype(np.float32)
    output_np = np.zeros_like(input_np, order="F")

    rms_norm_me.forward_device(input_np, weight_np, output_np, HIDDEN_SIZE, NUM_TOKENS, EPS)

    x = torch.from_numpy(input_np.reshape(HIDDEN_SIZE, -1).T.copy())
    w = torch.from_numpy(weight_np)
    torch_ref = torch.nn.functional.rms_norm(x, (HIDDEN_SIZE,), w, eps=EPS)
    ref_np = np.asfortranarray(torch_ref.T.reshape(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).numpy())

    ok = utils.compare_np_torch(output_np, torch.from_numpy(ref_np), atol=1e-5, rtol=1e-5)
    assert ok, "rms_norm output differs from reference"
    print("Passed")


if __name__ == "__main__":
    test_rms_norm()
