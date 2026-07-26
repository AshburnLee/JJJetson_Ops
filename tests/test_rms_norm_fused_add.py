import numpy as np
import rms_norm_fused_add_me
import torch

import utils

HIDDEN_SIZE = 128
NUM_TOKENS = 13
BATCH = 1
EPS = 1e-6
SEED = 24


def torch_rms_norm_fused_add_ref(
    input_np: np.ndarray, residual_np: np.ndarray, weight_np: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(input_np.reshape(HIDDEN_SIZE, -1).T.copy())
    residual = torch.from_numpy(residual_np.reshape(HIDDEN_SIZE, -1).T.copy())
    weight = torch.from_numpy(weight_np)
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(residual.dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x * weight
    input_out = np.asfortranarray(x.T.reshape(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).numpy())
    residual_out = np.asfortranarray(residual.T.reshape(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).numpy())
    return input_out, residual_out


def test_rms_norm_fused_add():
    np.random.seed(SEED)
    input_np = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )
    residual_np = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )
    weight_np = np.random.randn(HIDDEN_SIZE).astype(np.float32)
    input_me = np.array(input_np, copy=True, order="F")
    residual_me = np.array(residual_np, copy=True, order="F")

    rms_norm_fused_add_me.forward_device(
        input_me, residual_me, weight_np, HIDDEN_SIZE, NUM_TOKENS, EPS
    )

    ref_input, ref_residual = torch_rms_norm_fused_add_ref(input_np, residual_np, weight_np, EPS)

    ok_input = utils.compare_np_torch(input_me, torch.from_numpy(ref_input), atol=1e-5, rtol=1e-5)
    ok_res = utils.compare_np_torch(
        residual_me, torch.from_numpy(ref_residual), atol=1e-5, rtol=1e-5
    )
    assert ok_input and ok_res, "rms_norm_fused_add output differs from reference"
    print("Passed")


if __name__ == "__main__":
    test_rms_norm_fused_add()
