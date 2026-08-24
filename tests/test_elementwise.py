import elementwise_me
import numpy as np
import torch

import utils

HIDDEN_SIZE = 128
NUM_TOKENS = 13
BATCH = 1
SEED = 24

OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}


def run_case(op_name: str, a_np: np.ndarray, b_np: np.ndarray):
    out_np = np.zeros_like(a_np, order="F")
    elementwise_me.forward_host(op_name, a_np, b_np, out_np)

    ref_np = OPS[op_name](a_np, b_np)
    ok = utils.compare_np_torch(out_np, torch.from_numpy(ref_np), atol=1e-6, rtol=1e-6)
    assert ok, f"elementwise {op_name} differs from reference"
    print(f"Passed elementwise_{op_name}")


def test_elementwise_flat():
    np.random.seed(SEED)
    n_elem = HIDDEN_SIZE * NUM_TOKENS * BATCH
    a_np = np.asfortranarray(np.random.randn(n_elem).astype(np.float32))
    b_np = np.asfortranarray(np.random.randn(n_elem).astype(np.float32))
    b_np[np.abs(b_np) < 1e-3] = 1.0

    for op_name in OPS:
        run_case(op_name, a_np, b_np)
    print("Passed test_elementwise_flat")


def test_residual_add_hidden_layout():
    np.random.seed(SEED)
    residual_np = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )
    subblock_np = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_TOKENS, 1, BATCH).astype(np.float32)
    )
    out_np = np.zeros_like(residual_np, order="F")

    elementwise_me.forward_host("add", residual_np, subblock_np, out_np)

    ref_np = residual_np + subblock_np
    ok = utils.compare_np_torch(out_np, torch.from_numpy(ref_np), atol=1e-6, rtol=1e-6)
    assert ok, "residual add (hidden layout) differs from reference"
    print("Passed test_residual_add_hidden_layout")


if __name__ == "__main__":
    test_elementwise_flat()
    test_residual_add_hidden_layout()
