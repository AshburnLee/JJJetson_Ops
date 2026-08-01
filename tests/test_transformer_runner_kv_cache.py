"""TransformerRunner KV cache：prefill + decode 后 cache_len 递增。"""

import numpy as np
import transformer_runner_me

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 32
MAX_SEQ_LEN = 256
BATCH = 1
SEED = 24


def _make_runner():
    np.random.seed(SEED)
    w_q = np.random.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE).astype(np.float32)
    w_k = np.random.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE).astype(np.float32)
    w_v = np.random.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE).astype(np.float32)
    w_o = np.random.randn(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM).astype(np.float32)
    w_gate = np.random.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32)
    w_up = np.random.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32)
    w_down = np.random.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE).astype(np.float32)
    w_in = np.random.randn(HIDDEN_SIZE).astype(np.float32)
    w_post = np.random.randn(HIDDEN_SIZE).astype(np.float32)
    return transformer_runner_me.create_runner(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        MAX_SEQ_LEN,
        10000.0,
        w_q,
        w_k,
        w_v,
        w_o,
        w_gate,
        w_up,
        w_down,
        w_in,
        w_post,
    )


def test_transformer_runner_kv_cache_prefill_decode() -> None:
    runner = _make_runner()
    try:
        assert transformer_runner_me.kv_cache_len(runner) == 0

        num_prefill = 13
        hidden = np.asfortranarray(
            np.random.randn(HIDDEN_SIZE, num_prefill, 1, BATCH).astype(np.float32)
        )
        out = np.zeros_like(hidden)
        transformer_runner_me.forward_host(runner, num_prefill, 0, hidden, out)
        assert transformer_runner_me.kv_cache_len(runner) == num_prefill

        hidden_dec = np.asfortranarray(np.random.randn(HIDDEN_SIZE, 1, 1, BATCH).astype(np.float32))
        out_dec = np.zeros_like(hidden_dec)
        transformer_runner_me.forward_host(runner, 1, num_prefill, hidden_dec, out_dec)
        assert transformer_runner_me.kv_cache_len(runner) == num_prefill + 1
    finally:
        transformer_runner_me.destroy_runner(runner)
    print("Passed")


def test_transformer_runner_kv_cache_reset() -> None:
    runner = _make_runner()
    try:
        num_prefill = 13
        hidden = np.asfortranarray(
            np.random.randn(HIDDEN_SIZE, num_prefill, 1, BATCH).astype(np.float32)
        )
        out = np.zeros_like(hidden)
        transformer_runner_me.forward_host(runner, num_prefill, 0, hidden, out)
        assert transformer_runner_me.kv_cache_len(runner) == num_prefill

        transformer_runner_me.kv_cache_reset(runner)
        assert transformer_runner_me.kv_cache_len(runner) == 0

        out2 = np.zeros_like(hidden)
        transformer_runner_me.forward_host(runner, num_prefill, 0, hidden, out2)
        assert transformer_runner_me.kv_cache_len(runner) == num_prefill
    finally:
        transformer_runner_me.destroy_runner(runner)
    print("reset Passed")


if __name__ == "__main__":
    test_transformer_runner_kv_cache_prefill_decode()
    test_transformer_runner_kv_cache_reset()
