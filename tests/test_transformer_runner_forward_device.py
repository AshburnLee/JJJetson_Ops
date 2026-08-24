"""TransformerRunner 生产路径 forward_device：prefill + decode e2e。"""

import numpy as np
import torch
import transformer_runner_me

import test_transformer_runner as tr
import utils

HIDDEN_SIZE = tr.HIDDEN_SIZE
INTERMEDIATE_SIZE = tr.INTERMEDIATE_SIZE
NUM_Q_HEADS = tr.NUM_Q_HEADS
NUM_KV_HEADS = tr.NUM_KV_HEADS
HEAD_DIM = tr.HEAD_DIM
MAX_SEQ_LEN = tr.MAX_SEQ_LEN
BATCH = tr.BATCH
SEED = tr.SEED
NUM_PREFILL = tr.NUM_TOKENS


def _make_weights():
    np.random.seed(SEED)
    return {
        "w_q": np.random.randn(tr.Q_DIM, HIDDEN_SIZE).astype(np.float32),
        "w_k": np.random.randn(tr.KV_DIM, HIDDEN_SIZE).astype(np.float32),
        "w_v": np.random.randn(tr.KV_DIM, HIDDEN_SIZE).astype(np.float32),
        "w_o": np.random.randn(HIDDEN_SIZE, tr.Q_DIM).astype(np.float32),
        "w_gate": np.random.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32),
        "w_up": np.random.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32),
        "w_down": np.random.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE).astype(np.float32),
        "w_in": np.random.randn(HIDDEN_SIZE).astype(np.float32),
        "w_post": np.random.randn(HIDDEN_SIZE).astype(np.float32),
    }


def _create_runner(weights: dict) -> int:
    return transformer_runner_me.create_runner(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        MAX_SEQ_LEN,
        tr.ROPE_FREQ_BASE,
        weights["w_q"],
        weights["w_k"],
        weights["w_v"],
        weights["w_o"],
        weights["w_gate"],
        weights["w_up"],
        weights["w_down"],
        weights["w_in"],
        weights["w_post"],
    )


def test_forward_device_prefill_matches_ref() -> None:
    weights = _make_weights()
    hidden = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_PREFILL, 1, BATCH).astype(np.float32)
    )
    pos = np.arange(NUM_PREFILL, dtype=np.int32)

    runner = _create_runner(weights)
    try:
        out = np.zeros_like(hidden)
        transformer_runner_me.forward_device(runner, NUM_PREFILL, hidden, out, pos)
        assert transformer_runner_me.kv_cache_len(runner) == NUM_PREFILL
    finally:
        transformer_runner_me.destroy_runner(runner)

    ref = tr.chain_linear_me_ref(
        hidden,
        weights["w_q"],
        weights["w_k"],
        weights["w_v"],
        weights["w_o"],
        weights["w_gate"],
        weights["w_up"],
        weights["w_down"],
        weights["w_in"],
        weights["w_post"],
        pos_offset=0,
    )
    ok = utils.compare_np_torch(out, torch.from_numpy(ref), atol=1e-4, rtol=1e-4)
    assert ok, "forward_device prefill differs from chain ref"
    print("Passed test_forward_device_prefill_matches_ref")


def _weights_dict_to_ref_args(weights: dict) -> tuple:
    return (
        weights["w_q"],
        weights["w_k"],
        weights["w_v"],
        weights["w_o"],
        weights["w_gate"],
        weights["w_up"],
        weights["w_down"],
        weights["w_in"],
        weights["w_post"],
    )


def test_forward_device_decode_matches_kv_ref() -> None:
    """decode 步独立 ref：prefill 后 FA 使用 num_kv_tokens = L + T（非仅 device≈host）。"""
    weights = _make_weights()
    w_args = _weights_dict_to_ref_args(weights)
    hidden_prefill = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_PREFILL, 1, BATCH).astype(np.float32)
    )
    hidden_decode = np.asfortranarray(np.random.randn(HIDDEN_SIZE, 1, 1, BATCH).astype(np.float32))

    kv = tr.KvCacheRef(tr.HEAD_DIM, MAX_SEQ_LEN, tr.NUM_KV_HEADS)
    tr.chain_linear_me_ref_step(hidden_prefill, *w_args, 0, kv)
    assert kv.cache_len == NUM_PREFILL
    ref_decode = tr.chain_linear_me_ref_step(hidden_decode, *w_args, NUM_PREFILL, kv)
    assert kv.cache_len == NUM_PREFILL + 1

    runner = _create_runner(weights)
    try:
        out_prefill = np.zeros_like(hidden_prefill)
        transformer_runner_me.forward_host(runner, NUM_PREFILL, 0, hidden_prefill, out_prefill)
        assert transformer_runner_me.kv_cache_len(runner) == NUM_PREFILL

        out_decode = np.zeros_like(hidden_decode)
        transformer_runner_me.forward_host(runner, 1, NUM_PREFILL, hidden_decode, out_decode)
        assert transformer_runner_me.kv_cache_len(runner) == NUM_PREFILL + 1
    finally:
        transformer_runner_me.destroy_runner(runner)

    ok = utils.compare_np_torch(out_decode, torch.from_numpy(ref_decode), atol=1e-4, rtol=1e-4)
    assert ok, "decode: runner vs KV-cache ref (FA num_kv_tokens=L+T) mismatch"
    print("Passed test_forward_device_decode_matches_kv_ref")


def test_forward_device_prefill_decode() -> None:
    weights = _make_weights()
    hidden_prefill = np.asfortranarray(
        np.random.randn(HIDDEN_SIZE, NUM_PREFILL, 1, BATCH).astype(np.float32)
    )
    hidden_decode = np.asfortranarray(np.random.randn(HIDDEN_SIZE, 1, 1, BATCH).astype(np.float32))

    runner_dev = _create_runner(weights)
    runner_host = _create_runner(weights)
    try:
        pos_prefill = np.arange(NUM_PREFILL, dtype=np.int32)
        out_dev = np.zeros_like(hidden_prefill)
        out_host = np.zeros_like(hidden_prefill)
        transformer_runner_me.forward_device(
            runner_dev, NUM_PREFILL, hidden_prefill, out_dev, pos_prefill
        )
        transformer_runner_me.forward_host(runner_host, NUM_PREFILL, 0, hidden_prefill, out_host)
        assert transformer_runner_me.kv_cache_len(runner_dev) == NUM_PREFILL
        ok = utils.compare_np_torch(out_dev, torch.from_numpy(out_host), atol=1e-4, rtol=1e-4)
        assert ok, "prefill: forward_device vs forward_host mismatch"

        pos_decode = np.array([NUM_PREFILL], dtype=np.int32)
        out_dev_dec = np.zeros_like(hidden_decode)
        out_host_dec = np.zeros_like(hidden_decode)
        transformer_runner_me.forward_device(runner_dev, 1, hidden_decode, out_dev_dec, pos_decode)
        transformer_runner_me.forward_host(runner_host, 1, NUM_PREFILL, hidden_decode, out_host_dec)
        assert transformer_runner_me.kv_cache_len(runner_dev) == NUM_PREFILL + 1
        ok = utils.compare_np_torch(
            out_dev_dec, torch.from_numpy(out_host_dec), atol=1e-4, rtol=1e-4
        )
        assert ok, "decode: forward_device vs forward_host mismatch"
    finally:
        transformer_runner_me.destroy_runner(runner_dev)
        transformer_runner_me.destroy_runner(runner_host)
    print("Passed test_forward_device_prefill_decode")


if __name__ == "__main__":
    test_forward_device_prefill_matches_ref()
    test_forward_device_decode_matches_kv_ref()
    test_forward_device_prefill_decode()
