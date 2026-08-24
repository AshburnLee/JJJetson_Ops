"""InferenceEngine token forward: forward_token_host prefill/decode vs ref."""

import inference_engine_me
import numpy as np

import test_inference_engine_forward as tief
import test_transformer_model_embed_lm_head as tme
import test_transformer_runner as tr

HIDDEN_SIZE = tief.HIDDEN_SIZE
VOCAB_SIZE = tief.VOCAB_SIZE
MAX_SEQ_LEN = tief.MAX_SEQ_LEN
HEAD_DIM = tief.HEAD_DIM
NUM_KV_HEADS = tief.NUM_KV_HEADS
NUM_PREFILL = tief.NUM_PREFILL
SEED = 44


def _logits_ref(
    token_ids: np.ndarray,
    tensors: dict[str, np.ndarray],
    num_layers: int,
    pos_offset: int,
    kv_caches: list[tr.KvCacheRef],
) -> np.ndarray:
    num_tokens = int(token_ids.shape[0])
    hidden = tme._embed_ref(tensors["embed"], token_ids, num_tokens)
    hidden_4d = np.asfortranarray(hidden.reshape(HIDDEN_SIZE, num_tokens, 1, 1))
    normed = tief._chain_layers_ref(hidden_4d, tensors, num_layers, pos_offset, kv_caches)
    h2 = normed.reshape(HIDDEN_SIZE, num_tokens, order="F")
    return tme._lm_head_ref(h2, tensors["lm_head"], tensors["embed"], False, num_tokens)


def test_engine_forward_token_one_layer_prefill() -> None:
    np.random.seed(SEED)
    model, engine, fixture_dir, tensors = tief._setup_model_engine(1)
    try:
        # 随机方式生成 TOKEN id
        token_ids = np.random.randint(0, VOCAB_SIZE, size=NUM_PREFILL, dtype=np.int32)
        logits_out = np.zeros((VOCAB_SIZE, NUM_PREFILL), dtype=np.float32, order="F")
        inference_engine_me.forward_token_host(engine, NUM_PREFILL, 0, token_ids, logits_out)
        assert inference_engine_me.kv_cache_len(engine) == NUM_PREFILL
        assert inference_engine_me.next_pos(engine) == NUM_PREFILL

        kv = [tr.KvCacheRef(HEAD_DIM, MAX_SEQ_LEN, NUM_KV_HEADS)]
        ref = _logits_ref(token_ids, tensors, 1, 0, kv)
        max_abs = np.max(np.abs(logits_out - ref))
        assert np.allclose(logits_out, ref, atol=1e-4, rtol=1e-4), f"max_abs_diff={max_abs:e}"
        print("Passed test_engine_forward_token_one_layer_prefill")
    finally:
        tief._cleanup(model, engine, fixture_dir)


def test_engine_forward_token_two_layer_prefill() -> None:
    np.random.seed(SEED + 1)
    model, engine, fixture_dir, tensors = tief._setup_model_engine(2)
    try:
        # 随机方式生成 TOKEN id
        token_ids = np.random.randint(0, VOCAB_SIZE, size=NUM_PREFILL, dtype=np.int32)
        logits_out = np.zeros((VOCAB_SIZE, NUM_PREFILL), dtype=np.float32, order="F")
        inference_engine_me.forward_token_host(engine, NUM_PREFILL, 0, token_ids, logits_out)
        assert inference_engine_me.kv_cache_num_layers(engine) == 2
        assert inference_engine_me.kv_cache_len(engine) == NUM_PREFILL

        kv = [tr.KvCacheRef(HEAD_DIM, MAX_SEQ_LEN, NUM_KV_HEADS) for _ in range(2)]
        ref = _logits_ref(token_ids, tensors, 2, 0, kv)
        max_abs = np.max(np.abs(logits_out - ref))
        assert np.allclose(logits_out, ref, atol=1e-4, rtol=1e-4), f"max_abs_diff={max_abs:e}"
        print("Passed test_engine_forward_token_two_layer_prefill")
    finally:
        tief._cleanup(model, engine, fixture_dir)


def test_engine_forward_token_prefill_decode_and_reset() -> None:
    np.random.seed(SEED + 2)
    model, engine, fixture_dir, tensors = tief._setup_model_engine(1)
    try:
        # 随机方式生成 TOKEN id (prompt)
        prompt = np.random.randint(0, VOCAB_SIZE, size=NUM_PREFILL, dtype=np.int32)
        logits_p = np.zeros((VOCAB_SIZE, NUM_PREFILL), dtype=np.float32, order="F")
        inference_engine_me.forward_token_host(engine, NUM_PREFILL, 0, prompt, logits_p)
        assert inference_engine_me.kv_cache_len(engine) == NUM_PREFILL

        kv = [tr.KvCacheRef(HEAD_DIM, MAX_SEQ_LEN, NUM_KV_HEADS)]
        ref_p = _logits_ref(prompt, tensors, 1, 0, kv)
        assert np.allclose(logits_p, ref_p, atol=1e-4, rtol=1e-4)

        decode_id = np.array([17], dtype=np.int32)
        logits_d = np.zeros((VOCAB_SIZE, 1), dtype=np.float32, order="F")
        inference_engine_me.forward_token_host(engine, 1, NUM_PREFILL, decode_id, logits_d)
        assert inference_engine_me.kv_cache_len(engine) == NUM_PREFILL + 1
        assert inference_engine_me.next_pos(engine) == NUM_PREFILL + 1

        ref_d = _logits_ref(decode_id, tensors, 1, NUM_PREFILL, kv)
        max_abs = np.max(np.abs(logits_d - ref_d))
        assert np.allclose(logits_d, ref_d, atol=1e-4, rtol=1e-4), f"max_abs_diff={max_abs:e}"

        inference_engine_me.reset_engine(engine)
        assert inference_engine_me.kv_cache_len(engine) == 0
        assert inference_engine_me.next_pos(engine) == 0
        print("Passed test_engine_forward_token_prefill_decode_and_reset")
    finally:
        tief._cleanup(model, engine, fixture_dir)


if __name__ == "__main__":
    test_engine_forward_token_one_layer_prefill()
    test_engine_forward_token_two_layer_prefill()
    test_engine_forward_token_prefill_decode_and_reset()
