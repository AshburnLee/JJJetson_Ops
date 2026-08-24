"""InferenceEngine forward: N-layer prefill/decode vs ref."""

import os
import tempfile

import inference_engine_me
import numpy as np
import transformer_model_me

import test_transformer_runner as tr
from fixture_utils import write_weight_fixture

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 32
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
VOCAB_SIZE = 512
MAX_SEQ_LEN = 256
NUM_PREFILL = 13
EPS = 1e-6
SEED = 42


def _config(num_layers: int) -> dict:
    return {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_layers": num_layers,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "freq_base": 10000.0,
        "rms_norm_epsilon": EPS,
        "tie_word_embeddings": 0,
    }


def _layer_block_tensors(layer_idx: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        f"layer{layer_idx}.w_q": rng.standard_normal((HIDDEN_SIZE, Q_DIM)).astype(np.float32),
        f"layer{layer_idx}.w_k": rng.standard_normal((HIDDEN_SIZE, KV_DIM)).astype(np.float32),
        f"layer{layer_idx}.w_v": rng.standard_normal((HIDDEN_SIZE, KV_DIM)).astype(np.float32),
        f"layer{layer_idx}.w_o": rng.standard_normal((Q_DIM, HIDDEN_SIZE)).astype(np.float32),
        f"layer{layer_idx}.w_gate": rng.standard_normal((HIDDEN_SIZE, INTERMEDIATE_SIZE)).astype(
            np.float32
        ),
        f"layer{layer_idx}.w_up": rng.standard_normal((HIDDEN_SIZE, INTERMEDIATE_SIZE)).astype(
            np.float32
        ),
        f"layer{layer_idx}.w_down": rng.standard_normal((INTERMEDIATE_SIZE, HIDDEN_SIZE)).astype(
            np.float32
        ),
        f"layer{layer_idx}.w_input_layernorm": rng.standard_normal((HIDDEN_SIZE,)).astype(
            np.float32
        ),
        f"layer{layer_idx}.w_post_attention_layernorm": rng.standard_normal((HIDDEN_SIZE,)).astype(
            np.float32
        ),
    }


def _fixture_tensors(num_layers: int) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for layer_idx in range(num_layers):
        tensors.update(_layer_block_tensors(layer_idx, seed=100 + layer_idx * 50))
    rng = np.random.default_rng(999)
    tensors["embed"] = rng.standard_normal((VOCAB_SIZE, HIDDEN_SIZE)).astype(np.float32)
    tensors["lm_head"] = rng.standard_normal((HIDDEN_SIZE, VOCAB_SIZE)).astype(np.float32)
    tensors["final_norm"] = rng.standard_normal((HIDDEN_SIZE,)).astype(np.float32)
    return tensors


def _layer_weights_from_fixture(
    tensors: dict[str, np.ndarray], layer_idx: int
) -> dict[str, np.ndarray]:
    prefix = f"layer{layer_idx}."
    return {
        "w_q": tensors[f"{prefix}w_q"],
        "w_k": tensors[f"{prefix}w_k"],
        "w_v": tensors[f"{prefix}w_v"],
        "w_o": tensors[f"{prefix}w_o"],
        "w_gate": tensors[f"{prefix}w_gate"],
        "w_up": tensors[f"{prefix}w_up"],
        "w_down": tensors[f"{prefix}w_down"],
        "w_input_layernorm": tensors[f"{prefix}w_input_layernorm"],
        "w_post_attention_layernorm": tensors[f"{prefix}w_post_attention_layernorm"],
    }


def _rms_norm_ref(hidden: np.ndarray, weight: np.ndarray) -> np.ndarray:
    import rms_norm_me

    num_tokens = hidden.shape[1]
    h2 = np.asfortranarray(hidden.reshape(HIDDEN_SIZE, num_tokens))
    out2 = np.zeros_like(h2)
    rms_norm_me.forward_host(h2, weight, out2, HIDDEN_SIZE, num_tokens, EPS)
    return out2.reshape(hidden.shape, order="F")


def _chain_layers_ref(
    hidden: np.ndarray,
    tensors: dict[str, np.ndarray],
    num_layers: int,
    pos_offset: int,
    kv_caches: list[tr.KvCacheRef],
) -> np.ndarray:
    h = hidden
    for layer_idx in range(num_layers):
        w = _layer_weights_from_fixture(tensors, layer_idx)
        h = tr.chain_linear_me_ref_step(
            h,
            w["w_q"],
            w["w_k"],
            w["w_v"],
            w["w_o"],
            w["w_gate"],
            w["w_up"],
            w["w_down"],
            w["w_input_layernorm"],
            w["w_post_attention_layernorm"],
            pos_offset,
            kv_caches[layer_idx],
        )
    return _rms_norm_ref(h, tensors["final_norm"])


def _setup_model_engine(num_layers: int) -> tuple[int, int, str, dict[str, np.ndarray]]:
    cfg = _config(num_layers)
    tensors = _fixture_tensors(num_layers)
    fixture_dir = tempfile.mkdtemp(prefix="jj_engine_fwd_")
    write_weight_fixture(fixture_dir, cfg, tensors)
    model = transformer_model_me.create_model(**cfg)
    transformer_model_me.load_weights_from_fixture(model, fixture_dir)
    engine = inference_engine_me.create_engine(model)
    return model, engine, fixture_dir, tensors


def _cleanup(model: int, engine: int, fixture_dir: str) -> None:
    inference_engine_me.destroy_engine(engine)
    transformer_model_me.destroy_model(model)
    for fname in os.listdir(fixture_dir):
        os.remove(os.path.join(fixture_dir, fname))
    os.rmdir(fixture_dir)


def test_engine_forward_one_layer_prefill() -> None:
    np.random.seed(SEED)
    model, engine, fixture_dir, tensors = _setup_model_engine(1)
    try:
        hidden = np.asfortranarray(
            np.random.randn(HIDDEN_SIZE, NUM_PREFILL, 1, 1).astype(np.float32)
        )
        out = np.zeros_like(hidden)
        inference_engine_me.forward_hidden_host(engine, NUM_PREFILL, 0, hidden, out)
        assert inference_engine_me.kv_cache_len(engine) == NUM_PREFILL
        assert inference_engine_me.next_pos(engine) == NUM_PREFILL

        kv = [tr.KvCacheRef(HEAD_DIM, MAX_SEQ_LEN, NUM_KV_HEADS)]
        ref = _chain_layers_ref(hidden, tensors, 1, 0, kv)
        max_abs = np.max(np.abs(out - ref))
        assert np.allclose(out, ref, atol=1e-4, rtol=1e-4), f"max_abs_diff={max_abs:e}"
        print("Passed test_engine_forward_one_layer_prefill")
    finally:
        _cleanup(model, engine, fixture_dir)


def test_engine_forward_two_layer_prefill() -> None:
    np.random.seed(SEED + 1)
    model, engine, fixture_dir, tensors = _setup_model_engine(2)
    try:
        hidden = np.asfortranarray(
            np.random.randn(HIDDEN_SIZE, NUM_PREFILL, 1, 1).astype(np.float32)
        )
        out = np.zeros_like(hidden)
        inference_engine_me.forward_hidden_host(engine, NUM_PREFILL, 0, hidden, out)
        assert inference_engine_me.kv_cache_num_layers(engine) == 2
        assert inference_engine_me.kv_cache_len(engine) == NUM_PREFILL

        kv = [tr.KvCacheRef(HEAD_DIM, MAX_SEQ_LEN, NUM_KV_HEADS) for _ in range(2)]
        ref = _chain_layers_ref(hidden, tensors, 2, 0, kv)
        max_abs = np.max(np.abs(out - ref))
        assert np.allclose(out, ref, atol=1e-4, rtol=1e-4), f"max_abs_diff={max_abs:e}"
        print("Passed test_engine_forward_two_layer_prefill")
    finally:
        _cleanup(model, engine, fixture_dir)


def test_engine_prefill_decode_and_reset() -> None:
    np.random.seed(SEED + 2)
    model, engine, fixture_dir, _tensors = _setup_model_engine(1)
    try:
        hidden_p = np.asfortranarray(
            np.random.randn(HIDDEN_SIZE, NUM_PREFILL, 1, 1).astype(np.float32)
        )
        out_p = np.zeros_like(hidden_p)
        inference_engine_me.forward_hidden_host(engine, NUM_PREFILL, 0, hidden_p, out_p)
        assert inference_engine_me.kv_cache_len(engine) == NUM_PREFILL

        hidden_d = np.asfortranarray(np.random.randn(HIDDEN_SIZE, 1, 1, 1).astype(np.float32))
        out_d = np.zeros_like(hidden_d)
        inference_engine_me.forward_hidden_host(engine, 1, NUM_PREFILL, hidden_d, out_d)
        assert inference_engine_me.kv_cache_len(engine) == NUM_PREFILL + 1
        assert inference_engine_me.next_pos(engine) == NUM_PREFILL + 1

        inference_engine_me.reset_engine(engine)
        assert inference_engine_me.kv_cache_len(engine) == 0
        assert inference_engine_me.next_pos(engine) == 0
        print("Passed test_engine_prefill_decode_and_reset")
    finally:
        _cleanup(model, engine, fixture_dir)


if __name__ == "__main__":
    test_engine_forward_one_layer_prefill()
    test_engine_forward_two_layer_prefill()
    test_engine_prefill_decode_and_reset()
