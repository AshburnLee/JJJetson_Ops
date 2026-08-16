"""GenerateLoop binding: generate_loop_me.generate prefill + decode e2e."""

import os
import tempfile

import generate_loop_me
import inference_engine_me
import numpy as np
import transformer_model_me

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
EPS = 1e-6


def _config(num_layers: int = 1) -> dict:
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


def _fixture_tensors(num_layers: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(301)
    tensors: dict[str, np.ndarray] = {}
    for layer_idx in range(num_layers):
        seed = 200 + layer_idx * 50
        r = np.random.default_rng(seed)
        tensors[f"layer{layer_idx}.w_q"] = r.standard_normal((HIDDEN_SIZE, Q_DIM)).astype(
            np.float32
        )
        tensors[f"layer{layer_idx}.w_k"] = r.standard_normal((HIDDEN_SIZE, KV_DIM)).astype(
            np.float32
        )
        tensors[f"layer{layer_idx}.w_v"] = r.standard_normal((HIDDEN_SIZE, KV_DIM)).astype(
            np.float32
        )
        tensors[f"layer{layer_idx}.w_o"] = r.standard_normal((Q_DIM, HIDDEN_SIZE)).astype(
            np.float32
        )
        tensors[f"layer{layer_idx}.w_gate"] = r.standard_normal(
            (HIDDEN_SIZE, INTERMEDIATE_SIZE)
        ).astype(np.float32)
        tensors[f"layer{layer_idx}.w_up"] = r.standard_normal(
            (HIDDEN_SIZE, INTERMEDIATE_SIZE)
        ).astype(np.float32)
        tensors[f"layer{layer_idx}.w_down"] = r.standard_normal(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        ).astype(np.float32)
        tensors[f"layer{layer_idx}.w_input_layernorm"] = r.standard_normal((HIDDEN_SIZE,)).astype(
            np.float32
        )
        tensors[f"layer{layer_idx}.w_post_attention_layernorm"] = r.standard_normal(
            (HIDDEN_SIZE,)
        ).astype(np.float32)
    tensors["embed"] = rng.standard_normal((VOCAB_SIZE, HIDDEN_SIZE)).astype(np.float32)
    tensors["lm_head"] = rng.standard_normal((HIDDEN_SIZE, VOCAB_SIZE)).astype(np.float32)
    tensors["final_norm"] = rng.standard_normal((HIDDEN_SIZE,)).astype(np.float32)
    return tensors


def _setup_engine() -> tuple[int, int, str]:
    cfg = _config(1)
    fixture_dir = tempfile.mkdtemp(prefix="jj_gen_loop_")
    write_weight_fixture(fixture_dir, cfg, _fixture_tensors(1))
    model = transformer_model_me.create_model(**cfg)
    transformer_model_me.load_weights_from_fixture(model, fixture_dir)
    engine = inference_engine_me.create_engine(model)
    return model, engine, fixture_dir


def _cleanup(model: int, engine: int, fixture_dir: str) -> None:
    inference_engine_me.destroy_engine(engine)
    transformer_model_me.destroy_model(model)
    for fname in os.listdir(fixture_dir):
        os.remove(os.path.join(fixture_dir, fname))
    os.rmdir(fixture_dir)


def test_generate_loop_binding_prefill_decode() -> None:
    model, engine, fixture_dir = _setup_engine()
    try:
        prompt = np.array([3, 17, 42], dtype=np.int32)
        max_new = 4
        out = generate_loop_me.generate(engine, prompt, max_new)
        assert len(out) == max_new
        assert all(isinstance(t, int) for t in out)
        assert inference_engine_me.kv_cache_len(engine) == len(prompt) + max_new - 1
        print("generate_loop_binding_prefill_decode ok")
    finally:
        _cleanup(model, engine, fixture_dir)


def test_generate_loop_binding_eos_stop() -> None:
    model, engine, fixture_dir = _setup_engine()
    try:
        prompt = np.array([5, 9], dtype=np.int32)
        first = generate_loop_me.generate(engine, prompt, 1)
        eos_id = first[0]

        inference_engine_me.reset_engine(engine)
        out = generate_loop_me.generate(engine, prompt, 10, eos_token_id=eos_id)
        assert len(out) == 1
        assert out[0] == eos_id
        assert inference_engine_me.kv_cache_len(engine) == len(prompt)
        print("generate_loop_binding_eos_stop ok")
    finally:
        _cleanup(model, engine, fixture_dir)


def test_sampler_top_k_host_in_top_set() -> None:
    logits = np.array([0.1, 2.0, 0.5, -1.0, 1.5, 0.0], dtype=np.float32)
    for top_k in (1, 2, 3, len(logits)):
        top_indices = set(np.argsort(logits)[-top_k:].tolist())
        for seed in (0, 42, 99):
            got = generate_loop_me.sampler_top_k_host(logits, top_k, seed)
            assert got in top_indices, f"top_k={top_k} seed={seed} got={got} top={top_indices}"
    print("sampler_top_k_host_in_top_set ok")


def test_sampler_top_k_host_reproducible() -> None:
    logits = np.array([0.1, 2.0, 0.5, -1.0, 1.5, 0.0], dtype=np.float32)
    a = generate_loop_me.sampler_top_k_host(logits, 3, 42)
    b = generate_loop_me.sampler_top_k_host(logits, 3, 42)
    assert a == b
    print("sampler_top_k_host_reproducible ok")


def test_sampler_top_k_one_equals_greedy() -> None:
    logits = np.array([-1.0, 3.0, 2.0, 0.5], dtype=np.float32)
    assert generate_loop_me.sampler_top_k_host(logits, 1, 0) == 1
    print("sampler_top_k_one_equals_greedy ok")


def test_generate_top_k_one_matches_default() -> None:
    model, engine, fixture_dir = _setup_engine()
    try:
        prompt = np.array([3, 17, 42], dtype=np.int32)
        out_default = generate_loop_me.generate(engine, prompt, 3)
        inference_engine_me.reset_engine(engine)
        out_top1 = generate_loop_me.generate(engine, prompt, 3, top_k=1, seed=0)
        assert out_default == out_top1
        print("generate_top_k_one_matches_default ok")
    finally:
        _cleanup(model, engine, fixture_dir)


def test_generate_top_k_reproducible() -> None:
    model, engine, fixture_dir = _setup_engine()
    try:
        prompt = np.array([7, 11, 19], dtype=np.int32)
        out_a = generate_loop_me.generate(engine, prompt, 5, top_k=50, seed=12345)
        inference_engine_me.reset_engine(engine)
        out_b = generate_loop_me.generate(engine, prompt, 5, top_k=50, seed=12345)
        assert out_a == out_b
        assert len(out_a) == 5
        print("generate_top_k_reproducible ok")
    finally:
        _cleanup(model, engine, fixture_dir)


if __name__ == "__main__":
    test_generate_loop_binding_prefill_decode()
    test_generate_loop_binding_eos_stop()
    test_sampler_top_k_host_in_top_set()
    test_sampler_top_k_host_reproducible()
    test_sampler_top_k_one_equals_greedy()
    test_generate_top_k_one_matches_default()
    test_generate_top_k_reproducible()
