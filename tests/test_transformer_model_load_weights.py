"""TransformerModel: Loader fixture → H2D load weights."""

import os
import tempfile

import numpy as np
import transformer_model_me

from fixture_utils import write_weight_fixture

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_LAYERS = 1
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 32
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 256
SEED = 24


def _tiny_config() -> dict:
    return {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_layers": NUM_LAYERS,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "freq_base": 10000.0,
        "rms_norm_epsilon": 1e-6,
        "tie_word_embeddings": 0,
    }


def _fixture_tensors() -> dict[str, np.ndarray]:
    np.random.seed(SEED)
    return {
        "layer0.w_q": np.random.randn(HIDDEN_SIZE, Q_DIM).astype(np.float32),
        "layer0.w_k": np.random.randn(HIDDEN_SIZE, KV_DIM).astype(np.float32),
        "layer0.w_v": np.random.randn(HIDDEN_SIZE, KV_DIM).astype(np.float32),
        "layer0.w_o": np.random.randn(Q_DIM, HIDDEN_SIZE).astype(np.float32),
        "layer0.w_gate": np.random.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE).astype(np.float32),
        "layer0.w_up": np.random.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE).astype(np.float32),
        "layer0.w_down": np.random.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32),
        "layer0.w_input_layernorm": np.random.randn(HIDDEN_SIZE).astype(np.float32),
        "layer0.w_post_attention_layernorm": np.random.randn(HIDDEN_SIZE).astype(np.float32),
        "embed": np.random.randn(VOCAB_SIZE, HIDDEN_SIZE).astype(np.float32),
        "lm_head": np.random.randn(HIDDEN_SIZE, VOCAB_SIZE).astype(np.float32),
        "final_norm": np.random.randn(HIDDEN_SIZE).astype(np.float32),
    }


def test_load_weights_from_fixture():
    tensors = _fixture_tensors()
    fixture_dir = tempfile.mkdtemp(prefix="jj_model_fixture_")
    handle = transformer_model_me.create_model(**_tiny_config())
    try:
        write_weight_fixture(fixture_dir, _tiny_config(), tensors)
        assert not transformer_model_me.is_weights_loaded(handle)
        transformer_model_me.load_weights_from_fixture(handle, fixture_dir)
        assert transformer_model_me.is_weights_loaded(handle)

        got_w_q = transformer_model_me.read_layer_w_q_host(handle, 0, HIDDEN_SIZE, Q_DIM)
        if not np.array_equal(got_w_q, tensors["layer0.w_q"]):
            raise AssertionError("d_w_q mismatch after H2D")
        print("Passed test_load_weights_from_fixture")
    finally:
        transformer_model_me.destroy_model(handle)
        for fname in os.listdir(fixture_dir):
            os.remove(os.path.join(fixture_dir, fname))
        os.rmdir(fixture_dir)


def test_load_weights_immutable():
    tensors = _fixture_tensors()
    fixture_dir = tempfile.mkdtemp(prefix="jj_model_fixture_")
    handle = transformer_model_me.create_model(**_tiny_config())
    try:
        write_weight_fixture(fixture_dir, _tiny_config(), tensors)
        transformer_model_me.load_weights_from_fixture(handle, fixture_dir)
        try:
            transformer_model_me.load_weights_from_fixture(handle, fixture_dir)
        except RuntimeError:
            print("Passed test_load_weights_immutable")
            return
        raise AssertionError("expected RuntimeError on second load")
    finally:
        transformer_model_me.destroy_model(handle)
        for fname in os.listdir(fixture_dir):
            os.remove(os.path.join(fixture_dir, fname))
        os.rmdir(fixture_dir)


if __name__ == "__main__":
    test_load_weights_from_fixture()
    test_load_weights_immutable()
