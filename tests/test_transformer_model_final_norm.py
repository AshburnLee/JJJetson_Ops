"""TransformerModel: final RMSNorm after last layer block."""

import os
import tempfile

import numpy as np
import torch
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
VOCAB_SIZE = 512
MAX_SEQ_LEN = 256
NUM_TOKENS = 4
EPS = 1e-6
SEED = 37


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
        "rms_norm_epsilon": EPS,
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


def _rms_norm_ref(hidden: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros_like(hidden, order="F")
    for t in range(hidden.shape[1]):
        x = torch.from_numpy(hidden[:, t].copy())
        w = torch.from_numpy(weight)
        y = torch.nn.functional.rms_norm(x, (HIDDEN_SIZE,), w, eps=eps)
        out[:, t] = y.numpy()
    return out


def test_final_norm_forward_host():
    cfg = _tiny_config()
    tensors = _fixture_tensors()
    np.random.seed(SEED + 1)
    hidden_in = np.asfortranarray(np.random.randn(HIDDEN_SIZE, NUM_TOKENS).astype(np.float32))

    fixture_dir = tempfile.mkdtemp(prefix="jj_model_final_norm_")
    handle = transformer_model_me.create_model(**cfg)
    try:
        write_weight_fixture(fixture_dir, cfg, tensors)
        transformer_model_me.load_weights_from_fixture(handle, fixture_dir)

        got = transformer_model_me.final_norm_forward_host(handle, hidden_in, NUM_TOKENS)
        ref = _rms_norm_ref(hidden_in, tensors["final_norm"], EPS)
        if not np.allclose(got, ref, rtol=1e-5, atol=1e-5):
            raise AssertionError("final_norm_forward_host mismatch")
        print("final_norm_forward_host ok")
    finally:
        transformer_model_me.destroy_model(handle)
        for fname in os.listdir(fixture_dir):
            os.remove(os.path.join(fixture_dir, fname))
        os.rmdir(fixture_dir)


if __name__ == "__main__":
    test_final_norm_forward_host()
