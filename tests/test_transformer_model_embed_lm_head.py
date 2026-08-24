"""TransformerModel: embed gather + lm_head forward vs numpy ref."""

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
VOCAB_SIZE = 512
MAX_SEQ_LEN = 256
SEED = 31


def _tiny_config(tie_word_embeddings: int = 0) -> dict:
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
        "tie_word_embeddings": tie_word_embeddings,
    }


def _fixture_tensors(tie_word_embeddings: int = 0) -> dict[str, np.ndarray]:
    np.random.seed(SEED)
    embed = np.random.randn(VOCAB_SIZE, HIDDEN_SIZE).astype(np.float32)
    tensors = {
        "layer0.w_q": np.random.randn(HIDDEN_SIZE, Q_DIM).astype(np.float32),
        "layer0.w_k": np.random.randn(HIDDEN_SIZE, KV_DIM).astype(np.float32),
        "layer0.w_v": np.random.randn(HIDDEN_SIZE, KV_DIM).astype(np.float32),
        "layer0.w_o": np.random.randn(Q_DIM, HIDDEN_SIZE).astype(np.float32),
        "layer0.w_gate": np.random.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE).astype(np.float32),
        "layer0.w_up": np.random.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE).astype(np.float32),
        "layer0.w_down": np.random.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE).astype(np.float32),
        "layer0.w_input_layernorm": np.random.randn(HIDDEN_SIZE).astype(np.float32),
        "layer0.w_post_attention_layernorm": np.random.randn(HIDDEN_SIZE).astype(np.float32),
        "embed": embed,
        "final_norm": np.random.randn(HIDDEN_SIZE).astype(np.float32),
    }
    if not tie_word_embeddings:
        tensors["lm_head"] = np.random.randn(HIDDEN_SIZE, VOCAB_SIZE).astype(np.float32)
    return tensors


def _embed_ref(embed: np.ndarray, token_ids: np.ndarray, num_tokens: int) -> np.ndarray:
    hidden = np.zeros((HIDDEN_SIZE, num_tokens), dtype=np.float32, order="F")
    for t in range(num_tokens):
        hidden[:, t] = embed[token_ids[t], :]
    return hidden


def _lm_head_ref(
    hidden: np.ndarray, lm_head: np.ndarray, embed: np.ndarray, tied: bool, num_tokens: int
) -> np.ndarray:
    logits = np.zeros((VOCAB_SIZE, num_tokens), dtype=np.float32, order="F")
    for t in range(num_tokens):
        if tied:
            logits[:, t] = embed @ hidden[:, t]
        else:
            logits[:, t] = lm_head.T @ hidden[:, t]
    return logits


def _run_embed_lm_head_test(tie_word_embeddings: int, test_name: str) -> None:
    cfg = _tiny_config(tie_word_embeddings)
    tensors = _fixture_tensors(tie_word_embeddings)
    token_ids = np.array([3, 17, 42, 99], dtype=np.int32)
    num_tokens = int(token_ids.shape[0])

    fixture_dir = tempfile.mkdtemp(prefix="jj_model_embed_")
    handle = transformer_model_me.create_model(**cfg)
    try:
        write_weight_fixture(fixture_dir, cfg, tensors)
        transformer_model_me.load_weights_from_fixture(handle, fixture_dir)

        hidden = transformer_model_me.embed_forward_host(handle, token_ids, num_tokens)
        ref_hidden = _embed_ref(tensors["embed"], token_ids, num_tokens)
        if not np.allclose(hidden, ref_hidden, rtol=1e-5, atol=1e-5):
            raise AssertionError("embed_forward_host mismatch")

        lm_head = tensors.get("lm_head")
        logits = transformer_model_me.lm_head_forward_host(handle, hidden, num_tokens)
        ref_logits = _lm_head_ref(
            hidden, lm_head, tensors["embed"], tie_word_embeddings != 0, num_tokens
        )
        if not np.allclose(logits, ref_logits, rtol=1e-4, atol=1e-4):
            raise AssertionError("lm_head_forward_host mismatch")

        print(f"Passed {test_name}")
    finally:
        transformer_model_me.destroy_model(handle)
        for fname in os.listdir(fixture_dir):
            os.remove(os.path.join(fixture_dir, fname))
        os.rmdir(fixture_dir)


def test_embed_lm_head_untied():
    _run_embed_lm_head_test(tie_word_embeddings=0, test_name="test_embed_lm_head_untied")


def test_embed_lm_head_tied():
    _run_embed_lm_head_test(tie_word_embeddings=1, test_name="test_embed_lm_head_tied")


if __name__ == "__main__":
    test_embed_lm_head_untied()
    test_embed_lm_head_tied()
