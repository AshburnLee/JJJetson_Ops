"""TransformerModel skeleton: create/destroy GPU weight container."""

import transformer_model_me

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_LAYERS = 2
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 32
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 256


def _create_cfg(tie_word_embeddings: int = 0) -> dict:
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


def test_create_destroy():
    handle = transformer_model_me.create_model(**_create_cfg())
    try:
        assert transformer_model_me.get_num_layers(handle) == NUM_LAYERS
        assert transformer_model_me.is_tied_embeddings(handle) is False
        print("create_destroy ok")
    finally:
        transformer_model_me.destroy_model(handle)


def test_tied_embeddings():
    handle = transformer_model_me.create_model(**_create_cfg(tie_word_embeddings=1))
    try:
        assert transformer_model_me.is_tied_embeddings(handle) is True
        print("tied_embeddings ok")
    finally:
        transformer_model_me.destroy_model(handle)


if __name__ == "__main__":
    test_create_destroy()
    test_tied_embeddings()
