"""WeightLoader: ModelConfig validate + fixture load."""

import os
import tempfile

import numpy as np
import weight_loader_me

from fixture_utils import write_weight_fixture

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
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
        "num_layers": 1,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "freq_base": 10000.0,
        "rms_norm_epsilon": 1e-6,
        "tie_word_embeddings": 0,
    }


def test_model_config_validate_ok():
    cfg = _tiny_config()
    assert weight_loader_me.validate_config(**cfg)
    print("validate_config ok")


def test_model_config_validate_rejects_bad_heads():
    cfg = _tiny_config()
    cfg["head_dim"] = 31
    try:
        weight_loader_me.validate_config(**cfg)
    except ValueError:
        print("validate_config rejects bad head_dim")
        return
    raise AssertionError("expected ValueError for invalid head_dim")


def test_load_fixture_roundtrip():
    np.random.seed(SEED)
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
        "embed": np.random.randn(VOCAB_SIZE, HIDDEN_SIZE).astype(np.float32),
        "lm_head": np.random.randn(HIDDEN_SIZE, VOCAB_SIZE).astype(np.float32),
        "final_norm": np.random.randn(HIDDEN_SIZE).astype(np.float32),
    }

    fixture_dir = tempfile.mkdtemp(prefix="jj_weight_fixture_")
    try:
        # 实时写入 配置和二进制权重文件
        write_weight_fixture(fixture_dir, _tiny_config(), tensors)
        # loader 读这个目录 load 到Tensor
        loaded = weight_loader_me.load_fixture(fixture_dir)

        assert loaded["num_tensors"] == len(tensors)
        loaded_cfg = loaded["config"]
        for key, value in _tiny_config().items():
            got = loaded_cfg[key]
            if isinstance(value, float):
                if not np.isclose(got, value, rtol=0.0, atol=1e-9):
                    raise AssertionError(f"config mismatch: {key} got={got} expected={value}")
            elif got != value:
                raise AssertionError(f"config mismatch: {key} got={got} expected={value}")

        for name, expected in tensors.items():
            got = np.asarray(loaded["tensors"][name], dtype=np.float32)
            if not np.array_equal(got, expected):
                raise AssertionError(f"tensor mismatch: {name}")
        print("load_fixture roundtrip ok")
    finally:
        for fname in os.listdir(fixture_dir):
            os.remove(os.path.join(fixture_dir, fname))
        os.rmdir(fixture_dir)


if __name__ == "__main__":
    test_model_config_validate_ok()
    test_model_config_validate_rejects_bad_heads()
    test_load_fixture_roundtrip()
