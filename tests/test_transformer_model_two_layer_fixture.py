"""2-layer fixture 权重 layout：layer0/layer1 命名、shape、H2D 逐 tensor 对齐。"""

import os
import tempfile

import numpy as np
import transformer_model_me
import weight_loader_me

from fixture_utils import write_weight_fixture

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_LAYERS = 2
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 32
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 256

# fixture 与 Loader 约定：row-major，与 test_transformer_model_load_weights 一致
LAYER_WEIGHT_SPECS: list[tuple[str, tuple[int, ...]]] = [
    ("w_q", (HIDDEN_SIZE, Q_DIM)),
    ("w_k", (HIDDEN_SIZE, KV_DIM)),
    ("w_v", (HIDDEN_SIZE, KV_DIM)),
    ("w_o", (Q_DIM, HIDDEN_SIZE)),
    ("w_gate", (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
    ("w_up", (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
    ("w_down", (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
    ("w_input_layernorm", (HIDDEN_SIZE,)),
    ("w_post_attention_layernorm", (HIDDEN_SIZE,)),
]


def _config() -> dict:
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


def _layer_tensors(layer_idx: int, seed: int) -> dict[str, np.ndarray]:
    """每层用不同 seed，保证 layer0 != layer1。"""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for suffix, shape in LAYER_WEIGHT_SPECS:
        out[f"layer{layer_idx}.{suffix}"] = rng.standard_normal(shape).astype(np.float32)
    return out


def _two_layer_fixture_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    tensors.update(_layer_tensors(0, seed=100))
    tensors.update(_layer_tensors(1, seed=200))
    rng = np.random.default_rng(300)
    tensors["embed"] = rng.standard_normal((VOCAB_SIZE, HIDDEN_SIZE)).astype(np.float32)
    tensors["lm_head"] = rng.standard_normal((HIDDEN_SIZE, VOCAB_SIZE)).astype(np.float32)
    tensors["final_norm"] = rng.standard_normal((HIDDEN_SIZE,)).astype(np.float32)
    return tensors


def _assert_tensor_equal(got: np.ndarray, expected: np.ndarray, name: str) -> None:
    if not np.array_equal(got, expected):
        raise AssertionError(f"tensor mismatch after H2D: {name}")


def _read_layer_weight(
    handle: int, layer_idx: int, suffix: str, shape: tuple[int, ...]
) -> np.ndarray:
    return np.asarray(
        transformer_model_me.read_layer_weight_host(handle, layer_idx, suffix, list(shape)),
        dtype=np.float32,
    )


def _cleanup_fixture_dir(fixture_dir: str) -> None:
    for fname in os.listdir(fixture_dir):
        os.remove(os.path.join(fixture_dir, fname))
    os.rmdir(fixture_dir)


def test_two_layer_fixture_layout_roundtrip() -> None:
    cfg = _config()
    tensors = _two_layer_fixture_tensors()
    fixture_dir = tempfile.mkdtemp(prefix="jj_model_2layer_fixture_")
    handle = transformer_model_me.create_model(**cfg)
    try:
        write_weight_fixture(fixture_dir, cfg, tensors)

        loaded = weight_loader_me.load_fixture(fixture_dir)
        assert loaded["num_tensors"] == len(tensors)
        assert loaded["config"]["num_layers"] == NUM_LAYERS

        for name, expected in tensors.items():
            got_host = np.asarray(loaded["tensors"][name], dtype=np.float32)
            _assert_tensor_equal(got_host, expected, name)

        transformer_model_me.load_weights_from_fixture(handle, fixture_dir)
        assert transformer_model_me.get_num_layers(handle) == NUM_LAYERS

        for layer_idx in range(NUM_LAYERS):
            for suffix, shape in LAYER_WEIGHT_SPECS:
                name = f"layer{layer_idx}.{suffix}"
                got = _read_layer_weight(handle, layer_idx, suffix, shape)
                _assert_tensor_equal(got, tensors[name], name)

        _assert_tensor_equal(
            np.asarray(
                transformer_model_me.read_global_weight_host(
                    handle, "embed", [VOCAB_SIZE, HIDDEN_SIZE]
                ),
                dtype=np.float32,
            ),
            tensors["embed"],
            "embed",
        )
        _assert_tensor_equal(
            np.asarray(
                transformer_model_me.read_global_weight_host(
                    handle, "lm_head", [HIDDEN_SIZE, VOCAB_SIZE]
                ),
                dtype=np.float32,
            ),
            tensors["lm_head"],
            "lm_head",
        )
        _assert_tensor_equal(
            np.asarray(
                transformer_model_me.read_global_weight_host(handle, "final_norm", [HIDDEN_SIZE]),
                dtype=np.float32,
            ),
            tensors["final_norm"],
            "final_norm",
        )

        w_q0 = _read_layer_weight(handle, 0, "w_q", (HIDDEN_SIZE, Q_DIM))
        w_q1 = _read_layer_weight(handle, 1, "w_q", (HIDDEN_SIZE, Q_DIM))
        if np.array_equal(w_q0, w_q1):
            raise AssertionError("layer0.w_q and layer1.w_q must differ")

        print("two_layer_fixture_layout_roundtrip ok")
    finally:
        transformer_model_me.destroy_model(handle)
        _cleanup_fixture_dir(fixture_dir)


def test_two_layer_fixture_missing_layer_rejected() -> None:
    """缺 layer1 权重时 load 应失败。"""
    cfg = _config()
    tensors = _two_layer_fixture_tensors()
    del tensors["layer1.w_q"]
    fixture_dir = tempfile.mkdtemp(prefix="jj_model_2layer_bad_")
    handle = transformer_model_me.create_model(**cfg)
    try:
        write_weight_fixture(fixture_dir, cfg, tensors)
        try:
            transformer_model_me.load_weights_from_fixture(handle, fixture_dir)
        except RuntimeError:
            assert not transformer_model_me.is_weights_loaded(handle)
            print("missing layer1 weight rejected ok")
            return
        raise AssertionError("expected RuntimeError when layer1.w_q missing")
    finally:
        transformer_model_me.destroy_model(handle)
        _cleanup_fixture_dir(fixture_dir)


if __name__ == "__main__":
    """保证2层权重装对了，容器没问题，H2D 没问题"""
    test_two_layer_fixture_layout_roundtrip()
    test_two_layer_fixture_missing_layer_rejected()
