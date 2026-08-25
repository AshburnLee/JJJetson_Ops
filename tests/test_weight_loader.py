"""WeightLoader: ModelConfig validate + fixture load."""

import os
import tempfile

import numpy as np
import weight_loader_me

from fixture_utils import (
    export_fixture_dir_to_safetensors,
    internal_tensors_to_hf_llama_layout,
    write_hf_llama_config_json,
    write_safetensors_file,
    write_weight_fixture,
)

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
    print("Passed test_model_config_validate_ok")


def test_model_config_validate_rejects_bad_heads():
    cfg = _tiny_config()
    cfg["head_dim"] = 31
    try:
        weight_loader_me.validate_config(**cfg)
    except ValueError:
        print("Passed test_model_config_validate_rejects_bad_heads")
        return
    raise AssertionError("expected ValueError for invalid head_dim")


def test_load_fixture_roundtrip():
    tensors = _one_layer_tensors()
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
        print("Passed test_load_fixture_roundtrip")
    finally:
        for fname in os.listdir(fixture_dir):
            os.remove(os.path.join(fixture_dir, fname))
        os.rmdir(fixture_dir)


def test_load_safetensors_read_format():
    tensors = {
        "embed": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "layer0.w_q": np.array([0.5, -0.25, 1.25, 0.0], dtype=np.float32),
        "final_norm": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }
    st_path = tempfile.mktemp(suffix=".safetensors")
    try:
        write_safetensors_file(st_path, tensors)
        loaded = weight_loader_me.load_safetensors(st_path)
        assert loaded["num_tensors"] == len(tensors)
        for name, expected in tensors.items():
            got = np.asarray(loaded["tensors"][name], dtype=np.float32)
            if not np.array_equal(got, expected):
                raise AssertionError(f"safetensors tensor mismatch: {name}")
        print("Passed test_load_safetensors_read_format")
    finally:
        if os.path.exists(st_path):
            os.remove(st_path)


def test_load_safetensors_with_optional_config():
    tensors = {"embed": np.array([[1.0, 2.0]], dtype=np.float32)}
    tmp_dir = tempfile.mkdtemp(prefix="jj_st_cfg_")
    st_path = os.path.join(tmp_dir, "weights.safetensors")
    try:
        write_safetensors_file(st_path, tensors)
        write_weight_fixture(tmp_dir, _tiny_config(), {})
        loaded = weight_loader_me.load_safetensors(st_path)
        assert loaded["num_tensors"] == 1
        cfg = loaded["config"]
        assert cfg["hidden_size"] == HIDDEN_SIZE
        assert cfg["num_layers"] == 1
        print("Passed test_load_safetensors_with_optional_config")
    finally:
        for fname in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, fname))
        os.rmdir(tmp_dir)


def _one_layer_tensors() -> dict[str, np.ndarray]:
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


def _assert_loaded_tensors_match(expected: dict[str, np.ndarray], loaded: dict, label: str) -> None:
    if len(loaded) != len(expected):
        raise AssertionError(
            f"{label}: tensor count mismatch got={len(loaded)} expected={len(expected)}"
        )
    for name, exp in expected.items():
        got = np.asarray(loaded[name], dtype=np.float32)
        if not np.array_equal(got, exp):
            raise AssertionError(f"{label}: tensor mismatch: {name}")


def test_fixture_safetensors_roundtrip():
    """fixture 目录 -> 导出 .safetensors -> load_safetensors 与 load_fixture 逐 tensor 一致。"""
    tensors = _one_layer_tensors()
    fixture_dir = tempfile.mkdtemp(prefix="jj_fixture_st_roundtrip_")
    st_path = os.path.join(fixture_dir, "weights.safetensors")
    try:
        write_weight_fixture(fixture_dir, _tiny_config(), tensors)

        from_fixture = weight_loader_me.load_fixture(fixture_dir)
        export_fixture_dir_to_safetensors(fixture_dir, st_path)
        from_st = weight_loader_me.load_safetensors(st_path)

        assert from_fixture["num_tensors"] == len(tensors)
        assert from_st["num_tensors"] == len(tensors)
        _assert_loaded_tensors_match(tensors, from_fixture["tensors"], "load_fixture")
        _assert_loaded_tensors_match(tensors, from_st["tensors"], "load_safetensors")

        # safetensors 与同目录 config.txt：load_safetensors 应带上 ModelConfig
        for key, value in _tiny_config().items():
            got = from_st["config"][key]
            if isinstance(value, float):
                if not np.isclose(got, value, rtol=0.0, atol=1e-9):
                    raise AssertionError(f"config mismatch: {key} got={got} expected={value}")
            elif got != value:
                raise AssertionError(f"config mismatch: {key} got={got} expected={value}")

        print("Passed test_fixture_safetensors_roundtrip")
    finally:
        for fname in os.listdir(fixture_dir):
            os.remove(os.path.join(fixture_dir, fname))
        os.rmdir(fixture_dir)


# safetensors 步骤 3 单测：HF Llama 风格 key + 矩阵方向，经映射后应与内部 fixture 一致。
#
# Big picture 里它在哪？
#   验证 hf_llama_weight_map / load_safetensors_hf_llama：HF 名（如 model.layers.0.self_attn.q_proj.weight）
#   和 PyTorch Linear [out,in] 布局，能否还原成内部 layer0.w_q 等 + fixture row-major layout。
#   基准是同一批 tensor 走 write_weight_fixture + load_fixture；不要求下载真 HF 模型。
#   设计文档：doc/guide/hf_llama_weight_map.md
#
# 函数内部顺序（逐步）：
#   例：_one_layer_tensors() 共 12 张（embed、layer0.w_q … lm_head、final_norm）
#   step 1. 临时目录写 fixture + load_fixture 当基准
#   step 2. internal_tensors_to_hf_llama_layout 转成 HF key/layout，write_safetensors_file 写出 model.safetensors
#   step 3. load_safetensors_hf_llama 读回，与基准、原始 tensors 逐张对比
#   step 4. finally 删临时目录
#
# 调用契约：model.safetensors 运行时生成，仓库里无需事先保存；须已 build weight_loader_me
def test_hf_llama_safetensors_name_map():
    # step 0：1-layer 内部 tensor + 临时目录
    tensors = _one_layer_tensors()
    tmp_dir = tempfile.mkdtemp(prefix="jj_hf_llama_map_")
    # model.safetensors 不必事先放在仓库里；下面 write_safetensors_file 会按 HF 名/layout 运行时写出
    st_path = os.path.join(tmp_dir, "model.safetensors")
    try:
        # step 1：fixture 基准
        write_weight_fixture(tmp_dir, _tiny_config(), tensors)
        baseline = weight_loader_me.load_fixture(tmp_dir)

        # step 2：写 HF 风格 safetensors
        hf_tensors = internal_tensors_to_hf_llama_layout(tensors)
        write_safetensors_file(st_path, hf_tensors)

        # step 3：映射读回并对比
        mapped = weight_loader_me.load_safetensors_hf_llama(st_path)

        for label, loaded in (
            ("baseline (load_fixture)", baseline),
            ("mapped (load_safetensors_hf_llama)", mapped),
        ):
            print(f"--- {label} ---")
            print(f"  num_tensors={loaded['num_tensors']}")
            cfg = loaded["config"]
            print(
                f"  config: hidden_size={cfg['hidden_size']} num_layers={cfg['num_layers']} "
                f"vocab_size={cfg['vocab_size']} tie_word_embeddings={cfg['tie_word_embeddings']}"
            )
            for name in sorted(loaded["tensors"].keys()):
                arr = np.asarray(loaded["tensors"][name], dtype=np.float32)
                flat = arr.ravel()
                print(
                    f"  {name}: shape={tuple(arr.shape)} "
                    f"min={flat.min():.6f} max={flat.max():.6f} first8={flat[:8]}"
                )

        assert mapped["num_tensors"] == len(tensors)
        _assert_loaded_tensors_match(tensors, mapped["tensors"], "load_safetensors_hf_llama")
        _assert_loaded_tensors_match(tensors, baseline["tensors"], "load_fixture baseline")
        print("Passed test_hf_llama_safetensors_name_map")
    finally:
        # step 4：清理临时文件
        for fname in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, fname))
        os.rmdir(tmp_dir)


# safetensors 步骤 4：没有 config.txt 时，同目录 HF config.json 应填出合法 ModelConfig。
#
# 例：config.json 有 hidden_size=128、num_attention_heads=4，故意不写 head_dim
#     -> Loader 算出 head_dim=128/4=32，再和内部 fixture tensor 对比。
def test_hf_llama_config_json():
    import json

    tensors = _one_layer_tensors()
    tmp_dir = tempfile.mkdtemp(prefix="jj_hf_llama_cfg_json_")
    st_path = os.path.join(tmp_dir, "model.safetensors")
    json_path = os.path.join(tmp_dir, "config.json")
    try:
        write_safetensors_file(st_path, internal_tensors_to_hf_llama_layout(tensors))
        write_hf_llama_config_json(json_path, _tiny_config())
        with open(json_path, encoding="utf-8") as f:
            hf_cfg = json.load(f)
        del hf_cfg["head_dim"]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(hf_cfg, f)

        mapped = weight_loader_me.load_safetensors_hf_llama(st_path)
        cfg = mapped["config"]
        expected = _tiny_config()
        for key, value in expected.items():
            got = cfg[key]
            if isinstance(value, float):
                if not np.isclose(got, value, rtol=0.0, atol=1e-6):
                    raise AssertionError(f"config.json mismatch: {key} got={got} expected={value}")
            elif got != value:
                raise AssertionError(f"config.json mismatch: {key} got={got} expected={value}")
        _assert_loaded_tensors_match(tensors, mapped["tensors"], "config.json + hf_llama map")
        print("Passed test_hf_llama_config_json")
    finally:
        for fname in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, fname))
        os.rmdir(tmp_dir)


if __name__ == "__main__":
    test_model_config_validate_ok()
    test_model_config_validate_rejects_bad_heads()
    test_load_fixture_roundtrip()
    test_load_safetensors_read_format()
    test_load_safetensors_with_optional_config()
    test_fixture_safetensors_roundtrip()
    test_hf_llama_safetensors_name_map()
    test_hf_llama_config_json()
