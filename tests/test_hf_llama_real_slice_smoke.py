"""可选冒烟单测：真实 HF Llama 切片（scripts/export_hf_llama_slice.py 输出）走 Model + Engine。

未设置 JJ_HF_LLAMA_SLICE_DIR 时跳过。不要在 Python 里 load_safetensors_hf_llama
把全部权重再拷一份 numpy：Orin 内存不够。只读 config.txt，H2D 由 C++ Loader 完成。

这是smoke test，不是数值正确性测试。Passed 只表示：切片能 load、Engine 能返回、
logits 没有 nan/inf、KV 长度对上。没有和 HuggingFace 前 2 层比 logits，
也没有内部 fixture tensor ref。argmax 打印只是看一眼，不能当标准答案。

精度：引擎没有 BF16 权重类型。切片时已把 HF BF16 转成 F32（BF16 是 F32 的子集，
这一步不丢数）。Linear / RMSNorm / SwiGLU 都是 F32。FA 的 Q/K/V staging 是
IEEE fp16，不是 BF16。

TinyLlama 是 GQA g=8（32 Q / 4 KV）。生产 FA 每 block 仍 2 个 Q，要求 g 为偶数；
g=8 已支持。本测试仍不和 HF 比 logits。
"""

import os

import inference_engine_me
import numpy as np
import transformer_model_me

_NUM_PREFILL = 4


def _slice_dir() -> str:
    return os.environ.get("JJ_HF_LLAMA_SLICE_DIR", "").strip()


def _parse_config_txt(path: str) -> dict:
    cfg: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in ("freq_base", "rms_norm_epsilon"):
                cfg[key] = float(value)
            else:
                cfg[key] = int(float(value))
    return cfg


# 冒烟：路径能走通。不算对、也不覆盖 FA/GQA（见文件头注释）。
def test_hf_llama_real_slice_smoke() -> None:
    slice_dir = _slice_dir()
    if not slice_dir:
        print("Passed test_hf_llama_real_slice_smoke skipped")
        return

    st_path = os.path.join(slice_dir, "model.safetensors")
    cfg_path = os.path.join(slice_dir, "config.txt")
    if not os.path.isfile(st_path):
        raise AssertionError(f"missing {st_path}")
    if not os.path.isfile(cfg_path):
        raise AssertionError(f"missing {cfg_path}")

    cfg = _parse_config_txt(cfg_path)
    if cfg["num_layers"] < 1:
        raise AssertionError("num_layers < 1")

    # 1. create model
    model = transformer_model_me.create_model(**cfg)
    engine = None
    try:
        # 2. load 权重从正式模型的slice中
        transformer_model_me.load_weights_from_safetensors_hf_llama(model, st_path)
        if not transformer_model_me.is_weights_loaded(model):
            raise AssertionError("weights not loaded")
        # 3. 创建 engine
        engine = inference_engine_me.create_engine(model)
        vocab = int(cfg["vocab_size"])
        token_ids = np.array([1, 2, 3, 4], dtype=np.int32)
        assert token_ids.shape[0] == _NUM_PREFILL
        logits = np.zeros((vocab, _NUM_PREFILL), dtype=np.float32, order="F")
        # 4. prefill。FA 已支持 g=8；下面只检查 KV 长度和 logits 有限，不和 HF 比。
        inference_engine_me.forward_token_host(engine, _NUM_PREFILL, 0, token_ids, logits)
        assert inference_engine_me.kv_cache_len(engine) == _NUM_PREFILL
        assert inference_engine_me.kv_cache_num_layers(engine) == cfg["num_layers"]
        if not np.isfinite(logits).all():
            raise AssertionError("prefill logits not finite")

        decode_id = np.array([int(np.argmax(logits[:, -1]))], dtype=np.int32)
        logits_d = np.zeros((vocab, 1), dtype=np.float32, order="F")
        # 5. 一次 decode。argmax 只打印，不和 HF 对比。
        inference_engine_me.forward_token_host(engine, 1, _NUM_PREFILL, decode_id, logits_d)
        assert inference_engine_me.kv_cache_len(engine) == _NUM_PREFILL + 1
        if not np.isfinite(logits_d).all():
            raise AssertionError("decode logits not finite")

        print(
            f"  num_layers={cfg['num_layers']} hidden={cfg['hidden_size']} "
            f"prefill_argmax={int(np.argmax(logits[:, -1]))} "
            f"decode_argmax={int(np.argmax(logits_d[:, 0]))}"
        )
        print("Passed test_hf_llama_real_slice_smoke")
    finally:
        if engine is not None:
            # 6. 释放 engine
            inference_engine_me.destroy_engine(engine)
        # 7. 释放 model
        transformer_model_me.destroy_model(model)


if __name__ == "__main__":
    test_hf_llama_real_slice_smoke()
