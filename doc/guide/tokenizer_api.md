# Tokenizer / Detokenizer API（引擎外，模块 5）

Python 模块：`py/hf_tokenizer.py`（`import hf_tokenizer`）。

`run_tests.sh` 的 `PYTHONPATH` 含：`python/`（`*_me.so`，gitignore）+ `py/`（可提交的纯 Python）+ `tests/`。
**不要**把正式源码放进 `python/`，那边整目录被 ignore。

**边界**：Engine / GenerateLoop 只认 `token_ids`。本模块在 CPU 上做文本 <-> id，**不进 C++、不上 GPU**。对齐 vLLM / SGLang：分词在引擎外。

生命周期位置见 [`../design/phase2_lifecycle.md`](../design/phase2_lifecycle.md) §0 与 §5 Tokenizer。GenerateLoop 契约见 [`generate_loop_device_api.md`](generate_loop_device_api.md)。

---

## 数据流

~~~
文本  --encode (CPU, hf_tokenizer)-->  token_ids
      --GenerateLoop / Engine (GPU)-->  新 token_ids
      --decode (CPU, hf_tokenizer)-->  文本
~~~

正式入口优先用 `generate_text`（内部就是上面三步）。需要自己控 id 时，再拆开调 `encode` / `generate_loop_me.generate` / `decode`。

---

## 词表文件从哪来

当前对齐 **TinyLlama-1.1B-Chat-v1.0** 的 HF tokenizer（与切片权重同一词表，vocab=32000）。

目录解析顺序（`resolve_tokenizer_dir`）：

~~~
1. 环境变量 JJ_HF_TOKENIZER_DIR
2. 传入的 slice_dir 内已有 tokenizer.json + tokenizer.model
3. 默认 models/hf_src/TinyLlama__TinyLlama-1.1B-Chat-v1.0/
~~~

必需文件：`tokenizer.json`、`tokenizer.model`（另有 `tokenizer_config.json`、`special_tokens_map.json` 等，由 `AutoTokenizer` 读）。

权重切片目录 `models/tinyllama_2layer/` 可以没有 tokenizer 文件；默认会落到 `hf_src` 那份。

---

## API

| 函数 | 作用 |
|------|------|
| `load_tokenizer(...)` | 加载 HF `AutoTokenizer`（需 `transformers`） |
| `encode(text, ...)` | 文本 -> `list[int]` |
| `decode(ids, ...)` | ids -> 文本 |
| `eos_token_id(...)` | 从 tokenizer 读 EOS（TinyLlama-Chat 一般为 2） |
| `bos_token_id(...)` | 从 tokenizer 读 BOS（一般为 1；无则 `None`） |
| `generate_text(engine, prompt, max_new, ...)` | 文本进、**新生成**文本出 |

### encode / decode

~~~
encode("Hello")  默认 add_special_tokens=True  ->  [1, 15043]   # 带 BOS
decode([15043])  默认 skip_special_tokens=True ->  "Hello"
~~~

BOS / EOS / pad **不要手写硬编码**到业务路径；从 tokenizer 读。Engine 层测试仍可用手工 id（如 `[1,2,3,4]`）做数值回归。

### generate_text

~~~python
import hf_tokenizer
import inference_engine_me
import transformer_model_me

# 1. Model + Engine（与 id 路径相同）
model = transformer_model_me.create_model(**cfg)
transformer_model_me.load_weights_from_safetensors_hf_llama(model, st_path)
engine = inference_engine_me.create_engine(model)

# 2. 文本入口
text = hf_tokenizer.generate_text(engine, "Hello", max_new_tokens=4, top_k=1, seed=0)
# text = 仅新生成段，不含 prompt

inference_engine_me.destroy_engine(engine)
transformer_model_me.destroy_model(model)
~~~

内部：

~~~
prompt_ids = encode(prompt)                    # 默认含 BOS
new_ids    = generate_loop_me.generate(
               engine, prompt_ids, max_new,
               eos_token_id=eos_token_id(tok), ...)
return decode(new_ids)                         # 只 decode 新 id
~~~

采样参数（`top_k` / `temperature` / `top_p` / `seed`）原样透传 GenerateLoop。

---

## 不做的事

- 不写 C++ Tokenizer；不改 `ie_*` / `generate_loop_run` 签名
- 骨架不做 streaming 逐 token 吐字（整段 decode 即可）
- Chat template（`<|user|>` 等）可选增强，不是模块 5 收口必须项

---

## 测试

~~~
tests/test_text_generate_e2e.py
  test_text_generate_e2e_tinyllama_slice
    - 缺切片 / tokenizer / transformers 时 skip
    - roundtrip encode/decode
    - 分步 generate + generate_text 结果一致
~~~

~~~bash
cd JJJetson_Ops
./run_tests.sh --suite test_text_generate_e2e.py --case test_text_generate_e2e_tinyllama_slice
~~~

说明：2 层 F32 切片 + greedy 的生成内容可能不可读；本测验收的是**文本链能走通**，不是对话质量。整模质量依赖 roadmap 模块 6 量化。
