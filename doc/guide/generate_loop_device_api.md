# GenerateLoop / Sampler Device API（Phase 2 token 出环）

C 头文件：`src/engine/generate_loop.h`，实现：`src/engine/generate_loop.cpp`。Python 模块：`generate_loop_me`。

**四模块边界、GenerateLoop 在 lifecycle 里的位置**见 [`../design/phase2_lifecycle.md`](../design/phase2_lifecycle.md) 第 4 节。Engine 单步 forward 语义见 [`inference_engine_device_api.md`](inference_engine_device_api.md)。本文说明：**在已有 Engine session 上，如何把 prompt 跑成一串新 token**，以及 Sampler 各策略的契约（含尚未实现的 temperature / top-k / top-p）。

---

## GenerateLoop 和 Sampler 分别管什么

Engine 做完一步 forward 后，GPU 上有一份 logits（或 GenerateLoop 骨架期只 D2H 末 token 那一列）。**Sampler** 的工作是把 `[vocab]` 的 logits 变成一个 `next_token_id`。**GenerateLoop** 则负责 [何时 prefill、何时 decode、循环几次、什么时候停]——它调用 Engine 拿 logits，再调 Sampler 拿 token，把新 token 喂回下一步 decode。

GenerateLoop **不** create/destroy Engine，**不**持有 KV 或权重。调用方必须先：

~~~python
model = transformer_model_me.create_model(**cfg)
transformer_model_me.load_weights_from_fixture(model, fixture_dir)
engine = inference_engine_me.create_engine(model)
out = generate_loop_me.generate(engine, prompt_ids, max_new_tokens)
~~~

GenerateLoop 只借用 `InferenceEngine*`；session 里 KV 的增长、reset、destroy 仍由 Engine API 管。

---

## 一次 generate 在内部怎么跑（具体数字）

假设 prompt 长度 `prompt_len=3`，`max_new_tokens=2`，`eos_token_id=-1`（不启用 EOS 早停），`cache_len` 从 0 开始。

**Step A — prefill（T=3）**

`forward_token_step` 把 `prompt_token_ids[0..2]` H2D，调 `inference_engine_forward_token_device(..., num_tokens=3, pos_offset=0)`，再 D2H **最后一列** logits `logits_last[vocab]`（不是整表 `[vocab,3]`）。

此时 Engine 里 `kv_cache_len` 变为 3。

**Step B — 第一个新 token**

`next = sampler_greedy_host(logits_last, vocab_size)`，写入 `out[0]`。

**Step C — decode 第 1 次（T=1）**

用 `next` 作为唯一输入 token，`pos_offset = kv_cache_len()`（此时为 3），再 forward -> D2H 末列 logits -> greedy -> `out[1]`。

此时 `kv_cache_len` 变为 4。公式：`len(prompt) + num_generated - 1`（prefill 后已含 prompt 全长，decode 每步 +1）。

**Step D — 停止**

- 已生成 `max_new_tokens` 个 -> 返回；
- 或某步 `next == eos_token_id`（且 `eos_token_id >= 0`）-> 提前返回，已生成长度可能小于 `max_new_tokens`。

骨架期 `forward_token_step` 每步 `cudaMalloc` token/logits buffer，属于过渡实现；生产化迁入 Engine + BufferPool（lifecycle 第 4.4 节）。

---

## 末 token logits slice（已实现）

prefill 时 Engine 一次产出 `[vocab, T]` 的 logits，但采样只需要**最后一个输入位置**上的分布（预测 [下一个 token]）。
`generate_loop.cpp` 里 `forward_token_step` 在 D2H 时只拷贝：

~~~
d_logits_last = d_logits + vocab_size * (num_tokens - 1)
~~~

decode 时 `T=1`，末列即唯一列，prefill 与 decode 路径统一走 [末列 slice -> Sampler]。

---

## Sampler API

Sampler 当前全部在 **host** 上运行（logits 已在 skeleton 期 D2H）。输入都是长度 `vocab_size` 的 `float` 向量；输出为 token id，失败返回 `-1`。

### `sampler_greedy_host`（已实现）

对 logits 做 argmax，取最大值对应的 vocab 下标。

~~~c
int sampler_greedy_host(const float *logits, int vocab_size);
~~~

Python 不单独暴露；由 `generate_loop_me.generate` 内部使用。

**测试**：`tests/test_generate_loop.py`（间接覆盖）；单步 logits 对比见 `tests/test_inference_engine_forward_token.py`。

### `sampler_temperature_host`（规划，未实现）

对 logits 除以 `temperature`，再 softmax + 按概率采样（或 greedy 当 `temperature->0`）。
规划签名（实现前可能微调）：

~~~c
// temperature <= 0 视为非法；=1 等价于原始分布
int sampler_temperature_host(const float *logits, int vocab_size, float temperature,
                             uint64_t seed);
~~~

roadmap 模块 4 细节独立条目；实现时需补 host 单测 + generate 参数透传。

### `sampler_top_k_host` / `sampler_top_k_device`（已实现）

CUDA 算子：`src/cuda/sampler_top_k.{h,cu}`。在 **device logits [vocab]** 上做 top-k + softmax 采样；`top_k == 1` 走并行 greedy kernel。

~~~c
int sampler_top_k_device(void *stream, const float *d_logits, int vocab_size, int top_k,
                         uint64_t seed, int *d_out_token);
int sampler_top_k_host(const float *logits_host, int vocab_size, int top_k, uint64_t seed);
~~~

GenerateLoop 过渡路径：`forward_token_step` 在 GPU 末列 logits 上调用 `sampler_top_k_device`，只 D2H **token id**（不再 D2H 整段 vocab）。

Python 测试：`generate_loop_me.sampler_top_k_host`（内部 H2D -> device -> token）。

**TODO(perf-topk)** — `top_k > 1` 的性能债（代码内同 tag，便于 grep）：

- 现状：`sampler_top_k_kernel` 使用 `<<<1, 1>>>`，仅 1 个 thread 顺序扫完整个 vocab，再 softmax + 采样。功能正确，但 GPU SM 几乎空闲；大词表连续 decode 时 launch 开销 + O(vocab) 单线程会成为瓶颈。
- `top_k == 1` 不受影响：走 `sampler_greedy_kernel`（256 线程并行 argmax）。
- 目标：多 thread 分块维护 local top-k 再 block merge；或 CUB/thrust select-k；长期对齐 vLLM/SGLang 的 fused top-k + top-p + sample kernel。
- 相关文件：`src/cuda/sampler_top_k.cu`（kernel + launch）、`src/engine/generate_loop.cpp`（调用点）；roadmap 模块 4 生产化 / Phase 2.6。

### `sampler_top_p_host`（规划，未实现）

nucleus sampling：按概率质量累加到 `top_p`，在最小集合上采样。

~~~c
int sampler_top_p_host(const float *logits, int vocab_size, float top_p, uint64_t seed);
~~~

### 策略组合（规划）

生产常见组合：`temperature` + `top_k` / `top_p`。top-k 已接入 `generate_loop_run`（`top_k` 默认 1）；temperature / top-p 待实现。

---

## C API：`generate_loop_run`

~~~c
int generate_loop_run(InferenceEngine *engine,
                      const int *prompt_token_ids,
                      int prompt_len,
                      int max_new_tokens,
                      int eos_token_id,
                      int top_k,
                      uint64_t seed,
                      int *out_token_ids,
                      int out_capacity);
~~~

**参数**

~~~
参数              含义
──────────────────────────────────────────────────────────────────
engine            已 create、权重已 load 的 Engine；GenerateLoop 不 reset
prompt_token_ids  host 上 1-D prompt，长度 prompt_len
prompt_len        prompt token 个数，必须 > 0
max_new_tokens    最多生成几个新 token（不含 prompt）
eos_token_id      >=0 时遇该 id 早停；<0 表示禁用
top_k             >0；1 为 greedy；>1 为 top-k 采样
seed              传给 mt19937_64；0 用内部 mix 种子
out_token_ids     输出缓冲区；至少 max_new_tokens
out_capacity      容量；须 >= max_new_tokens
~~~

**返回值**

- 成功：写入的新 token 个数 `n`（`1 <= n <= max_new_tokens`）；
- 失败：`-1`（null 指针、权重未 load、forward 失败、超 `max_seq_len` 等）；原因见 stderr。

**边界**

- `prompt_len + max_new_tokens - 1` 不得超过 `ModelConfig.max_seq_len`；
- 不在内部调 `engine_reset`；若需新对话，调用方先 `inference_engine_reset`。

---

## Python API：`generate_loop_me`

~~~python
generate(engine_handle, prompt_token_ids, max_new_tokens, eos_token_id=-1, top_k=1, seed=0) -> list[int]
~~~

- `prompt_token_ids`：非空 1-D `int32` numpy array；
- `max_new_tokens <= 0`：返回空 list（不报错）；
- 失败：抛 `RuntimeError`。

返回的是**新生成** token 列表，不含 prompt。KV 状态留在 Engine 内，可用 `inference_engine_me.kv_cache_len(engine)` 断言。

---

## 与 Engine 的分工（调用关系）

~~~
调用方
  │  create Model + load + create Engine
  ▼
generate_loop_me.generate(engine, prompt, max_new, eos)
  │
  ├─ prefill: forward_token_step -> inference_engine_forward_token_device
  ├─ sampler_top_k_device(d_logits_last, top_k, seed)
  └─ decode × (max_new-1): forward_token_step(T=1, pos=kv_cache_len)
       └─ sampler_top_k_device(...)

Engine 细节（pos_offset、KV、layout）不在本文重复；见 inference_engine_device_api.md。
~~~

GenerateLoop **不**调用 `inference_engine_forward_token_host`；host token 单步测试走 Engine binding，GenerateLoop 走 device 路径 + 局部 H2D/D2H glue。

---

## 停止条件与 KV 末态

**max_new_tokens**
硬上限：循环最多写出这么多个新 token。

**EOS**
`eos_token_id >= 0` 时，某步采样结果等于该 id 立即停止；**该 EOS token 计入**已生成列表。
测试做法：先 `generate(..., 1)` 拿到第一个 token 当作 `eos_id`，`reset_engine` 后再 `generate(..., 10, eos_token_id=eos_id)`，应只返回 1 个 token 且 `kv_cache_len == len(prompt)`。见 `tests/test_generate_loop.py`。

**KV 长度**
prefill + 生成 `n` 个新 token 后（未 EOS 截断）：`kv_cache_len == prompt_len + n - 1`。
EOS 在第 1 个新 token 就触发时：`kv_cache_len == prompt_len`（decode 未再跑）。

---

## 测试与 ref

~~~
场景                          测试文件
────────────────────────────────────────────────────────────
generate binding e2e          tests/test_generate_loop.py
Engine 单步 token->logits     tests/test_inference_engine_forward_token.py
hidden 路径（无 GenerateLoop） tests/test_inference_engine_forward.py
~~~

GenerateLoop e2e 目前用随机 prompt token id，不依赖真实 tokenizer；与 Engine token 测试同一 fixture 权重策略。

---

## 生产化（未做，不在本文 API 承诺内）

骨架期限制：

- `forward_token_step` 每步 device malloc / free；
- Sampler 已在 GPU 末列 logits 上采样，但 **top_k>1 为 TODO(perf-topk) 单线程 kernel**（见上文）；
- temperature / top-p 未实现。

收工方向见 lifecycle 第 4.4 节：`inference_engine_forward_token_last_logits`、session 级 buffer 池、GenerateLoop 只保留循环 + stop、**并行 top-k sampler**。

---

## 相关文件

~~~
类型              路径
────────────────────────────────────────────────────────────
实现              src/engine/generate_loop.{h,cpp}
Python binding    src/bindings/generate_loop_binding.cpp
生命周期          doc/design/phase2_lifecycle.md（第 4 节）
Engine API        doc/guide/inference_engine_device_api.md
e2e 测试          tests/test_generate_loop.py
~~~
