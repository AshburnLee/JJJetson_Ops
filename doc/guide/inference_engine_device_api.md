# InferenceEngine Device API（Phase 2 多 layer session）

C 头文件：`src/engine/inference_engine.h`。Python 模块：`inference_engine_me`。

**生命周期、四模块边界、prefill/decode 时序**仍以 [`../design/phase2_lifecycle.md`](../design/phase2_lifecycle.md) 为准。本文回答的问题是：如果你要在 C++ 或 Python 里**亲手调 Engine**，每一步该传什么、内部会改什么、和 Phase 1 的 `TransformerRunner` 差在哪里。

---

## Engine 在整条推理链里干什么

你可以把 Phase 2 想成四块拼图：**Loader 读文件 -> Model 把权重常驻 GPU -> Engine 管一次对话的 session -> GenerateLoop 在 Engine 上反复 forward 并采样出 token**。

Engine 只管中间这块 session。它**借用**已经 create 好、并且 load 完权重的 `TransformerModel`，自己额外持有：

- 多层 `KVCache`（每一层各有一份 K/V 历史）；
- 按 `num_tokens` 分桶的 layer workspace（和 Phase 1 Runner 里的 `buffers_by_tokens[T]` 同类东西）。`create` 预热 T=1；其它 T 仍首次 forward 时懒分配；
- FA 用的 fp16 K/V staging；
- 一个 `cudaStream` 和 `cublasHandle`。

权重不在 Engine 里再拷一份。`engine_destroy` 也**不会** `destroy` Model——顺序永远是：先 destroy 所有 Engine，再 destroy Model。

如果你只做过 Phase 1：`TransformerRunner` 是 [单层 + 权重在 runner 里]；`InferenceEngine` 是 [N 层 + 权重在 Model 里 + 多 layer KV]。单 layer（N=1）时，一次 forward 的数学和 Phase 1 对齐，只是权重来源和 KV 层数不同。

---

## 调用前：Model 必须先就绪

Engine 不会帮你 load 权重。典型 Python 顺序是：

~~~python
model = transformer_model_me.create_model(**cfg)
transformer_model_me.load_weights_from_fixture(model, fixture_dir)
engine = inference_engine_me.create_engine(model)
~~~

C 里等价于 `inference_engine_create(model, stream)`，其中 `model` 必须已经 `transformer_model_load_weights` 成功。若权重未 load，`forward_*` 会打印 `model weights not loaded` 并返回 `-1`。

`stream` 传 `nullptr` 时 Engine 自建 non-blocking stream；传外部 stream 则 forward 走你的 stream（binding 里 Python 目前总是传 `nullptr`）。

---

## 一次对话在 Engine 里怎么演化

下面用**具体数字**走一遍，避免抽象说法。

假设 `max_seq_len=256`，prompt 有 3 个 token，你还想再生成 2 个新 token（GenerateLoop 的事，但 KV 语义在 Engine）。

**1. `create_engine` 之后**

- `kv_cache_len()` -> `0`
- `next_pos()` -> `0`
- KV 里没有任何历史。

**2. prefill：一次 forward 吃进 3 个 token（T=3）**

调用 `forward_token_host(..., num_tokens=3, pos_offset=0, token_ids=[a,b,c], ...)` 时，Engine 内部会把这一步的绝对位置设成 `[0,1,2]`，跑完 embed -> 第 0 层 -> ... -> 第 N-1 层 -> **一次** `advance_len(3)` -> final_norm -> lm_head。

调用返回后：

- `kv_cache_len()` -> `3`（不是 2；三步都写进 KV 了）
- `next_pos()` -> `3`（下一步 decode 应从位置 3 开始）

**3. decode：一次 forward 只吃 1 个 token（T=1）**

上一步若 greedy 采出 token `x`，下一步应：

~~~python
inference_engine_me.forward_token_host(engine, 1, kv_cache_len(engine), np.array([x], dtype=np.int32), logits_out)
~~~

这里 `pos_offset` 必须等于**当前** `kv_cache_len()`（上例里是 3），因为 decode 时 RoPE 和 FA 的 `num_kv_tokens` 都依赖 [历史已有 L 个 token，本步再 append 1 个]。

返回后 `kv_cache_len()` -> `4`。

**4. 开新对话：`reset_engine`**

只把 KV 逻辑长度清零，**不**释放 buffer pool，**不**动 Model 权重：

- `kv_cache_len()` -> `0`
- `next_pos()` -> `0`

然后可以重新 prefill。

**5. 结束：`destroy_engine`**

释放 KV、pool、stream；Model handle 仍然有效，可再 `create_engine` 复用同一份权重。

---

## 单步 forward 内部在做什么

无论走哪条 API，核心都是 `inference_engine_forward_device`。用 prefill、`T=3`、`cache_len` 从 0 开始举例，顺序是：

1. **入口 hidden**
   - 若 ctx 带了 `d_token_ids`：调 Model 的 embed，得到 `[hidden_size, 3]` 的 hidden。
   - 若 ctx 带了 `d_hidden_in`：**跳过 embed**，把你给的 hidden 拷进 layer workspace（测试路径）。

2. **N 个 Transformer block**
   对 `layer_idx = 0 .. N-1` 各调一次 `transformer_layer_linears_forward_device`。
   每一层都会 append 本步的 K/V 到 `KVCache[layer_idx]`，但**此时还不改** `cache_len`——三层 block 都跑完，KV 里才 [逻辑上准备好写入 3 个位置]。

3. **`kv_cache_advance_len(3)`**
   全 layer 完成后**只 advance 一次**。这是和 [每层 advance] 的错误做法的分界线。

4. **`final_norm`**
   用 Model 上的 `d_w_final_norm` 作用在最后一层输出上。

5. **可选 `lm_head`**
   若 ctx 的 `d_logits` 非空，再投影到 `[vocab_size, T]`。
   `forward_hidden_host` 不传 logits，所以只到 final_norm 的 hidden。
   `forward_token_host` 会要 logits。

整条 forward 结束时会 `cudaStreamSynchronize`，所以 host 测试 API 返回时 GPU 已写完。

---

## 三条 forward 入口，该用哪条

Engine 对外有三层 [包装厚度] 不同的入口。名字里带 `_host` 的都是**测试用**：内部会 `cudaMalloc` 输入/输出 buffer，跑完再 free——**不要**当生产热路径。

### `inference_engine_forward_device` — 生产核心

你自己在 GPU 上准备好 `d_token_ids` 或 `d_hidden_in`、`d_pos`、`d_hidden_out`、可选 `d_logits`，栈上填一个 `InferenceForwardCtx`，调这一层。

GenerateLoop 的生产路径不走这个直接暴露的 ctx，而是走 `forward_token_sample`（内部 `last_logits` -> `forward_token_device` -> `forward_device`）。token / logits / hidden / out_token 来自 create 时的 BufferPool。

### `inference_engine_forward_hidden_host` — 从 hidden 进，跳过 embed

名字里的 `_hidden_` 就是在强调：**这一步不从 token 开始**，调用方已经有一份 hidden 状态。

典型用途：像 Phase 1 一样，用随机 hidden 或 ref hidden 单测 [N 层 block + KV + final_norm]，而不想牵扯 embed/lm_head。

Python：

~~~python
hidden = np.asfortranarray(np.random.randn(HIDDEN_SIZE, T, 1, 1).astype(np.float32))
out = np.zeros_like(hidden)
inference_engine_me.forward_hidden_host(engine, T, pos_offset, hidden, out)
~~~

输出 `out` 与 `hidden` 同 shape：`[hidden_size, T, 1, 1]`，是 final_norm 之后、**lm_head 之前**的 hidden。

### `inference_engine_forward_token_host` — 从 token 进，到 logits 出

这是完整 [token 环] 的单步测试 API：`token_ids` -> embed -> N 层 -> final_norm -> lm_head -> host logits。

Python：

~~~python
token_ids = np.array([3, 17, 42], dtype=np.int32)
T = len(token_ids)
logits = np.zeros((VOCAB_SIZE, T), dtype=np.float32, order="F")  # 注意 Fortran order
inference_engine_me.forward_token_host(engine, T, 0, token_ids, logits)
~~~

`logits[v, t]` 是第 `t` 个输入 token 位置上、词表第 `v` 维的 logit。GenerateLoop 在 decode 时只关心**最后一列** `logits[:, T-1]` 做 greedy；这个 API 把整表都 D2H 回来，方便和 ref 逐列对比。

### `inference_engine_forward_token_device` — token 环的 GPU 侧单步

`d_token_ids`、`d_logits` 已在 device 上；`d_hidden_out` 用 Engine BufferPool（create 时按 max_seq 分配），调 `forward_device`。
GenerateLoop 不直接调本函数，走 `forward_token_sample`（内部再进 `last_logits`，再进这里）。

`ENABLE_NVTX` 打开时，本函数钉一块 `forward`。

### `inference_engine_forward_token_last_logits` — 单步进 GPU，末列留在 device

host `token_ids` 进，H2D 写入 pool `d_token_ids`，forward 写入 pool `d_logits`。**不每步 malloc**。
成功后末列在 GPU 上：`d_logits + vocab*(T-1)`。给测试或 [先 forward 再自己采样] 用。
GenerateLoop 生产步进不直接调本函数，走下面的 `forward_token_sample`。

例：`token_ids_host=[3,17,42]`，T=3，pos=0 -> 一次 H2D 3 个 int，logits 表 `[vocab,3]` 落在 pool 前 3 列。
随后 decode T=1 三次，仍用同一块 pool，只覆盖第 0 列。

### `inference_engine_forward_token_sample` — GenerateLoop 生产步进

在 `last_logits` 之后，用 pool 的 `d_out_token` 在末列上采样，再 D2H **一个** token id。
Python **不暴露**（测试走 `generate_loop_me.generate`）。

例：prompt `[3,17,42]` T=3 -> 返回 x0；下一步 ids=`[x0]` T=1 -> 返回 x1。
vocab 从 Model cfg 读，调用方不用传。

---

## 张量 layout（传错 order 是最常见的坑）

本仓库 hidden 和 logits 的 [列] 是 **token 维**，并且用 **Fortran（列主序）** 存储。

**Hidden**（`forward_hidden_host`）
逻辑 shape：`[hidden_size, num_tokens, 1, batch]`，batch 目前固定 1。
在内存里等价于列主序的 `[hidden_size, num_tokens]`：第 `t` 列是第 `t` 个 token 的 hidden 向量。

**Logits**（`forward_token_host`）
shape：`[vocab_size, num_tokens]`，同样列主序。
第 `t` 列对应输入里第 `t` 个 token 位置的词表 logits。

**token_ids**
一维 `int32`，长度等于本步 `num_tokens`；C 风格行主序即可。

**pos_offset 与 d_pos**
测试 API 不传 `d_pos` 指针；Engine 根据 `pos_offset` 自动生成 `[pos_offset, pos_offset+1, ..., pos_offset+T-1]` 并 H2D。
这是**绝对** token 位置，不是相对步数。prefill 从 0 开始；decode 必须传当前 `kv_cache_len()`。

用小例子核对 layout：
`hidden_size=4`, `T=2` 时，Fortran `hidden[4,2]` 在内存里顺序是
`h[0,0], h[1,0], h[2,0], h[3,0], h[0,1], h[1,1], h[2,1], h[3,1]`。
Python 用 `order="F"` 的 `(4,2,1,1)` 或 `(4,2)` 与之一致。

---

## `InferenceForwardCtx` 各字段含义

生产路径在 C++ 里每步栈上填这个 struct，**不持久化**：

~~~
字段            谁提供          说明
──────────────────────────────────────────────────────────────────
num_tokens      调用方          本步 T；prefill 常 >1，decode 常 =1
stream          调用方          nullptr 则用 engine 自带 stream
d_token_ids     调用方（GPU）   与 d_hidden_in 二选一；有则走 embed
d_hidden_in     调用方（GPU）   与 d_token_ids 二选一；测试跳过 embed
d_hidden_out    调用方（GPU）   必填；[hidden_size, T] col-major
d_pos           调用方（GPU）   必填；长度 T 的绝对位置
d_logits        调用方（GPU）   可选；非空则写 lm_head 输出
~~~

若两个入口指针都为空，或 `d_pos` 为空，返回 `-1`。

---

## C API 速查（附语义）

~~~
函数                                    返回值          你调用它时期望发生什么
────────────────────────────────────────────────────────────────────────────────────────
inference_engine_create(model, stream)   Engine*/null    挂 N 层 KV、pool、stream；cache_len=0
inference_engine_destroy(engine)        void            释放 session 资源；不碰 model
inference_engine_reset(engine)            void            KV 清零；pool 保留
inference_engine_kv_cache_len(engine)     int             当前已 advance 的序列长度 L
inference_engine_next_pos(engine)       int             与 L 同步；下一步 decode 的 pos_offset
inference_engine_forward_device(...)    0 / -1          生产单步 forward
inference_engine_forward_hidden_host(...) 0 / -1        测试：hidden 进，hidden 出
inference_engine_forward_token_host(...)  0 / -1        测试：token 进，logits 出（走 pool）
inference_engine_forward_token_device(...) 0 / -1       GPU token 单步；hidden 用 pool
inference_engine_forward_token_last_logits(...) 0 / -1  生产步进：host token -> pool + forward
inference_engine_d_logits_last(engine)    float*        pool 上末列 logits（采样用）
inference_engine_d_out_token(engine)      int*          pool 上 1 个 token 槽（采样输出）
inference_engine_get_model / ...        指针            集成或调试
~~~

**常见失败原因**

- `cache_len + T > max_seq_len`：这一步会超出模型最大上下文，stderr 打 `exceeds max_seq_len`，返回 `-1`。
- 权重未 load：forward 拒绝执行。
- decode 时 `pos_offset` 与真实 `cache_len` 不一致：不会一定报错，但 FA/RoPE 语义错，数值会和 ref 对不上——测试里务必 `pos_offset=kv_cache_len(engine)`。

---

## Python API（`inference_engine_me`）

Python 用 `uintptr_t` 传递 handle，**没有 RAII**；必须 `create_engine` / `destroy_engine` 配对。

~~~
Python                                              对应 C                         用途
────────────────────────────────────────────────────────────────────────────────────────────
create_engine(model_handle)                         inference_engine_create        开始 session
destroy_engine(engine_handle)                       inference_engine_destroy       结束 session
reset_engine(engine_handle)                         inference_engine_reset         新对话
kv_cache_len(engine_handle)                         inference_engine_kv_cache_len  断言 L
next_pos(engine_handle)                             inference_engine_next_pos      断言下一位置
kv_cache_num_layers(engine_handle)                  kv_cache_get_num_layers        层数 N
forward_hidden_host(engine, T, pos, h_in, h_out)    forward_hidden_host            测试 block 链
forward_token_host(engine, T, pos, ids, logits)     forward_token_host             测试 token 环
~~~

参数错误时 binding 抛 `RuntimeError`（shape/order 不对或 C 返回 -1）。

---

## 和 GenerateLoop 怎么配合

[`generate_loop_me.generate`](../../src/bindings/generate_loop_binding.cpp) **不**经过 `inference_engine_me.forward_token_host`。它在 C++ 里循环调用 `inference_engine_forward_token_sample`，只拿回 **token id**，并处理 EOS。

关系可以这么记：

- **`forward_token_host`**：你手动控每一步，拿**完整** logits 表做数值对比。
- **`forward_token_last_logits`**：单步进 GPU，末列留在 device（测试或自采）。
- **`generate_loop_run`**：帮你把 prefill + decode 循环写完，每步调 `forward_token_sample`。

两者底层 forward 语义一致；GenerateLoop 不 create/destroy Engine，调用前你必须已有 load 好权重的 Model 和 create 好的 Engine。

生产化：token/logits/hidden/out_token 在 Engine BufferPool（create 一次）。GenerateLoop 只留循环 + stop；采样 CUDA 在 `forward_token_sample`。并行 top-k/top-p 仍是 TODO(perf-*)。

---

## 与 Loader / Model 的关系（同文档分节）

Engine 文档以 session 为主；上游只需知道：

**Loader**（`weight_loader_me` / fixture）
把 checkpoint 读成 host 上的 name->tensor，**不参与 GPU session**。

**Model**（`transformer_model_me`）
`create_model` 分配 GPU 权重容器；H2D 有两条入口：

~~~cpp
load_weights_from_fixture(model, fixture_dir)
    Loader: load_fixture -> transformer_model_load_weights

load_weights_from_safetensors_hf_llama(model, path/to/model.safetensors)
    Loader: load_safetensors_hf_llama -> transformer_model_load_weights
    同目录需 config.txt，或 HF config.json（切片后 num_hidden_layers 须等于实际层数）
~~~

`ENABLE_NVTX` 打开时：Loader 钉 `load_fixture` / `load_safetensors`（host 读盘），Model 钉 `load_weights`（H2D）。两条色块前后相接，中间不应再空白。
embed / lm_head / final_norm / 各层权重都挂在 Model 上；Engine forward 时按 `layer_idx` 向 Model 要指针。

典型 ownership：`Loader` 输出 host 数据 -> `Model` 拥有 GPU 权重 -> `Engine` 引用 Model + 拥有 KV -> `GenerateLoop` 引用 Engine -> 销毁顺序反过来。

更细的 Model API 见 `tests/test_transformer_model_*.py` 与 phase2 lifecycle 第 2 节。

---

## 测试里怎么构造输入

**`forward_hidden_host`**（`tests/test_inference_engine_forward.py`）
hidden 直接 `np.random.randn(HIDDEN_SIZE, T, 1, 1)`，Fortran order；ref 用 `chain_linear_me_ref` 叠 layer，不经过 embed。

**`forward_token_host`**（`tests/test_inference_engine_forward_token.py`）
token 不需要真实文本；`np.random.randint(0, VOCAB_SIZE, size=T)` 即可。
ref 先用 fixture 里的 `embed` 表查 token 得到 hidden，再走同一套 layer ref，最后用 `lm_head` 权重得到 logits。

**GenerateLoop e2e**（`tests/test_generate_loop.py`）
同样用随机 token id 作 prompt；断言生成个数和 `kv_cache_len == len(prompt) + num_generated - 1`（prefill 一步产出第一个新 token 的 logits 时，KV 已含 prompt 全部 token）。

---

## 和 Phase 1 TransformerRunner 对照

若你熟悉 [`transformer_runner_device_api.md`](transformer_runner_device_api.md)：

~~~
概念              Phase 1 Runner                    Phase 2 Engine
────────────────────────────────────────────────────────────────────────────
权重在哪          runner create 时 H2D              Model 上，Engine 只引用
层数              固定 1                            cfg.num_layers
KV                单层                              KVCache(num_layers=N)
单步核心          transformer_runner_forward_device inference_engine_forward_device
final_norm        无（Phase 1 block 外）            Engine 内建
embed / lm_head   无                                Model + forward 可选
测试 hidden 路径  forward_host                      forward_hidden_host
测试 token 路径   无                                forward_token_host
~~~

N=1 且不走 embed/lm_head 时，hidden 路径的数值行为应与 Phase 1 同权重设定下对齐（见 `test_inference_engine_forward.py`）。

---

## 相关文件

~~~
类型              路径
────────────────────────────────────────────────────────────
实现              src/engine/inference_engine.{h,cpp}
Python binding    src/bindings/inference_engine_binding.cpp
生命周期图纸      doc/design/phase2_lifecycle.md（第 3 节）
hidden 路径测试   tests/test_inference_engine_forward.py
token 路径测试    tests/test_inference_engine_forward_token.py
生成环测试        tests/test_generate_loop.py
~~~
