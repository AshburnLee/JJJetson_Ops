# TransformerRunner Device API（Phase 1 单层）

C 头文件：`src/engine/transformer_runner.h`。Python：`transformer_runner_me`。

**生命周期、ownership、prefill/decode 时序**见 [`lifecycle.md`](lifecycle.md)。本文补充 **API 视角的对象关系与数据流**。

---

## 对象关系（API 视角）

~~~
调用方（Engine / 测试）
    │
    │  每步栈上
    ▼
TransformerRunnerForwardCtx          ephemeral；不持有 GPU 内存
    · num_tokens, stream
    · d_hidden_in / d_hidden_out / d_pos  ──► 调用方或 binding 分配
    │
    │  transformer_runner_forward_device(runner, &ctx)
    ▼
TransformerRunner                      session 边界；create 一次，destroy 释放
    ├── d_w_* ×9                       只读权重（create 时 H2D）
    ├── RopeCosSinCache                只读 cos/sin 表
    ├── KVCache                        cache_len；历史 K/V float cache
    ├── d_k_fa_fp16 / d_v_fa_fp16      每步 cast 覆写；FA 读 K/V
    ├── buffers_by_tokens[T]           layer workspace（懒分配）
    │       · d_hidden, d_q, d_k, d_v, d_attn_out, …
    │       · d_q_fp16                 本步 Q staging
    └── stream / cublas_handle

Python：create_runner → uintptr_t handle ──► 同一 TransformerRunner*
        destroy_runner / kv_cache_reset / forward_* 均传 handle
~~~

**谁传什么**

| 对象 | 谁创建 | 谁提供指针给 forward |
|------|--------|----------------------|
| `TransformerRunner` | `create` / `create_runner` | — |
| `ForwardCtx` | 调用方每步填充 | 调用方 |
| `d_hidden_in/out`, `d_pos` | 生产：Engine GPU 池；测试：binding H2D 临时 malloc | 写入 ctx |
| `buffers[T]`, KV cache, FA staging | runner 内部 | 不暴露给调用方 |

---

## 单步 forward 数据流

`transformer_runner_forward_device` 一次调用（layer 内 + layer 外）：

~~~
ctx.d_hidden_in ──D2D──► buffers[T].d_hidden
                              │
         ┌────────────────────┼────────────────────────────────────┐
         ▼                    ▼                                    │
   Pre-Attn RMSNorm+fused   Q/K/V Linear + RoPE (查 RopeCosSinCache) │
         │                    │                                    │
         │              d_q ──► q pack ──► d_q_fp16  (本步 T)       │
         │              d_k/d_v flat [kv_dim,T]  (workspace)       │
         │                    │                                    │
         │                    ├── append ──► KVCache [L,L+T)       │
         │                    │              (不改 cache_len)       │
         │                    ├── cast ──► d_k/v_fa_fp16 [L+T]     │
         │                    └── FA(d_q_fp16, d_k/v_fa_fp16)      │
         │                              ──► d_attn_out              │
         │                    O Linear + Pre-FFN + SwiGLU FFN      │
         │                    Post-FFN add ──► buffers.d_hidden_out│
         └────────────────────┴────────────────────────────────────┘
                              │
buffers.d_hidden_out ──D2D──► ctx.d_hidden_out
                              │
                    kv_cache_advance_len(T)   ◄── layer 外；cache_len += T
~~~

**两块 K/V（数据流分叉点）**

~~~
buffers->d_k / d_v          本步 Linear 输出，每 forward 覆盖
       │
       └── append ──►  KVCache.layers[0]     历史累积，session 级
                              │
                              └── cast ──► d_k_fa_fp16 / d_v_fa_fp16 ──► FA
~~~

---

## prefill / decode 调用流（API 层）

~~~
create_runner / transformer_runner_create
  cache_len = 0

── prefill (T=13) ──────────────────────────────────────────────
  forward_host(runner, 13, pos_offset=0, h_in, h_out)
    或 forward_device(..., pos=[0..12])
      ctx: num_tokens=13, d_pos 绝对位置 0..12
      → forward_device → advance_len(13)
  kv_cache_len == 13

── decode (T=1) ───────────────────────────────────────────────
  forward_host(runner, 1, pos_offset=13, h_in, h_out)
    或 forward_device(..., pos=[13])
      ctx: num_tokens=1, d_pos=[13]
      layer 内 FA: num_q_tokens=1, num_kv_tokens=14
      → advance_len(1)
  kv_cache_len == 14

── 新对话（可选）── kv_cache_reset → cache_len=0 → 再 prefill

destroy_runner
~~~

---

## 分层

~~~
生产（C++ 集成 / Engine）
  transformer_runner_forward_device(runner, &ctx)   ctx 指针已在 GPU

测试（Python / numpy）
  forward_host  → transformer_runner_test（H2D + auto d_pos + D2H）
  forward_device → binding 内临时 cudaMalloc I/O，再调 forward_device
~~~

---

## Layout

| 张量 | Layout | 说明 |
|------|--------|------|
| `hidden_in` / `hidden_out` | col-major `[hidden_size, num_tokens, 1, batch]` fp32 | batch=1 |
| `d_pos` | `[num_tokens]` int32 device | **绝对** token 位置；prefill `0..T-1`，decode 通常 `[cache_len]` |
| 权重 `w_*` | create 时 host → device 一次 H2D | 9 个：`q,k,v,o,gate,up,down,input_ln,post_ln` |

---

## `TransformerRunnerForwardCtx`

| 字段 | 类型 | 约束 |
|------|------|------|
| `num_tokens` | int | 本步 T；prefill T>1，decode 通常 T=1 |
| `stream` | void* | `nullptr` → 用 runner 自建 stream |
| `d_hidden_in` | const float* | GPU；`[hidden_size, T, 1, 1]` |
| `d_hidden_out` | float* | GPU；同 shape |
| `d_pos` | const int* | GPU；长度 T；**不可为 null** |

每步 forward 栈上填充，**不持久化**（见 `lifecycle.md` §2 对象表）。

---

## C API

| API | 返回值 | 作用 |
|-----|--------|------|
| `transformer_runner_create(...)` | `TransformerRunner*` / `nullptr` | 分配 runner、权重 H2D、KVCache、FA staging |
| `transformer_runner_destroy(runner)` | void | 释放 session 全部 GPU 资源 |
| `transformer_runner_forward_device(runner, ctx)` | 0 / -1 | 生产单步 forward；末尾 `advance_len(T)` |
| `transformer_runner_test(runner, hidden_in, hidden_out, num_tokens, pos_offset)` | 0 / -1 | 测试：H2D → forward → D2H；`d_pos[t]=pos_offset+t` |
| `transformer_runner_kv_cache_len(runner)` | int | 当前 `cache_len`（advance 之后） |
| `transformer_runner_kv_cache_reset(runner)` | void | `cache_len→0`，不 free buffer |
| `transformer_runner_buffers_get(runner, num_tokens)` | buffers* | 按 T 懒分配 layer workspace（Graph 复用） |
| `transformer_layer_linears_forward_device(...)` | void | 单层 Pre-LN + FA + FFN；**不** advance_len |

### `transformer_runner_create` 参数

| 参数 | 含义 |
|------|------|
| `hidden_size`, `intermediate_size` | 模型维 |
| `num_q_heads`, `num_kv_heads`, `head_dim` | Attention；GQA 支持 |
| `max_seq_len`, `freq_base` | KV cache 容量 + RoPE 表 |
| `w_*_host` ×9 | host 权重；内部 cudaMalloc + H2D |
| `stream` | 外部 stream；`nullptr` 时 runner 自建 |

失败：`stderr` 打印原因，返回 `nullptr`（invalid shape / null weight / 子对象 create 失败）。

### `transformer_runner_forward_device` 错误

| 条件 | 返回 |
|------|------|
| `runner` / `ctx` null | -1 |
| `ctx->d_pos` null | -1 |
| `kv_cache_advance_len` 失败（如超 `max_seq`） | -1 |
| 成功 | 0（末尾 `cudaStreamSynchronize`） |

Layer 内算子失败：`transformer_layer_linears_forward_device` 打 stderr 后 **return**（void），但 runner 仍可能 advance_len——当前测试路径假定 layer 成功。

---

## Python API（`transformer_runner_me`）

| Python | C | 路径 |
|--------|---|------|
| `create_runner(...)` → `uintptr_t` | `transformer_runner_create` | session 开始 |
| `destroy_runner(handle)` | `transformer_runner_destroy` | session 结束 |
| `forward_host(handle, num_tokens, pos_offset, hidden_in, hidden_out)` | `transformer_runner_test` | **测试** |
| `forward_device(handle, num_tokens, hidden_in, hidden_out, pos)` | `transformer_runner_forward_device` | binding 带 H2D 的**测试包装** |
| `kv_cache_len(handle)` | `transformer_runner_kv_cache_len` | 断言 |
| `kv_cache_reset(handle)` | `transformer_runner_kv_cache_reset` | 新对话 |

Python handle 无 RAII，必须配对 `destroy_runner`。

**生产集成**：C/C++ 直接调 `transformer_runner_forward_device`，hidden/pos 已在 GPU；不要依赖 Python `forward_device` 的每步 cudaMalloc。

---

## 数值 ref（测试）

| 场景 | ref | 测试 |
|------|-----|------|
| prefill 单步 | `chain_linear_me_ref`（`KvCacheRef` 空 cache） | `test_transformer_runner.py` |
| decode 步 | `chain_linear_me_ref_step` + 累积 `KvCacheRef`，FA `num_kv_tokens=L+T` | `test_forward_device_decode_matches_kv_ref` |
| device vs host | 同 runner 双路径 | `test_transformer_runner_forward_device.py` |

---

## KV cache：float → fp16 layout 索引

实现：`src/cuda/kv_cache.cu`。三块 buffer 角色见 `lifecycle.md`「两块 K/V」表。

### 1. append：workspace flat → cache

**src**（本步 RoPE 后）：flat col-major `[kv_dim, T]`，`kv_dim = head_dim × num_kv_heads`

**dst**（session cache）：`[head_dim, max_seq, num_kv_heads, 1]` float

写入 slot `[offset, offset + T)`，`offset = cache_len`（append 时不改 len）。

~~~
(d, h, t_local) 在 src flat:
  src_idx = d + head_dim * h + kv_dim * t_local

同一逻辑元素在 cache slot t = offset + t_local:
  dst_idx = d + head_dim * t + head_dim * max_seq * h

即 cache[d, t, h, 0] = src_flat[d + head_dim*h, t_local]
~~~

### 2. cast_fp16：cache → FA staging

**src**：cache 前 `num_kv_tokens` 个 slot（`num_kv_tokens = L + T`）

**dst**：`d_k_fa_fp16` / `d_v_fa_fp16`，`[head_dim, num_kv_tokens, num_kv_heads, 1]` fp16

~~~
线性下标 i → (d, t, h):
  d = i % head_dim
  t = (i / head_dim) % num_kv_tokens
  h = i / (head_dim * num_kv_tokens)

cache 读：
  cache_idx = d + t * head_dim + h * head_dim * max_seq

dst 写（FA layout，与 Q pack 一致）：
  dst[i] = fp16(cache[cache_idx])
~~~

**decode 例**：prefill 后 `L=13`，本步 `T=1` → cast 读 `t=0..13` 共 14 slot，FA `num_kv_tokens=14`，`num_q_tokens=1`。

### 3. 与 FA Q 的 shape 对照

| 张量 | layout | prefill T=13 | decode L=13,T=1 |
|------|--------|--------------|-----------------|
| Q fp16 | `[head_dim, num_q_tokens, num_q_heads, 1]` | 13 | 1 |
| K/V fp16 | `[head_dim, num_kv_tokens, num_kv_heads, 1]` | 13 | **14** |

FA：`fa_double_buffer_forward_device`，`scale = 1/sqrt(head_dim)`。
