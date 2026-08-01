# TransformerRunner 对象生命周期（Phase 1 单层）

Phase 1 交付的是 一个完整的 Pre-LN Transformer decoder block（RMSNorm + residual + QKV + RoPE + FA + KV cache + O + SwiGLU FFN），对外暴露的是 TransformerRunner

对应当前代码：`transformer_runner.cpp`、`kv_cache.cpp`、`transformer_runner_binding.cpp`。

Python 只持有 `uintptr_t` handle；**无自动析构**，必须 `destroy_runner`。

---

## 1. 包含关系

~~~
Python handle (整数)
    └── TransformerRunner          [堆对象，手动 delete]
            ├── cudaStream         (runner 自建时 owns)
            ├── cublasHandle
            ├── RopeCosSinCache    cos/sin 表 GPU
            ├── KVCache            session 级 K/V 历史
            │       └── layer[0]   d_k, d_v  (float, max_seq)
            ├── d_k_fa_fp16        FA 读 K staging (fp16)
            ├── d_v_fa_fp16        FA 读 V staging (fp16)
            ├── d_w_* × 9          权重 GPU（create 时 H2D 一次）
            ├── buffers_by_tokens[T]   layer 工作区（按 T 懒分配）
            └── d_pos_by_tokens[T]     测试路径 pos（按 T 懒分配）
~~~

---

## 2. 对象表

~~~
对象                      创建                          销毁                          reset / 复用
------------------------  ----------------------------  ----------------------------  ---------------------------
TransformerRunner         transformer_runner_create     transformer_runner_destroy    无；整 session 存活
                          (Python create_runner)

权重 d_w_q…d_w_post       create 内 cudaMalloc+H2D     destroy 内 cudaFree           不变

RopeCosSinCache           create 内                     destroy 内                    不变；只读查表

KVCache                   create 内 kv_cache_create    destroy 内 kv_cache_destroy
  · layers[i].d_k/d_v     create 内 cudaMalloc         destroy 内 cudaFree           kv_cache_reset：
  · cache_len (=0)        create 置 0                  —                             仅 cache_len→0，buffer 不 free

d_k_fa_fp16 / d_v_fa_fp16 create 内 cudaMalloc        destroy 内 cudaFree           每 forward cast 覆写有效段

buffers_by_tokens[T]      首次 forward(T) 懒分配       destroy 内全部 free           同 T 多次 forward 复用
  · d_hidden,d_q,d_k…     transformer_layer_linear_
                          buffers_create

d_pos_by_tokens[T]        首次 forward_host(T)         destroy 内 cudaFree           每步刷新 pos 内容
                          (测试路径)

TransformerRunnerForwardCtx  每步 forward 栈上填充      函数返回即失效                不持久化
  · d_hidden_in/out,d_pos  调用方或 binding 提供

本步 flat d_k/d_v          在 buffers[T] 内             随 buffers[T]                 每 forward 被 Linear 覆盖
                          (非 KV cache)
~~~

**区分两块 K/V：**

| 指针 | 归属 | 含义 |
|------|------|------|
| `buffers->d_k` / `d_v` | layer workspace | 本步 Linear 输出，flat `[kv_dim, T]` |
| `kv_cache layers[0].d_k/d_v` | KVCache | 历史累积，layout `[head_dim, max_seq, num_kv_heads, 1]` |

append 把本步 flat K/V **写入** cache `[cache_len, cache_len+T)`。

---

## 3. 时序

### 3.1 标准对话（prefill → decode）

~~~
create_runner
  cache_len = 0

prefill (T=13, pos=[0..12])
  forward → append [0,13) → cast/FA → advance_len(13)
  cache_len = 13

decode (T=1, pos=[13])
  forward → append [13,14) → cast num_kv_tokens=14 → advance_len(1)
  cache_len = 14

decode × N …
~~~

### 3.2 新对话（同一 runner）

~~~
kv_cache_reset          cache_len = 0，GPU cache 指针不变

prefill (T=13, pos=[0..12])   从 slot 0 重新 append
  cache_len = 13
~~~

**不要**在 prefill→decode 之间 reset。

### 3.3 结束 session

~~~
destroy_runner
  kv_cache_destroy → cudaFree cache
  cudaFree 权重、FA fp16、workspace…
  delete TransformerRunner
~~~

---

## 4. 单步 forward 内（layer 内 vs 外）

~~~
layer 内 transformer_layer_linears_forward_device:
  [1] Pre-Attn fused add + Q/K/V Linear + RoPE
  [2] q pack → d_q_fp16
  [3] L = kv_cache_get_len()
  [4] kv_cache_append(d_k, d_v, T)     ← 不改 cache_len
  [5] kv_cache_cast_fp16 → d_k/v_fa_fp16 (num_kv_tokens = L+T)
  [6] FA → d_attn_out
  [7] O Linear + Pre-FFN + FFN + Post-FFN add

layer 外 transformer_runner_forward_device / test:
  [8] kv_cache_advance_len(T)          ← cache_len += T
~~~

---

## 5. Python API ↔ C

~~~
create_runner(...)              → transformer_runner_create
destroy_runner(handle)          → transformer_runner_destroy
forward_host(..., pos_offset)   → transformer_runner_test（自动生成 d_pos）
forward_device(..., pos)        → transformer_runner_forward_device
kv_cache_len(handle)            → kv_cache_get_len
kv_cache_reset(handle)          → kv_cache_reset
~~~

`forward_device` binding 内对 hidden/pos 的 cudaMalloc 仅为测试 H2D 包装；生产路径应自带 GPU 指针调 C API。

API 契约（签名、`ForwardCtx`、错误码、KV layout 索引）见 `doc/transformer_runner_device_api.md`。

---

## 6. Phase 2 预留（当前未实现）

~~~
TransformerModel / InferenceEngine   持有多 layer 权重 + KVCache(num_layers>1)
  └── 可能包装或替代单层 TransformerRunner
~~~

当前 Phase 1：`num_layers=1`，Runner 即 session 边界。

Phase 2 顶层设计见 [`phase2_lifecycle.md`](phase2_lifecycle.md)（`InferenceEngine` + `TransformerModel`）。
