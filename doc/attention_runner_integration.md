# Attention 段 Runner 集成（Phase 1）

TransformerRunner Attention 前数据流（当前进度）。

## RoPE 之后

~~~
d_q / d_k / d_v   fp32 flat [feat_dim, num_tokens]

d_q  ──► qkv_pack_fp16 ──► d_q_fp16 [head_dim, T, num_q_heads, 1]     （本步 FA Q）

d_k ─┐
     ├──► kv_cache_append_device（flat fp32 → cache float）
d_v ─┘         layout cache: [head_dim, max_seq, num_kv_heads, 1]
               写入 [cache_len, cache_len + T)

cache K/V ──► kv_cache_cast_fp16 ──► d_k_fa_fp16 / d_v_fa_fp16
                                    [head_dim, cache_len+T, num_kv_heads, 1] fp16
                                    （FA 读全历史 + 本步）

Attention 占位：仍 D2D d_q(fp32) → d_attn_out
FA 接入：fa_double_buffer_forward_device(d_q_fp16, d_k_fa_fp16, d_v_fa_fp16, ...)
~~~

## 生命周期（单层 Phase 1）

### 核心变量

- **`cache_len`（L）**：KV cache 里**已经提交**的历史 token 数。刚 create 时为 **0**。
- **`T`（num_tokens）**：**本步 forward** 要处理的 token 数（prefill 可 >1，decode 通常为 1）。
- **`append` 只写 GPU buffer，不改 `cache_len`**；**`advance_len(T)`** 在整步 forward 结束后才把 `cache_len += T`。

### create：分配一块 session 级 cache

`transformer_runner_create` 内部调用：

~~~
kv_cache_create(max_seq, head_dim, num_kv_heads, num_layers=1)
~~~

- GPU 上预分配 K/V 各一块：`[head_dim, max_seq, num_kv_heads, 1]` float（col-major）。
- 同时分配 Runner 级 **`d_k_fa_fp16` / `d_v_fa_fp16`**，长度按 `max_seq`（供 FA 读「历史 + 本步」）。
- 此时 **`cache_len = 0`**，cache 里尚无有效 token。

### forward 一步（layer 内 + layer 外）

**Layer 内**（`transformer_layer_linears_forward_device`，RoPE 之后）：

| 顺序 | 动作 | 含义 |
|------|------|------|
| 1 | `qkv_pack_fp16(d_q → d_q_fp16)` | 仅本步 Q：fp32 flat → fp16 FA layout，长度 **T** |
| 2 | `L = kv_cache_get_len()` | 记下 append **之前** 的历史长度 |
| 3 | `kv_cache_append_device(d_k, d_v, T)` | 把本步 K/V（flat fp32）写入 cache 的 **`[L, L+T)`** 槽位；**此时 `cache_len` 仍为 L** |
| 4 | `kv_cache_cast_fp16(cache → d_k/v_fa_fp16, num_kv_tokens=L+T)` | 把 cache **从头读到 L+T** cast 成 fp16，供 FA 使用（**历史 + 刚 append 的本步**） |
| 5 | Attention / O / FFN … | 当前仍为 D2D 占位；接 FA 时用 `d_q_fp16` + `d_k/v_fa_fp16` |

**Layer 外**（`transformer_runner_test` / `forward_device` 返回前）：

| 顺序 | 动作 | 含义 |
|------|------|------|
| 6 | `kv_cache_advance_len(T)` | **`cache_len` 从 L 变为 L+T**，表示本步 K/V 已「入账」 |

要点：**append 与 advance 成对**——append 先写 slot，advance 再 bump 长度；下一步 append 的 offset 自动等于新的 `cache_len`。

### 例子：prefill 13 + decode 1（与 `test_transformer_runner_kv_cache.py` 一致）

~~~
初始:     cache_len = 0

Prefill (T=13, pos=[0..12]):
  append  →  K/V 写入 cache[0:13)，cache_len 仍为 0
  cast    →  FA 侧 num_kv_tokens = 0+13 = 13
  advance →  cache_len = 13

Decode (T=1, pos=[13]):
  append  →  K/V 写入 cache[13:14)，cache_len 仍为 13
  cast    →  FA 侧 num_kv_tokens = 13+1 = 14（Q 只有 1 个 token，K/V 看全长 14）
  advance →  cache_len = 14
~~~

decode 时 **Q 的 T=1**，但 **K/V 的 FA 输入长度 = cache_len + T**，即「全部历史 + 当前 token」。

### 新 request / 新对话

- **`kv_cache_reset()`**（后续暴露给 Runner API）：`cache_len = 0`，GPU buffer 复用、不 free。
- 或 **`destroy_runner` + `create_runner`**：彻底释放再建。

同一对话内 **prefill → decode 不要 reset**；换 prompt / 新 session 才 reset。

## 相关模块

| 模块 | 文件 |
|------|------|
| KV Session | `src/model/kv_cache.h` |
| append / cast kernel | `src/cuda/kv_cache.cu` |
| Q pack | `src/cuda/qkv_pack_fp16.h` |
| Runner glue | `src/engine/transformer_runner.cpp` |
