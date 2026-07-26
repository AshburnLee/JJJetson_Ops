# FA Device API

Flash Attention **生产路径**：`fa_double_buffer_forward_device`（`src/cuda/fa/fa.h`，实现 `src/cuda/fa/fa_double_buffer.cu`）。

`fa/` 下其余 kernel 为中间优化实验，**engine 勿用**。

## Layout (col-major)

| Tensor | dtype | shape |
|--------|-------|-------|
| Q | fp16 | [head_dim, num_q_tokens, num_q_heads, 1] |
| K/V | fp16 | [head_dim, num_kv_tokens, num_kv_heads, 1] |
| dst | fp32 | [head_dim, num_q_tokens, num_q_heads, 1] |

## 约束（Phase 1）

- `head_dim` ∈ {32, 64, 128}（16 的倍数，WMMA）
- `num_q_heads == 2 * num_kv_heads`（每 block 2 个 Q head）
- `1 <= num_q_tokens <= 16`
- `num_kv_tokens >= 1`（末 tile 自动 mask）
- Q/K/V fp16；scale 通常为 `1/sqrt(head_dim)`

## 数据流

~~~
Linear(Q/K/V)  ──► fp16 device Q/K/V
    │
    ▼
fa_double_buffer_forward_device(stream, &shape, d_q, d_k, d_v, d_dst, scale)
    │  WMMA + K/V 双缓冲 cp.async；num_kv_heads blocks × 2 Q-heads
    ▼
d_dst (fp32) ──► Linear(O)
~~~

## API

| 函数 | 用途 |
|------|------|
| `FaDoubleBufferShape` | head_dim / tokens / heads |
| `fa_double_buffer_validate_shape` | host 校验 |
| `fa_double_buffer_forward_device` | 生产 device 入口 |
| `fa_double_buffer_forward_host` / `fa_double_buffer_forward_host_legacy` | 测试 H2D/D2H |
| `fa_me.forward_host_shape` | Python：从 Q/K/V shape 推断并调用 |
| `fa_me.forward_host` | Python：legacy 固定 shape |

Legacy 固定 128×13×16 / KV256 仍可用 `fa_me.launch_fa` / `forward_host`。

## 待办（Runner 接入）

- fp32 Q/K/V 与 Linear 输出对接（cast 或 fp32 路径）
- KV cache 读写与 decode 变长 seq
