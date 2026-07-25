# RoPE Device API

NeoX RoPE 生产路径：`rope_neox_forward_device`（`include/rope.h`）。

## 数据流

~~~
Model load
    │
    ▼
rope_cossin_cache_create(max_len, head_dim, freq_base)
    │  GPU: d_cos_sin [max_len][head_dim/2][2]
    ▼
Per forward step (prefill or decode)
    │
    ├─ Linear ──► d_q / d_k   layout [head_dim, num_heads, num_tokens, batch]
    │
    ├─ d_pos[device]          absolute positions, len = num_tokens
    │
    ▼
rope_neox_forward_device(stream, cache, d_in, d_out, d_pos, ...)
    │  kernel 查 d_cos_sin[pos[t]]
    ▼
d_out  ──► Flash Attention（生产：`fa_double_buffer_forward_device`）
~~~

## 接入 TransformerRunner

~~~
hidden ──► Linear(Q/K/V)
              │
              ├─► rope_neox_forward_device(Q, in-place)
              ├─► rope_neox_forward_device(K, in-place)
              │
              ▼
         Attention 占位 (d_q ──D2D──► d_attn_out)  →  目标：fa_double_buffer_forward_device
              │
              ▼
         Linear(O) ──► FFN
~~~

Runner 在 create 时持有 `RopeCosSinCache`；ForwardCtx 需传入 device 侧 `d_pos`。
测试入口 `forward_host(..., pos_offset)` 自动生成 pos = [offset, offset+T)。

## API

| 函数 | 用途 |
|------|------|
| `rope_cossin_cache_create` | Model 加载时创建 global cos/sin |
| `rope_neox_forward_device` | 生产：device 上 in/out/pos |
| `rope_with_global_cossin_cache` | 测试：host 包装 |

Layout 与 Qwen NeoX 一致：col-major `[head_dim, num_heads, num_tokens, batch]`。
