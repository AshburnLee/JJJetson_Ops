# RoPE Device API

NeoX RoPE 生产路径：`rope_neox_forward_device`（`include/rope.h`）。

## 数据流

```
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
d_out  ──► Flash Attention (下一步接入)
```

## 模块关系（与上、下游各一个模块的关系）

```
RopeCosSinCache          rope_neox_forward_device          TransformerRunner (待接)
 (model/session)    ◄───  (src/cuda/rope_neox_global_cache.cu) ───►  d_q / d_k buffers
     │                              │
     └── d_cos_sin ─────────────────┘

Python: rope_global_cache_me.forward_device  ──H2D/D2H──►  测试包装（非生产 I/O 路径）
```

## API

| 函数 | 用途 |
|------|------|
| `rope_cossin_cache_create` | Model 加载时创建 global cos/sin |
| `rope_neox_forward_device` | 生产：device 上 in/out/pos |
| `rope_with_global_cossin_cache` | 测试：host 包装 |

Layout 与 Qwen NeoX 一致：col-major `[head_dim, num_heads, num_tokens, batch]`。
