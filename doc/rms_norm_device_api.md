# RMSNorm Device API

Pre-LN RMSNorm 生产路径：`rms_norm_forward_device`、`rms_norm_fused_add_forward_device`（`src/cuda/rms_norm.h`）。

## 数据流（Pre-LN block 片段）

### `rms_norm_fused_add_forward_device`（子块入口：add + RMSNorm 一次 kernel）

每个 Attention / FFN 子块**入口**各调用一次；`d_input` / `d_residual` in-place 更新。

**Pre-Attn 示例**（`d_residual` 先 memset 为 0，`w = input_layernorm`）：

~~~
d_input (= h，层输入)
    │
    ├──────────────────────────────────────────────────────────────┐
    │                              d_residual（skip；入口时为 0）   │
    ▼                                                              │
 rms_norm_fused_add_forward_device(w_input_layernorm)              │
    │  z = d_input + d_residual  (= h)                             │
    │  d_residual 覆盖为 z (= h，供子块出口 / 下一入口使用)           │
    │  d_input    覆盖为 RMSNorm(z) * w                             │
    ▼                                                              │
 Q/K/V Linear ──► RoPE ──► FA ──► O Linear ──► attn_out            │
    │                                                              │
    └──────────────► (Post-Attn：见下一次 fused_add 或 elementwise_add)
~~~

**Pre-FFN 示例**（`d_input = attn_out`，`d_residual = h`，`w = post_attention_layernorm`）：

~~~
d_input (= attn_out)
    │
    ├──────────────────────────────────────────────────────────────┐
    │                              d_residual (= h，Pre-Attn 保存)  │
    ▼                                                              │
 rms_norm_fused_add_forward_device(w_post_attention_layernorm)     │
    │  z = attn_out + h          （Post-Attn add 合并在此次 kernel）│
    │  d_residual 覆盖为 z (= h + attn_out)                        │
    │  d_input    覆盖为 RMSNorm(z) * w                            │
    ▼                                                              │
 gate/up Linear ──► SwiGLU ──► down Linear ──► ffn_out             │
    │                                                              │
    └──────────────► elementwise_add_forward_device ◄──────────────┘
                       (Post-FFN：ffn_out + d_residual)
                       │
                       ▼
                 hidden_out
~~~

### `rms_norm_forward_device`（非 fused：仅 Norm，无 residual 环，单独的 norm 不适用，独立算子）

~~~
d_in ──► RMSNorm(d_in) * weight ──► d_out
~~~

无 add、不更新 residual stream；当前 Runner Pre-LN 路径使用 fused add，不调用此 API。

## 接入 TransformerRunner

单层 block 内 **两次** `rms_norm_fused_add`（40b 已接入）；Post-FFN plain add 见 40c / `elementwise_add_forward_device`：

~~~
hidden_in
    │
    ├──────────────────────────────────────────────────────────────┐
    ▼                                                              │
 rms_norm_fused_add (input_layernorm)                              │
    ▼                                                              │
 Q/K/V ──► RoPE ──► FA ──► O ──► attn_out                          │
    │                                                              │
    └──────────────► rms_norm_fused_add (post_attention) ◄─────────┘
                       │  （此次 fused add 内含 attn_out + h）
                       ▼
                 gate/up ──► SwiGLU ──► down ──► ffn_out
                       │
                       ▼
              elementwise_add (Post-FFN) ──► hidden_out
~~~

Runner create 时 H2D 一次 norm 权重；forward 内仅调用 device API，无 H2D/D2H。

## API

| 函数 | 用途 |
|------|------|
| `rms_norm_forward_device` | 生产：device 上 RMSNorm |
| `rms_norm_fused_add_forward_device` | 生产：Pre-LN add + RMSNorm（in-place input/residual） |
| `rms_norm_forward_host` / `rms_norm_fused_add_forward_host` | 测试：host 包装（H2D → device API → D2H）；Python 入口 `forward_host` |

Layout：col-major `[hidden_size, num_tokens, 1, batch]`，在 `hidden_size` 维归一化。
