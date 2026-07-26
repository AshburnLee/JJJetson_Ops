# RMSNorm Device API

Pre-LN RMSNorm 生产路径：`rms_norm_forward_device`、`rms_norm_fused_add_forward_device`（`include/rms_norm.h`）。

## 数据流（Pre-LN block 片段）

~~~
hidden_in[device]
    │
    ├─ residual stream（attn 或 FFN 子块前保存的 hidden）
    │
    ▼
rms_norm_fused_add_forward_device(stream, d_input, d_residual, d_weight, H, T, eps)
    │  residual ← input + residual (= z)
    │  input    ← RMSNorm(z) * weight
    ▼
d_input ──► Linear(Q/K/V) ──► RoPE ──► FA ──► ...
~~~

非 fused 路径（单独归一化）：

~~~
rms_norm_forward_device(stream, d_in, d_weight, d_out, H, T, eps)
    │  y = x / RMS(x) * weight
    ▼
d_out
~~~

## 接入 TransformerRunner（目标）

~~~
input hidden
    │
    ▼
rms_norm_fused_add (input_layernorm weight)  ──► QKV + RoPE + FA + O
    │
    ▼
+ residual
    │
    ▼
rms_norm_fused_add (post_attention_layernorm weight) ──► SwiGLU FFN
    │
    ▼
+ residual ──► output
~~~

Runner create 时 H2D 一次 norm 权重；forward 内仅调用 device API，无 H2D/D2H。

## API

| 函数 | 用途 |
|------|------|
| `rms_norm_forward_device` | 生产：device 上 RMSNorm |
| `rms_norm_fused_add_forward_device` | 生产：Pre-LN add + RMSNorm（in-place input/residual） |
| `rms_norm_forward_host` / `rms_norm_fused_add_forward_host` | 测试：host 包装（H2D → device API → D2H） |

Layout：col-major `[hidden_size, num_tokens, 1, batch]`，在 `hidden_size` 维归一化。
