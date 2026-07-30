# QKV Pack FP16 Device API

将 Linear/RoPE 输出的 **flat fp32** Q/K/V 重排并 cast 为 FA / KV cache 消费的 **head 分离 fp16** layout。

## 数据流（TransformerRunner Attention 前）

~~~
Linear/RoPE 输出（fp32 col-major）
  d_q / d_k / d_v : [feat_dim, num_tokens, 1, 1]
  feat_dim = head_dim * num_heads

        │
        ▼  qkv_pack_fp16_forward_device（每个 Q/K/V 各一次）
  d_q_fp16 : [head_dim, num_tokens, num_q_heads, 1]  fp16
  d_k_fp16 : [head_dim, num_tokens, num_kv_heads, 1] fp16
  d_v_fp16 : [head_dim, num_tokens, num_kv_heads, 1] fp16

        │
        ▼  （后续 roadmap）FA / KV cache
~~~

Attention 占位阶段仍用 fp32 `d_q` D2D；fp16 buffer 为 FA 与 KV cache 接入预置。

## API

| 层级 | C | Python |
|------|---|--------|
| 生产 | `qkv_pack_fp16_forward_device` | 不暴露 |
| 测试 | `qkv_pack_fp16_forward_host` | `qkv_pack_fp16_me.forward_host` |

~~~c
int qkv_pack_fp16_forward_device(void *stream, const float *d_src, uint16_t *d_dst,
                                 int head_dim, int num_tokens, int num_heads);
~~~

## 索引映射

flat src：`src_idx = (h * head_dim + d) + t * (head_dim * num_heads)`

packed dst：`dst_idx = d + t * head_dim + h * head_dim * num_tokens`

与 `fa_double_buffer` / `FaDoubleBufferShape` col-major 一致。
