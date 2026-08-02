# Pre-LN vs Post-LN Transformer Block

JJJetson_Ops Phase 1 目标：**LLaMA 式 Pre-LN + RMSNorm** 单层 block（见 roadmap）。本文对比 Post-LN / Pre-LN 结构差异，并给出 engine 侧数据流与算子命名。

## 单层结构对比 (Attention + FFN)

原始 Transformer (Vaswani et al.) 采用 **Post-LN**：子层 (Attention / FFN) 先算，再 LayerNorm，residual 接在 norm 之后。现代 LLM (LLaMA, Qwen 等) 普遍采用 **Pre-LN**：先 RMSNorm/LayerNorm，再子层，residual 接在子层输出之后。

### Post-LN (原始 GPT/BERT 系)

~~~
hidden_in (x)
    │
    ├──────────────────────────────┐  residual skip（绕开 Attn）
    │                              │
    ▼                              │
 Attention(x)                      │
    │                              │
    └──────────────► (+) ◄─────────┘
                       │
                       ▼
                 LayerNorm(z1)  ──►  hidden_mid
                       │
    ┌──────────────────┴──────────────────┐  residual skip（绕开 FFN）
    │                                     │
    ▼                                     │
 FFN(hidden_mid)                          │
    │                                     │
    └──────────────► (+) ◄────────────────┘
                       │
                       ▼
                 LayerNorm(z2)  ──►  hidden_out
~~~

Norm 在 **子层输出 + residual 之后**；residual 环只绕过子层 (Attn / FFN)，**不绕过** LayerNorm。深层训练时梯度路径较长，需 careful init / warmup。

### Pre-LN (LLaMA / Qwen / JJJetson_Ops 目标)

~~~
hidden_in (x)
    │
    ├──────────────────────────────┐  residual skip（绕开 Norm+Attn）
    │                              │
    ▼                              │
 RMSNorm(x)  ──►  x_norm           │
    │                              │
    ▼                              │
 Attention(x_norm)                 │
    │                              │
    └──────────────► (+) ◄─────────┘
                       │
                       ▼
                  hidden_mid
                       │
    ┌──────────────────┴──────────────────┐  residual skip（绕开 Norm+FFN）
    │                                     │
    ▼                                     │
 RMSNorm  ──►  x_norm2                    │
    │                                     │
    ▼                                     │
 FFN(x_norm2)                             │
    │                                     │
    └──────────────► (+) ◄────────────────┘
                       │
                       ▼
                  hidden_out
~~~

Norm 在 **子层入口**；residual stream 绕开 norm，梯度更直接，深层 stack 更稳定。JJJetson_Ops 使用 **RMSNorm** (无 mean center，仅 scale by RMS + learnable weight)，与 LayerNorm 相比计算更轻、与 LLaMA 权重布局一致。

## JJJetson_Ops / LLaMA 式 Pre-LN 数据流

Phase 1 block 目标链（roadmap）：

`input → Pre-Attn RMSNorm(+residual) → QKV + RoPE → FA(+KV cache) → O proj → +residual → Pre-FFN RMSNorm(+residual) → SwiGLU FFN → +residual → output`

算子级数据流（device 生产路径命名；`──┐` / `◄─┘` 为 residual 环）：

~~~
hidden_in[device]
    │                                      residual stream（attn 子块前保存）
    ├─────────────────────────────────────────────────────────┐
    │                                                         │
    ▼                                                         │
 rms_norm_fused_add_forward_device  (input_layernorm weight)  │
    │  residual ← input + residual                            │
    │  input    ← RMSNorm(z) * weight                         │
    ▼                                                         │
 Q/K/V Linear ──► RoPE ──► FA ──► O Linear                    │
    │                                                         │
    └──────────────► elementwise_add_forward_device ◄─────────┘  attn 子块出口
                       │
                       ▼
              hidden_mid[device]
                       │                    residual stream（FFN 子块前保存）
    ┌──────────────────┴──────────────────────────────────────────────┐
    │                                                                 │
    ▼                                                                 │
 rms_norm_fused_add_forward_device  (post_attention_layernorm weight) │
    │                                                                 │
    ▼                                                                 │
 gate/up Linear ──► SwiGLU ──► down Linear                            │
    │                                                                 │
    └──────────────► elementwise_add_forward_device ◄─────────────────┘  FFN 子块出口
                       │
                       ▼
                 hidden_out[device]
~~~

`rms_norm_fused_add_forward_device` 将 Pre-LN 的 **add + RMSNorm** 融合为一次 kernel（in-place input/residual）；详见 [`../guide/rms_norm_device_api.md`](../guide/rms_norm_device_api.md)。Norm 权重在 Runner create 时 H2D 一次，forward 内无额外 H2D/D2H。

## 对比摘要

~~~
维度: Norm 位置
  Post-LN: 子层之后 (Attn->add->Norm->FFN->add->Norm)
  Pre-LN:  子层之前 (Norm->Attn->add->Norm->FFN->add)

维度: 常用 Norm
  Post-LN: LayerNorm (mean + var)
  Pre-LN:  RMSNorm (RMS only，无 bias)

维度: 训练稳定性
  Post-LN: 深层需 careful init；梯度经 norm 回传
  Pre-LN:  residual 直连，深层更稳；现代 LLM 默认

维度: 典型模型
  Post-LN: Transformer, GPT-2, BERT
  Pre-LN:  LLaMA, Mistral, Qwen2/3, Gemma
~~~

## Qwen3 与 LLaMA 同族

Qwen3 与 LLaMA 同属 **Pre-LN + RMSNorm** decoder stack：GQA、SwiGLU、RoPE，block 级 norm 在 Attention / FFN **入口**。

公开来源可核对：

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (§3.1)：*"The architecture of the Qwen3 dense models is similar to Qwen2.5, including using ... RMSNorm with **pre-normalization**."*
- HuggingFace `Qwen3DecoderLayer`：`input_layernorm` / `post_attention_layernorm` 均为 `Qwen3RMSNorm`，forward 为 `residual + sublayer(RMSNorm(x))` 的 Pre-LN 模式。

Qwen3 额外特性 (不影响 block 级 Pre-LN 判定)：**QK-Norm** (对 Q/K 向量再做 RMSNorm)、MoE 变体、移除 QKV bias。JJJetson_Ops Phase 1 先对齐 dense LLaMA 式单层；QK-Norm / MoE 属后续 scope。

## 参考

- Roadmap block 定义：`.cursor/rules/jjjetson-ops-roadmap.mdc`
- RMSNorm device API：[`../guide/rms_norm_device_api.md`](../guide/rms_norm_device_api.md)
- Runner 集成状态：`src/engine/transformer_runner.h`, `src/engine/transformer_runner.cpp`
