#pragma once

#include <stddef.h>
#include <stdint.h>

#include "rope_cossin_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TransformerRunner TransformerRunner;

// Device buffer 内部工作区
typedef struct TransformerLayerLinearDeviceBuffers {
    int num_tokens;
    int hidden_size;
    int q_dim;
    int kv_dim;
    int intermediate_size;

    float *d_hidden;
    float *d_q;
    float *d_k;
    float *d_v;
    float *d_attn_out;
    float *d_hidden_mid;
    float *d_gate;
    float *d_up;
    float *d_ffn_mid;
    float *d_hidden_out;
    float *d_residual; // Pre-LN residual stream（fused add 写入 z）

    // FA/KV layout fp16：[head_dim, num_tokens, num_heads, 1]（由 flat fp32 Q/K/V pack）
    uint16_t *d_q_fp16;
    uint16_t *d_k_fp16;
    uint16_t *d_v_fp16;
} TransformerLayerLinearDeviceBuffers;

typedef struct TransformerRunnerForwardCtx {
    int num_tokens;
    void *stream;
    const float *d_hidden_in;
    float *d_hidden_out;
    const int *d_pos; // device，长度 num_tokens；绝对 token 位置
} TransformerRunnerForwardCtx;

// max_seq_len / freq_base 用于创建 model 级 RopeCosSinCache
// w_input_layernorm / w_post_attention_layernorm: RMSNorm weight，各 [hidden_size]
TransformerRunner *
transformer_runner_create(int hidden_size, int intermediate_size, int num_q_heads, int num_kv_heads,
                          int head_dim, int max_seq_len, float freq_base, const float *w_q_host,
                          const float *w_k_host, const float *w_v_host, const float *w_o_host,
                          const float *w_gate_host, const float *w_up_host,
                          const float *w_down_host, const float *w_input_layernorm_host,
                          const float *w_post_attention_layernorm_host, void *stream);

// 释放 Runner、weights及按 num_tokens 缓存的中间 buffer
void transformer_runner_destroy(TransformerRunner *runner);

// 按 num_tokens 获取持久化 device 中间 buffer，供 Graph / 多次 forward 复用
TransformerLayerLinearDeviceBuffers *transformer_runner_buffers_get(TransformerRunner *runner,
                                                                    int num_tokens);

// 单层 Pre-LN（fused add + RMSNorm）+ Linear + RoPE + Attention 占位 + FFN
void transformer_layer_linears_forward_device(
    void *stream, void *cublas_handle, TransformerLayerLinearDeviceBuffers *buffers,
    const float *d_w_input_layernorm, const float *d_w_post_attention_layernorm, const float *d_w_q,
    const float *d_w_k, const float *d_w_v, const float *d_w_o, const float *d_w_gate,
    const float *d_w_up, const float *d_w_down, const RopeCosSinCache *rope_cache, const int *d_pos,
    int head_dim, int num_q_heads, int num_kv_heads, float rms_norm_epsilon);

// 生产入口：ctx 中 d_hidden_in/out 已在 GPU，内部 D2D 拷贝后执行 7 Linear 链
int transformer_runner_forward_device(TransformerRunner *runner,
                                      const TransformerRunnerForwardCtx *ctx);

// 测试入口：pos_offset 生成本步 d_pos = [offset, offset+num_tokens)
int transformer_runner_test(TransformerRunner *runner, const float *hidden_in_host,
                            float *hidden_out_host, int num_tokens, int pos_offset);

#ifdef __cplusplus
}
#endif
