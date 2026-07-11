#pragma once

#include <stddef.h>

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
} TransformerLayerLinearDeviceBuffers;

typedef struct TransformerRunnerForwardCtx {
    int num_tokens;
    void *stream;
    const float *d_hidden_in;
    float *d_hidden_out;
} TransformerRunnerForwardCtx;

// 7 个 Linear 权重一次性 H2D
TransformerRunner *transformer_runner_create(int hidden_size, int intermediate_size,
                                             int num_q_heads, int num_kv_heads, int head_dim,
                                             const float *w_q_host, const float *w_k_host,
                                             const float *w_v_host, const float *w_o_host,
                                             const float *w_gate_host, const float *w_up_host,
                                             const float *w_down_host, void *stream);

// 释放 Runner、weights及按 num_tokens 缓存的中间 buffer
void transformer_runner_destroy(TransformerRunner *runner);

// 按 num_tokens 获取持久化 device 中间 buffer，供 Graph / 多次 forward 复用
TransformerLayerLinearDeviceBuffers *transformer_runner_buffers_get(TransformerRunner *runner,
                                                                    int num_tokens);

// 单层 7 个 Linear + SwiGLU：Q/K/V -> O -> gate/up -> silu(gate)*up -> down，
// Attention 暂用 D2D 占位
void transformer_layer_linears_forward_device(void *stream, void *cublas_handle,
                                              TransformerLayerLinearDeviceBuffers *buffers,
                                              const float *d_w_q, const float *d_w_k,
                                              const float *d_w_v, const float *d_w_o,
                                              const float *d_w_gate, const float *d_w_up,
                                              const float *d_w_down);

// 生产入口：ctx 中 d_hidden_in/out 已在 GPU，内部 D2D 拷贝后执行 7 Linear 链
int transformer_runner_forward_device(TransformerRunner *runner,
                                      const TransformerRunnerForwardCtx *ctx);

// 测试入口：host hidden H2D -> forward_device 链 -> D2H 写回 hidden_out_host
int transformer_runner_test(TransformerRunner *runner, const float *hidden_in_host,
                            float *hidden_out_host, int num_tokens);

#ifdef __cplusplus
}
#endif
