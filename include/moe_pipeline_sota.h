#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 固定 shape (num_tokens, H, I, E, top_k) 的持久化 device buffer
// CUDA Graph capture 会记录 kernel 读写的**固定** GPU 地址
// replay 要求多次运行间 d_* 指针不变
// 一次 cudaMalloc，多次 memcpy 更新输入
typedef struct MoePipelineSotaBuffers {
    int num_tokens;
    int hidden_size;
    int intermediate_size;
    int num_experts;
    int top_k;

    float *d_x;
    float *d_logits;
    float *d_y;
    float *d_w_gate;
    float *d_w_up;
    float *d_w_down;
    float *d_permuted_x;
    float *d_gate;
    float *d_up;
    float *d_mid;
    float *d_expert_out;
    float *d_route_weights;

    int *d_expert_ids;
    int *d_sorted_expert_ids;
    int *d_expanded_src_for_dest_row;
    int *d_source_token;
    int *d_source_k;
    int *d_expert_offsets;
    int *d_arange_buf;
    int *d_inv_permuted_idx;

    void *d_sort_workspace;
    size_t sort_workspace_bytes;
} MoePipelineSotaBuffers;

MoePipelineSotaBuffers *moe_pipeline_sota_buffers_create(int num_tokens, int hidden_size,
                                                         int intermediate_size, int num_experts,
                                                         int top_k);

void moe_pipeline_sota_buffers_destroy(MoePipelineSotaBuffers *buffers);

void moe_pipeline_sota_forward_device(void *stream, MoePipelineSotaBuffers *buffers);

void moe_pipeline_sota_forward(const float *x_host, const float *logits_host,
                               const float *w_gate_host, const float *w_up_host,
                               const float *w_down_host, float *y_host, int num_tokens,
                               int hidden_size, int intermediate_size, int num_experts, int top_k,
                               void *stream);

#ifdef __cplusplus
}
#endif
