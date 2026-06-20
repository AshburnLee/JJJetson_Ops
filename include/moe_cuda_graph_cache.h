#pragma once

#include "moe_pipeline_sota.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MoECudaGraphCache MoECudaGraphCache;

MoECudaGraphCache *moe_cuda_graph_cache_create(int hidden_size, int intermediate_size,
                                               int num_experts, int top_k, void *stream);

void moe_cuda_graph_cache_destroy(MoECudaGraphCache *cache);

int moe_cuda_graph_cache_has(const MoECudaGraphCache *cache, int num_tokens);

int moe_cuda_graph_cache_capture(MoECudaGraphCache *cache, int num_tokens);

int moe_cuda_graph_cache_replay(MoECudaGraphCache *cache, int num_tokens);

MoePipelineSotaBuffers *moe_cuda_graph_cache_buffers_get(MoECudaGraphCache *cache, int num_tokens);

int moe_cuda_graph_cache_run(MoECudaGraphCache *cache, int num_tokens, const float *x_host,
                             const float *logits_host, const float *w_gate_host,
                             const float *w_up_host, const float *w_down_host, float *y_host);

#ifdef __cplusplus
}
#endif
