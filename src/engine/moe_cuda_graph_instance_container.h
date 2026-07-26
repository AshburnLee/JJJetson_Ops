#pragma once

#include "moe_pipeline_sota.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MoECudaGraphInstanceContainer MoECudaGraphInstanceContainer;

MoECudaGraphInstanceContainer *moe_cuda_graph_instance_container_create(int hidden_size,
                                                                        int intermediate_size,
                                                                        int num_experts, int top_k,
                                                                        void *stream);

void moe_cuda_graph_instance_container_destroy(MoECudaGraphInstanceContainer *container);

int moe_cuda_graph_instance_container_has(const MoECudaGraphInstanceContainer *container,
                                          int num_tokens);

int moe_cuda_graph_instance_container_capture(MoECudaGraphInstanceContainer *container,
                                              int num_tokens);

int moe_cuda_graph_instance_container_replay(MoECudaGraphInstanceContainer *container,
                                             int num_tokens);

MoePipelineSotaBuffers *
moe_cuda_graph_instance_container_buffers_get(MoECudaGraphInstanceContainer *container,
                                              int num_tokens);

int moe_cuda_graph_instance_container_run(MoECudaGraphInstanceContainer *container, int num_tokens,
                                          const float *x_host, const float *logits_host,
                                          const float *w_gate_host, const float *w_up_host,
                                          const float *w_down_host, float *y_host);

#ifdef __cplusplus
}
#endif
