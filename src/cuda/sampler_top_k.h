#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// top_k==1: parallel greedy (temperature ignored). top_k>1: top-k + softmax/T + sample.
int sampler_top_k_device(void *stream, const float *d_logits, int vocab_size, int top_k,
                         float temperature, uint64_t seed, int *d_out_token);

// Test wrapper: host logits -> H2D -> sampler_top_k_device -> token id on host.
int sampler_top_k_host(const float *logits_host, int vocab_size, int top_k, float temperature,
                       uint64_t seed);

#ifdef __cplusplus
}
#endif
