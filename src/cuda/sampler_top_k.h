#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// top_k==1: parallel greedy on device. top_k>1: functional but TODO(perf-topk) single-thread scan.
int sampler_top_k_device(void *stream, const float *d_logits, int vocab_size, int top_k,
                         uint64_t seed, int *d_out_token);

// Test wrapper: host logits -> H2D -> sampler_top_k_device -> token id on host.
int sampler_top_k_host(const float *logits_host, int vocab_size, int top_k, uint64_t seed);

#ifdef __cplusplus
}
#endif
