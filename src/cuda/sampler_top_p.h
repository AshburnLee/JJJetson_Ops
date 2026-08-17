#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// top_p in (0,1]: nucleus on device. Functional but TODO(perf-topp) single-thread + per-call temp
// alloc.
int sampler_top_p_device(void *stream, const float *d_logits, int vocab_size, float top_p,
                         float temperature, int top_k, uint64_t seed, int *d_out_token);

// Test wrapper: host logits -> H2D -> sampler_top_p_device -> token id on host.
int sampler_top_p_host(const float *logits_host, int vocab_size, float top_p, float temperature,
                       int top_k, uint64_t seed);

#ifdef __cplusplus
}
#endif
