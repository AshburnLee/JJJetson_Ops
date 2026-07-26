#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// mid[i] = silu(gate[i]) * up[i]，gate/up/mid 为连续 flat buffer
void swiglu_silu_mul_launch_device(void *stream, const float *gate, const float *up, float *mid,
                                   int n_elem);

#ifdef __cplusplus
}
#endif
