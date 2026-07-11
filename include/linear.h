#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void linear_forward_device(void *stream, void *cublas_handle, const float *input,
                           const float *weight, float *output, int in_features, int out_features,
                           int num_tokens);

#ifdef __cplusplus
}
#endif
