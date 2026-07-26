#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void linear_forward_device(void *stream, void *cublas_handle, const float *input,
                           const float *weight, float *output, int in_features, int out_features,
                           int num_tokens);

// 测试：host H2D → linear_forward_device → D2H（仅供 Python binding）
void linear_forward_host(float *input, float *weight, float *output, int in_features,
                         int num_tokens, int out_features);

#ifdef __cplusplus
}
#endif
