#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// RMSNorm 生产入口。
// Layout: col-major [hidden_size, num_tokens, 1, batch]；在 hidden_size 维归一化。
// d_output 可与 d_input 相同（in-place）。
// stream 为 void*，实现处 static_cast 为 cudaStream_t。
int rms_norm_forward_device(void *stream, const float *d_input, const float *d_weight,
                            float *d_output, int hidden_size, int num_tokens, float epsilon);

// Pre-LN fused add + RMSNorm 生产入口。
// z = input + residual；residual 写入 z；input 写入 RMSNorm(z) * weight。
// d_input / d_residual 均在 device，in-place 更新。
int rms_norm_fused_add_forward_device(void *stream, float *d_input, float *d_residual,
                                      const float *d_weight, int hidden_size, int num_tokens,
                                      float epsilon);

// 测试：host numpy H2D → *_forward_device → D2H（仅供 Python binding）
void rms_norm_forward_host(const float *input_host, const float *weight_host, float *output_host,
                           int hidden_size, int num_tokens, float epsilon);

void rms_norm_fused_add_forward_host(float *input_host, float *residual_host,
                                     const float *weight_host, int hidden_size, int num_tokens,
                                     float epsilon);

#ifdef __cplusplus
}
#endif
