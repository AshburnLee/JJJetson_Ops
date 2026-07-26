#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 测试：host H2D → layer_norm kernel → D2H（仅供 Python binding）
void layer_norm_forward_host(float *input, float *weight, float *bias, float *output,
                             int hidden_size, int num_tokens, float epsilon);

#ifdef __cplusplus
}
#endif
