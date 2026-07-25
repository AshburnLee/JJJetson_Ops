#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 固定形状 (Phase 1 kernel 常量，与 fa_test_common 一致):
//   Q: fp16 col-major [128, 13, 16, 1]
//   K/V: fp16 col-major [128, 256, 8, 1]
//   dst: fp32 col-major [128, 13, 16, 1]
// fa/ 下其余实现 (one_pass / tc / tc_true 等) 为中间优化实验，engine 勿用。
// d_q/d_k/d_v/d_dst 均已在 GPU；stream 为 void*，实现处 static_cast 为 cudaStream_t

// 生产 FA：fa_double_buffer.cu（WMMA + K/V 双缓冲 cp.async）
// layout (col-major): Q [head_dim, num_q_tokens, num_q_heads, 1]
//                     K/V [head_dim, num_kv_tokens, num_kv_heads, 1]
//                     dst [head_dim, num_q_tokens, num_q_heads, 1] fp32
// Q/K/V: fp16 device; stream 为 void*，实现处 static_cast 为 cudaStream_t
typedef struct FaDoubleBufferShape {
    int head_dim;
    int num_q_tokens;
    int num_kv_tokens;
    int num_q_heads;
    int num_kv_heads;
} FaDoubleBufferShape;

// 校验 shape；0=通过，-1=失败（错误信息写 stderr）
int fa_double_buffer_validate_shape(const FaDoubleBufferShape *shape);

int fa_double_buffer_forward_device(void *stream, const FaDoubleBufferShape *shape,
                                    const uint16_t *d_q, const uint16_t *d_k, const uint16_t *d_v,
                                    float *d_dst, float scale);

#ifdef __cplusplus
}
#endif
