#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 生产 FA：fa_double_buffer.cu（WMMA + K/V 双缓冲 cp.async）
// 固定形状 (Phase 1 kernel 常量，与 fa_test_common 一致):
//   Q: fp16 col-major [128, 13, 16, 1]
//   K/V: fp16 col-major [128, 256, 8, 1]
//   dst: fp32 col-major [128, 13, 16, 1]
// fa/ 下其余实现 (one_pass / tc / tc_true 等) 为中间优化实验，engine 勿用。
// d_q/d_k/d_v/d_dst 均已在 GPU；stream 为 void*，实现处 static_cast 为 cudaStream_t
int fa_double_buffer_forward_device(void *stream, const uint16_t *d_q, const uint16_t *d_k,
                                    const uint16_t *d_v, float *d_dst, float scale);

#ifdef __cplusplus
}
#endif
