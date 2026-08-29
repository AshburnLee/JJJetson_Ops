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
// GQA：g = num_q_heads/num_kv_heads 须为偶数且 >=2；每 block 2 个 Q。g=2 与旧 2:1 相同。
// 末 KV tile 不满 32 时只读该头有效 token，越界行不从 gmem 取。
// Q/K/V: fp16 device; stream 为 void*，实现处 static_cast 为 cudaStream_t
typedef struct FaDoubleBufferShape {
    int head_dim;
    int num_q_tokens;
    int num_kv_tokens;
    int num_q_heads;
    int num_kv_heads;
    // 0=只 mask pad；1=Llama 因果：再把 kv_abs > q_abs 打成 -inf
    int causal;
    // Q 第 0 行的绝对位置。q_abs = q_pos_offset + q_row
    int q_pos_offset;
} FaDoubleBufferShape;

// 校验 shape；0=通过，-1=失败（错误信息写 stderr）
int fa_double_buffer_validate_shape(const FaDoubleBufferShape *shape);

int fa_double_buffer_forward_device(void *stream, const FaDoubleBufferShape *shape,
                                    const uint16_t *d_q, const uint16_t *d_k, const uint16_t *d_v,
                                    float *d_dst, float scale);

// 测试：host H2D → fa_double_buffer_forward_device → D2H（仅供 Python binding）
void fa_double_buffer_forward_host(const FaDoubleBufferShape *shape, const uint16_t *q_host,
                                   const uint16_t *k_host, const uint16_t *v_host, float *dst_host,
                                   float scale);

// 测试：legacy 固定 shape (128/13/256/16/8)
void fa_double_buffer_forward_host_legacy(const uint16_t *q_host, const uint16_t *k_host,
                                          const uint16_t *v_host, float *dst_host, float scale);

#ifdef __cplusplus
}
#endif
