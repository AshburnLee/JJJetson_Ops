#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pack Linear/RoPE fp32 col-major [head_dim * num_heads, num_tokens]（即 feat_dim = head_dim *
// num_heads） 到 FA/KV 用的 fp16 col-major [head_dim, num_tokens, num_heads, 1]。 d_src 与 d_dst
// 均在 device；stream 为 void*，实现处 static_cast 为 cudaStream_t。
int qkv_pack_fp16_forward_device(void *stream, const float *d_src, uint16_t *d_dst, int head_dim,
                                 int num_tokens, int num_heads);

// 测试：host numpy H2D → qkv_pack_fp16_forward_device → D2H
void qkv_pack_fp16_forward_host(const float *src_host, uint16_t *dst_host, int head_dim,
                                int num_tokens, int num_heads);

#ifdef __cplusplus
}
#endif
