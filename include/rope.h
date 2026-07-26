#pragma once

#include "rope_cossin_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

// NeoX RoPE 生产入口（声明见 rope.h）。
// 推理链路（如 TransformerRunner）在 Linear 产出 d_q/d_k 后直接调用；不做 host I/O。
// Layout: col-major [head_dim, num_heads, num_tokens, batch]
// d_pos: device 指针，长度 num_tokens；绝对 token 位置，用于查 global cos/sin cache
// d_output 可与 d_input 相同（in-place）
// stream 为 void*，实现处 static_cast 为 cudaStream_t
int rope_neox_forward_device(void *stream, const RopeCosSinCache *cache, const float *d_input,
                             float *d_output, const int *d_pos, int head_dim, int num_heads,
                             int num_tokens, int batch);

// 测试：host H2D → rope_neox_forward_device → D2H（仅供 Python binding）
void rope_neox_forward_host(float *input, int *pos, float *output, int head_dim, int num_heads,
                            int num_tokens, int batch, const RopeCosSinCache *cache);

#ifdef __cplusplus
}
#endif
