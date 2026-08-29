#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 把 FA 出口拆回 O Linear 要的 flat。
// FA dst col-major [head_dim, num_tokens, num_heads, 1]
// Linear 入口 col-major [head_dim * num_heads, num_tokens]
//
// 小例子：head_dim=2, T=2, H=2
//   FA:  token0 head0=[1,2], token1 head0=[3,4], token0 head1=[5,6], token1 head1=[7,8]
//   unpack 后两列: t=0 -> [1,2,5,6]；t=1 -> [3,4,7,8]
// T=1 时两种 layout 碰巧相同。
// d_src / d_dst 均在 device；stream 为 void*。
int fa_dst_unpack_forward_device(void *stream, const float *d_src, float *d_dst, int head_dim,
                                 int num_tokens, int num_heads);

// 测试：host H2D -> fa_dst_unpack_forward_device -> D2H
void fa_dst_unpack_forward_host(const float *src_host, float *dst_host, int head_dim,
                                int num_tokens, int num_heads);

#ifdef __cplusplus
}
#endif
