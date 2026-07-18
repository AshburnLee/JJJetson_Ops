#include "kv_cache.h"

#include "cuda_utils.cuh"

// src: [head_dim, n_tokens, num_kv_heads, 1] col-major 等价 flat [kv_dim, n_tokens]
// dst: [head_dim, max_seq, num_kv_heads, 1] col-major，写入 [offset, offset + n_tokens)
static __global__ void kv_cache_append_kernel(const float *src_k, const float *src_v, float *dst_k,
                                              float *dst_v, int head_dim, int num_kv_heads,
                                              int kv_dim, int max_seq, int offset, int n_tokens) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_elem = head_dim * num_kv_heads * n_tokens;
    if (idx >= n_elem) {
        return;
    }

    const int d = idx % head_dim;
    const int rest = idx / head_dim;
    const int h = rest % num_kv_heads;
    const int t_local = rest / num_kv_heads;

    const int src_idx = d + head_dim * h + kv_dim * t_local;
    const int dst_idx = d + head_dim * (offset + t_local) + head_dim * max_seq * h;

    dst_k[dst_idx] = src_k[src_idx];
    dst_v[dst_idx] = src_v[src_idx];
}

extern "C" void kv_cache_append_launch_device(void *stream, const float *d_k, const float *d_v,
                                              float *d_k_cache, float *d_v_cache, int head_dim,
                                              int num_kv_heads, int max_seq, int offset,
                                              int n_tokens) {
    if (n_tokens <= 0) {
        return;
    }

    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int kv_dim = head_dim * num_kv_heads;
    const int n_elem = kv_dim * n_tokens;
    const int threads = 256;
    const int blocks = (n_elem + threads - 1) / threads;

    kv_cache_append_kernel<<<blocks, threads, 0, s>>>(
        d_k, d_v, d_k_cache, d_v_cache, head_dim, num_kv_heads, kv_dim, max_seq, offset, n_tokens);
    LAUNCH_CHECK();
}
