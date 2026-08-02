#include "kv_cache.h"

#include <cstdio>

#include "cuda_fp16.h"
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

// cache: [head_dim, max_seq, num_kv_heads, 1] float；
// dst: [head_dim, num_kv_tokens, num_kv_heads, 1] fp16；
// 索引见 doc/guide/transformer_runner_device_api.md
static __global__ void kv_cache_cast_fp16_kernel(const float *cache, uint16_t *dst, int head_dim,
                                                 int max_seq, int num_kv_heads, int num_kv_tokens) {
    const int n_out = head_dim * num_kv_tokens * num_kv_heads;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) {
        return;
    }

    const int d = i % head_dim;
    const int t = (i / head_dim) % num_kv_tokens;
    const int h = i / (head_dim * num_kv_tokens);

    const int cache_idx = d + t * head_dim + h * head_dim * max_seq;
    dst[i] = __half_as_ushort(__float2half(cache[cache_idx]));
}

extern "C" int kv_cache_cast_fp16_forward_device(void *stream, const float *d_cache,
                                                 uint16_t *d_dst, int head_dim, int max_seq,
                                                 int num_kv_heads, int num_kv_tokens) {
    if (head_dim <= 0 || max_seq <= 0 || num_kv_heads <= 0 || num_kv_tokens <= 0) {
        std::fprintf(stderr,
                     "kv_cache_cast_fp16_forward_device: invalid shape head_dim=%d max_seq=%d "
                     "num_kv_heads=%d num_kv_tokens=%d\n",
                     head_dim, max_seq, num_kv_heads, num_kv_tokens);
        return -1;
    }
    if (num_kv_tokens > max_seq) {
        std::fprintf(stderr,
                     "kv_cache_cast_fp16_forward_device: num_kv_tokens=%d exceeds max_seq=%d\n",
                     num_kv_tokens, max_seq);
        return -1;
    }
    if (d_cache == nullptr || d_dst == nullptr) {
        std::fprintf(stderr, "kv_cache_cast_fp16_forward_device: null pointer argument\n");
        return -1;
    }

    cudaStream_t s = stream != nullptr ? static_cast<cudaStream_t>(stream) : nullptr;
    if (s == nullptr) {
        std::fprintf(stderr, "kv_cache_cast_fp16_forward_device: stream is null\n");
        return -1;
    }

    const int n_out = head_dim * num_kv_tokens * num_kv_heads;
    const int threads = 256;
    const int blocks = (n_out + threads - 1) / threads;
    kv_cache_cast_fp16_kernel<<<blocks, threads, 0, s>>>(d_cache, d_dst, head_dim, max_seq,
                                                         num_kv_heads, num_kv_tokens);
    LAUNCH_CHECK();
    return 0;
}

// ======================== 仅供 Python 测试 ================================
extern "C" void kv_cache_cast_fp16_forward_host(const float *cache_host, uint16_t *dst_host,
                                                int head_dim, int max_seq, int num_kv_heads,
                                                int num_kv_tokens) {
    if (head_dim <= 0 || max_seq <= 0 || num_kv_heads <= 0 || num_kv_tokens <= 0) {
        return;
    }
    if (cache_host == nullptr || dst_host == nullptr) {
        std::fprintf(stderr, "kv_cache_cast_fp16_forward_host: null pointer argument\n");
        return;
    }

    const int64_t n_cache = static_cast<int64_t>(head_dim) * max_seq * num_kv_heads;
    const int64_t n_dst = static_cast<int64_t>(head_dim) * num_kv_tokens * num_kv_heads;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_cache = nullptr;
    uint16_t *d_dst = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_cache, static_cast<size_t>(n_cache) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, static_cast<size_t>(n_dst) * sizeof(uint16_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cache, cache_host, static_cast<size_t>(n_cache) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    if (kv_cache_cast_fp16_forward_device(stream, d_cache, d_dst, head_dim, max_seq, num_kv_heads,
                                          num_kv_tokens) != 0) {
        cudaFreeAsync(d_cache, stream);
        cudaFreeAsync(d_dst, stream);
        cudaStreamDestroy(stream);
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, static_cast<size_t>(n_dst) * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFreeAsync(d_cache, stream);
    cudaFreeAsync(d_dst, stream);
    cudaStreamDestroy(stream);
}
