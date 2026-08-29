#include "fa_dst_unpack.h"

#include <cstdio>

#include "cuda_utils.cuh"

#define FA_DST_UNPACK_BLOCK_SIZE 256

// src: FA [head_dim, T, H]  ->  dst: Linear [H*head_dim, T]
// 例：D=2,T=2,H=2，src[d,t,h] 对应 dst[h*2+d, t]
__global__ void fa_dst_unpack_kernel(const float *__restrict__ src, float *__restrict__ dst,
                                     int head_dim, int num_tokens, int num_heads) {
    const int n_out = head_dim * num_tokens * num_heads;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) {
        return;
    }

    const int d = i % head_dim;
    const int t = (i / head_dim) % num_tokens;
    const int h = i / (head_dim * num_tokens);

    const int feat_dim = head_dim * num_heads;
    const int dst_feat = h * head_dim + d;
    dst[dst_feat + t * feat_dim] = src[i];
}

static int fa_dst_unpack_check_args(int head_dim, int num_tokens, int num_heads) {
    if (head_dim <= 0 || num_tokens <= 0 || num_heads <= 0) {
        std::fprintf(stderr,
                     "fa_dst_unpack_forward_device: invalid head_dim=%d num_tokens=%d "
                     "num_heads=%d\n",
                     head_dim, num_tokens, num_heads);
        return -1;
    }
    return 0;
}

extern "C" int fa_dst_unpack_forward_device(void *stream, const float *d_src, float *d_dst,
                                            int head_dim, int num_tokens, int num_heads) {
    if (fa_dst_unpack_check_args(head_dim, num_tokens, num_heads) != 0) {
        return -1;
    }
    if (d_src == nullptr || d_dst == nullptr) {
        std::fprintf(stderr, "fa_dst_unpack_forward_device: null pointer argument\n");
        return -1;
    }
    if (stream == nullptr) {
        std::fprintf(stderr, "fa_dst_unpack_forward_device: stream is null\n");
        return -1;
    }

    const int n_out = head_dim * num_tokens * num_heads;
    const int threads = FA_DST_UNPACK_BLOCK_SIZE;
    const int blocks = (n_out + threads - 1) / threads;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    fa_dst_unpack_kernel<<<blocks, threads, 0, s>>>(d_src, d_dst, head_dim, num_tokens, num_heads);
    return 0;
}

// ======================== 仅供 Python 测试 ================================
extern "C" void fa_dst_unpack_forward_host(const float *src_host, float *dst_host, int head_dim,
                                           int num_tokens, int num_heads) {
    if (fa_dst_unpack_check_args(head_dim, num_tokens, num_heads) != 0) {
        return;
    }
    if (src_host == nullptr || dst_host == nullptr) {
        std::fprintf(stderr, "fa_dst_unpack_forward_host: null pointer argument\n");
        return;
    }

    const int64_t n = static_cast<int64_t>(head_dim) * num_tokens * num_heads;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_src = nullptr;
    float *d_dst = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_src, static_cast<size_t>(n) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, static_cast<size_t>(n) * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_src, src_host, static_cast<size_t>(n) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    if (fa_dst_unpack_forward_device(stream, d_src, d_dst, head_dim, num_tokens, num_heads) != 0) {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaStreamDestroy(stream);
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, static_cast<size_t>(n) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFreeAsync(d_src, stream);
    cudaFreeAsync(d_dst, stream);
    cudaStreamDestroy(stream);
}
