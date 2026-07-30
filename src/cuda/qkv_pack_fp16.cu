#include "qkv_pack_fp16.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

#include "cuda_utils.cuh"

#define QKV_PACK_FP16_BLOCK_SIZE 256

// 将 flat fp32 [feat_dim, num_tokens] 重排为 [head_dim, num_tokens, num_heads, 1] fp16
__global__ void qkv_pack_fp16_kernel(const float *__restrict__ src, uint16_t *__restrict__ dst,
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
    const int src_feat = h * head_dim + d;
    const int src_idx = src_feat + t * feat_dim;

    dst[i] = __half_as_ushort(__float2half(src[src_idx]));
}

static int qkv_pack_fp16_check_args(int head_dim, int num_tokens, int num_heads) {
    if (head_dim <= 0 || num_tokens <= 0 || num_heads <= 0) {
        std::fprintf(
            stderr,
            "qkv_pack_fp16_forward_device: invalid head_dim=%d num_tokens=%d num_heads=%d\n",
            head_dim, num_tokens, num_heads);
        return -1;
    }
    return 0;
}

extern "C" int qkv_pack_fp16_forward_device(void *stream, const float *d_src, uint16_t *d_dst,
                                            int head_dim, int num_tokens, int num_heads) {
    if (qkv_pack_fp16_check_args(head_dim, num_tokens, num_heads) != 0) {
        return -1;
    }
    if (d_src == nullptr || d_dst == nullptr) {
        std::fprintf(stderr, "qkv_pack_fp16_forward_device: null pointer argument\n");
        return -1;
    }

    cudaStream_t s = stream != nullptr ? static_cast<cudaStream_t>(stream) : nullptr;
    if (s == nullptr) {
        std::fprintf(stderr, "qkv_pack_fp16_forward_device: stream is null\n");
        return -1;
    }

    const int n_out = head_dim * num_tokens * num_heads;
    const int threads = QKV_PACK_FP16_BLOCK_SIZE;
    const int blocks = (n_out + threads - 1) / threads;

#if defined(MY_OPS_DEBUG)
    std::printf("qkv_pack_fp16 launch: blocks=%d threads=%d n_out=%d\n", blocks, threads, n_out);
    std::fflush(stdout);
#endif

    qkv_pack_fp16_kernel<<<blocks, threads, 0, s>>>(d_src, d_dst, head_dim, num_tokens, num_heads);
    LAUNCH_CHECK();
    return 0;
}

// ======================== 仅供 Python 测试 ================================
extern "C" void qkv_pack_fp16_forward_host(const float *src_host, uint16_t *dst_host, int head_dim,
                                           int num_tokens, int num_heads) {
    if (qkv_pack_fp16_check_args(head_dim, num_tokens, num_heads) != 0) {
        return;
    }
    if (src_host == nullptr || dst_host == nullptr) {
        std::fprintf(stderr, "qkv_pack_fp16_forward_host: null pointer argument\n");
        return;
    }

    const int feat_dim = head_dim * num_heads;
    const int64_t n_src = static_cast<int64_t>(feat_dim) * num_tokens;
    const int64_t n_dst = static_cast<int64_t>(head_dim) * num_tokens * num_heads;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_src = nullptr;
    uint16_t *d_dst = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_src, static_cast<size_t>(n_src) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, static_cast<size_t>(n_dst) * sizeof(uint16_t), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_src, src_host, static_cast<size_t>(n_src) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    if (qkv_pack_fp16_forward_device(stream, d_src, d_dst, head_dim, num_tokens, num_heads) != 0) {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaStreamDestroy(stream);
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, static_cast<size_t>(n_dst) * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFreeAsync(d_src, stream);
    cudaFreeAsync(d_dst, stream);
    cudaStreamDestroy(stream);
}
