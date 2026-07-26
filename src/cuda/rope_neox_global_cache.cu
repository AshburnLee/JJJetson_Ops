// NeoX RoPE kernel，查表 model 层提供的 d_cos_sin
#include "rope.h"

#include <cuda_runtime.h>
#include <cstdio>

#include "cuda_utils.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

template <bool forward>
static __global__ void
rope_neox_global_cache_kernel(const float *x, float *dst, const int ne0, const int ne1,
                              const int es1, const int es2, const int n_dims, const int *pos,
                              const float *cos_sin /* [max_len][n_dims/2][2] */) {
    const int id_fast = threadIdx.y + blockDim.y * blockIdx.y;

    if (id_fast >= ne0) {
        return;
    }

    const int id_flat_ht = threadIdx.x + blockDim.x * blockIdx.x;
    const int id_head = id_flat_ht % ne1;
    const int id_token = id_flat_ht / ne1;

    const int id_dst = id_fast + ne0 * id_flat_ht;
    const int id_x = id_fast + es1 * id_head + es2 * id_token;

    if (id_fast >= n_dims) {
        dst[id_dst] = x[id_x];
        return;
    }
    if (id_fast >= n_dims / 2) {
        return;
    }

    const int half = n_dims / 2;
    const int pos_id = pos[id_token];
    const int cs_idx = (pos_id * half + id_fast) * 2;
    const float cos_theta = cos_sin[cs_idx + 0];
    float sin_theta = cos_sin[cs_idx + 1];
    if (!forward) {
        sin_theta *= -1.0f;
    }

    const float x0 = x[id_x + 0];
    const float x1 = x[id_x + n_dims / 2];

    dst[id_dst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[id_dst + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

static int rope_neox_check_shape(const RopeCosSinCache *cache, int head_dim, int num_heads,
                                 int num_tokens, int batch) {
    if (cache == nullptr || rope_cossin_cache_device_ptr(cache) == nullptr) {
        std::fprintf(stderr, "rope_neox_forward_device: cache is null or not initialized\n");
        return -1;
    }
    if (head_dim <= 0 || num_heads <= 0 || num_tokens <= 0 || batch <= 0) {
        std::fprintf(stderr,
                     "rope_neox_forward_device: invalid shape head_dim=%d num_heads=%d "
                     "num_tokens=%d batch=%d\n",
                     head_dim, num_heads, num_tokens, batch);
        return -1;
    }
    if (head_dim != rope_cossin_cache_n_dims(cache)) {
        std::fprintf(stderr, "rope_neox_forward_device: head_dim=%d mismatch cache n_dims=%d\n",
                     head_dim, rope_cossin_cache_n_dims(cache));
        return -1;
    }
    return 0;
}

static void rope_neox_launch_device(cudaStream_t stream, const float *d_input, float *d_output,
                                    const int *d_pos, int head_dim, int num_heads, int num_tokens,
                                    int batch, const float *d_cos_sin) {
    const int es1 = head_dim;
    const int es2 = head_dim * num_heads;
    const int nr = num_heads * num_tokens * batch;

    const dim3 threads(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (head_dim + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 blocks(static_cast<unsigned>(nr), static_cast<unsigned>(n_blocks_x), 1);

#if defined(MY_OPS_DEBUG)
    std::printf("rope_neox_forward_device launch: block=(%u,%u,%u), grid=(%u,%u,%u)\n", threads.x,
                threads.y, threads.z, blocks.x, blocks.y, blocks.z);
    std::fflush(stdout);
#endif

    rope_neox_global_cache_kernel<true><<<blocks, threads, 0, stream>>>(
        d_input, d_output, head_dim, num_heads, es1, es2, head_dim, d_pos, d_cos_sin);
    LAUNCH_CHECK();
}

// -========================-- 生产（device）--========================-
// ★ 业务入口（对外声明见 src/cuda/rope.h）
// TransformerRunner 等在 Linear 产出 d_q/d_k 后应直接调用本函数：
//   - d_input / d_output / d_pos 均已在 GPU，本路径不做 H2D/D2H
//   - d_output 可与 d_input 相同（in-place）
// rope_neox_forward_host 与 Python forward_host 仅为测试包装，内部也会调用
// rope_neox_forward_device。
extern "C" int rope_neox_forward_device(void *stream, const RopeCosSinCache *cache,
                                        const float *d_input, float *d_output, const int *d_pos,
                                        int head_dim, int num_heads, int num_tokens, int batch) {
    if (rope_neox_check_shape(cache, head_dim, num_heads, num_tokens, batch) != 0) {
        return -1;
    }
    if (d_input == nullptr || d_output == nullptr || d_pos == nullptr) {
        std::fprintf(stderr, "rope_neox_forward_device: d_input/d_output/d_pos is null\n");
        return -1;
    }

    cudaStream_t s = stream != nullptr ? static_cast<cudaStream_t>(stream) : nullptr;
    if (s == nullptr) {
        std::fprintf(stderr, "rope_neox_forward_device: stream is null\n");
        return -1;
    }

    rope_neox_launch_device(s, d_input, d_output, d_pos, head_dim, num_heads, num_tokens, batch,
                            rope_cossin_cache_device_ptr(cache));
    return 0;
}

// ======================== 仅供 Python 测试 ================================
extern "C" void rope_neox_forward_host(float *input, int *pos, float *output, int head_dim,
                                       int num_heads, int num_tokens, int batch,
                                       const RopeCosSinCache *cache) {
    if (cache == nullptr) {
        std::fprintf(stderr, "rope_neox_forward_host: cache is empty; create RopeCosSinCache when "
                             "model loading\n");
        return;
    }
    if (input == nullptr || pos == nullptr || output == nullptr) {
        std::fprintf(stderr, "rope_neox_forward_host: null pointer argument\n");
        return;
    }

    const int64_t n_elem = static_cast<int64_t>(head_dim) * num_heads * num_tokens * batch;

    if (rope_neox_check_shape(cache, head_dim, num_heads, num_tokens, batch) != 0) {
        return;
    }
    if (rope_cossin_cache_check_pos(cache, pos, num_tokens) != 0) {
        return;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_x = nullptr;
    float *d_y = nullptr;
    int *d_pos = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_x, static_cast<size_t>(n_elem) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, static_cast<size_t>(n_elem) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_pos, static_cast<size_t>(num_tokens) * sizeof(int), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_x, input, static_cast<size_t>(n_elem) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos, pos, static_cast<size_t>(num_tokens) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    if (rope_neox_forward_device(stream, cache, d_x, d_y, d_pos, head_dim, num_heads, num_tokens,
                                 batch) != 0) {
        CUDA_CHECK(cudaFreeAsync(d_x, stream));
        CUDA_CHECK(cudaFreeAsync(d_y, stream));
        CUDA_CHECK(cudaFreeAsync(d_pos, stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(output, d_y, static_cast<size_t>(n_elem) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, stream));
    CUDA_CHECK(cudaFreeAsync(d_y, stream));
    CUDA_CHECK(cudaFreeAsync(d_pos, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
