// NeoX RoPE kernel，查表 model 层提供的 d_cos_sin
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"
#include "rope_cossin_cache.h"

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

extern "C" void rope_with_global_cossin_cache(float *input, int *pos, float *output,
                                              std::vector<int> &input_dims,
                                              const RopeCosSinCache *cache) {
    if (cache == nullptr) {
        std::fprintf(stderr,
                     "rope_with_global_cossin_cache: cache is empty; create RopeCosSinCache when "
                     "model loading\n");
        return;
    }

    const float *d_cos_sin = rope_cossin_cache_device_ptr(cache);
    if (d_cos_sin == nullptr) {
        std::fprintf(
            stderr,
            "rope_with_global_cossin_cache: cache d_cos_sin is empty; create RopeCosSinCache "
            "when model loading\n");
        return;
    }

    const int64_t ne0_0 = input_dims[0];
    const int64_t ne0_1 = input_dims[1];
    const int64_t ne0_2 = input_dims[2];
    const int64_t ne0_3 = input_dims[3];

    const int64_t s0_1 = ne0_0;
    const int64_t s0_2 = ne0_0 * ne0_1;
    const int64_t n_elem = ne0_0 * ne0_1 * ne0_2 * ne0_3;
    const int64_t nr = ne0_1 * ne0_2 * ne0_3;

    const int n_dims = static_cast<int>(ne0_0);
    const int n_tokens = static_cast<int>(ne0_2);

    if (n_dims != rope_cossin_cache_n_dims(cache)) {
        std::fprintf(stderr,
                     "rope_with_global_cossin_cache: n_dims=%d mismatch with cache n_dims=%d\n",
                     n_dims, rope_cossin_cache_n_dims(cache));
        return;
    }

    if (rope_cossin_cache_check_pos(cache, pos, n_tokens) != 0) {
        return;
    }

    float *d_x = nullptr;
    float *d_y = nullptr;
    int *d_pos = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_x, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_pos, static_cast<size_t>(n_tokens) * sizeof(int), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_x, input, n_elem * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos, pos, static_cast<size_t>(n_tokens) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    const dim3 threads(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (static_cast<int>(ne0_0) + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 blocks(static_cast<unsigned>(nr), static_cast<unsigned>(n_blocks_x), 1);

#if defined(MY_OPS_DEBUG)
    std::printf("rope_with_global_cossin_cache launch: block=(%u,%u,%u), grid=(%u,%u,%u)\n",
                threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z);
    std::fflush(stdout);
#endif

    rope_neox_global_cache_kernel<true><<<blocks, threads, 0, stream>>>(
        d_x, d_y, static_cast<int>(ne0_0), static_cast<int>(ne0_1), static_cast<int>(s0_1),
        static_cast<int>(s0_2), n_dims, d_pos, d_cos_sin);
    LAUNCH_CHECK();

    CUDA_CHECK(
        cudaMemcpyAsync(output, d_y, n_elem * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, stream));
    CUDA_CHECK(cudaFreeAsync(d_y, stream));
    CUDA_CHECK(cudaFreeAsync(d_pos, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
