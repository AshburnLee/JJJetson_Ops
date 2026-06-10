// RoPE NeoX 优化
// Host 预计算 cos/sin cache，Device 访问cache即可
#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

// 与rope_neox.cu 完全相同
template <bool forward>
static __global__ void
rope_neox_search_table_kernel(const float *x, float *dst, const int ne0, const int ne1,
                              const int es1, const int es2, const int n_dims,
                              const float *cos_sin /* [n_tokens][n_dims/2][2] cos,sin */) {
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
    const int cs_idx = (id_token * half + id_fast) * 2;
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

// sin cos 计算只与维度长度有关，与维度中具体数值无关，这是使用 sin/cos cache 的前提
static void cache_sin_cos_table(const int *pos, const int n_tokens, const int n_dims,
                                const float theta_scale, float *cos_sin_host) {
    const int half = n_dims / 2;
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < half; ++i) {
            const float theta =
                static_cast<float>(pos[t]) * powf(theta_scale, static_cast<float>(i));
            const int idx = (t * half + i) * 2;
            cos_sin_host[idx + 0] = cosf(theta);
            cos_sin_host[idx + 1] = sinf(theta);
        }
    }
}

extern "C" void rope_search_table(float *input, int *pos, float *output,
                                  std::vector<int> &input_dims) {
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
    const int half = n_dims / 2;
    const int64_t cos_sin_elems = static_cast<int64_t>(n_tokens) * half * 2;

    const float freq_base = 10000.0f;
    const float theta_scale = powf(freq_base, -2.0f / static_cast<float>(n_dims));

    std::vector<float> cos_sin_host(static_cast<size_t>(cos_sin_elems));
    cache_sin_cos_table(pos, n_tokens, n_dims, theta_scale, cos_sin_host.data());

    float *d_x = nullptr;
    float *d_y = nullptr;
    float *d_cos_sin = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_x, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_cos_sin, cos_sin_elems * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_x, input, n_elem * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cos_sin, cos_sin_host.data(), cos_sin_elems * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    const dim3 threads(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x =
        (static_cast<int>(ne0_0) + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 blocks(static_cast<unsigned>(nr), static_cast<unsigned>(n_blocks_x), 1);

#if defined(MY_OPS_DEBUG)
    std::printf("rope_search_table launch: block=(%u,%u,%u), grid=(%u,%u,%u)\n", threads.x,
                threads.y, threads.z, blocks.x, blocks.y, blocks.z);
    std::fflush(stdout);
#endif

    rope_neox_search_table_kernel<true><<<blocks, threads, 0, stream>>>(
        d_x, d_y, static_cast<int>(ne0_0), static_cast<int>(ne0_1), static_cast<int>(s0_1),
        static_cast<int>(s0_2), n_dims, d_cos_sin);
    LAUNCH_CHECK();

    CUDA_CHECK(
        cudaMemcpyAsync(output, d_y, n_elem * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, stream));
    CUDA_CHECK(cudaFreeAsync(d_y, stream));
    CUDA_CHECK(cudaFreeAsync(d_cos_sin, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
