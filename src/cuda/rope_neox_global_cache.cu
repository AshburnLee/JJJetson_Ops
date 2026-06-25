// RoPE NeoX + 生产推理用全局 cos/sin cache
// 模型加载时 初始化 Global Cache；Prefill/Decode forward 仅传 pos，按绝对位置查表
#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

// init 一次、多次 forward 复用, 故属于推理 runtime / engine 的资源层，不是纯算子
// TODO: Move to ../model/
struct RopeGlobalCosSinCache {
    float *d_cos_sin{nullptr}; // [max_len][n_dims/2][2]
    int max_len{0};
    int n_dims{0};
    float freq_base{10000.f};
    bool initialized{false};

    int half_dims() const { return n_dims / 2; }
    int64_t floats_per_pos() const { return static_cast<int64_t>(half_dims()) * 2; }
};

static RopeGlobalCosSinCache g_rope_cos_sin_cache;

extern "C" void destroy_cossin_cache();

static void fill_cossin_global_host(const int max_len, const int n_dims, const float theta_scale,
                                    float *cos_sin_host) {
    const int half = n_dims / 2;
    for (int p = 0; p < max_len; ++p) {
        for (int i = 0; i < half; ++i) {
            const float theta = static_cast<float>(p) * powf(theta_scale, static_cast<float>(i));
            const int idx = (p * half + i) * 2;
            cos_sin_host[idx + 0] = cosf(theta);
            cos_sin_host[idx + 1] = sinf(theta);
        }
    }
}

extern "C" void init_cossin_cache(int max_len, int n_dims, float freq_base) {
    if (max_len <= 0 || n_dims <= 0 || (n_dims % 2) != 0) {
        std::fprintf(stderr,
                     "init_cossin_cache: invalid max_len=%d n_dims=%d (n_dims "
                     "must be even)\n",
                     max_len, n_dims);
        return;
    }

    destroy_cossin_cache();

    const float theta_scale = powf(freq_base, -2.0f / static_cast<float>(n_dims));
    const int64_t total_floats =
        static_cast<int64_t>(max_len) * static_cast<int64_t>(n_dims / 2) * 2;

    std::vector<float> cos_sin_host(static_cast<size_t>(total_floats));
    fill_cossin_global_host(max_len, n_dims, theta_scale, cos_sin_host.data());

    float *d_cos_sin = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cos_sin, static_cast<size_t>(total_floats) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cos_sin, cos_sin_host.data(),
                          static_cast<size_t>(total_floats) * sizeof(float),
                          cudaMemcpyHostToDevice));

    g_rope_cos_sin_cache.d_cos_sin = d_cos_sin;
    g_rope_cos_sin_cache.max_len = max_len;
    g_rope_cos_sin_cache.n_dims = n_dims;
    g_rope_cos_sin_cache.freq_base = freq_base;
    g_rope_cos_sin_cache.initialized = true;
}

extern "C" void destroy_cossin_cache() {
    if (g_rope_cos_sin_cache.d_cos_sin != nullptr) {
        cudaFree(g_rope_cos_sin_cache.d_cos_sin);
        g_rope_cos_sin_cache.d_cos_sin = nullptr;
    }
    g_rope_cos_sin_cache.max_len = 0;
    g_rope_cos_sin_cache.n_dims = 0;
    g_rope_cos_sin_cache.initialized = false;
}

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
                                              std::vector<int> &input_dims) {
    if (!g_rope_cos_sin_cache.initialized || g_rope_cos_sin_cache.d_cos_sin == nullptr) {
        std::fprintf(stderr,
                     "rope_with_global_cossin_cache: global cos/sin cache not initialized; call "
                     "init_cossin_cache(max_len, n_dims) first\n");
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

    if (n_dims != g_rope_cos_sin_cache.n_dims) {
        std::fprintf(stderr,
                     "rope_with_global_cossin_cache: n_dims=%d mismatch with cache n_dims=%d\n",
                     n_dims, g_rope_cos_sin_cache.n_dims);
        return;
    }

    for (int t = 0; t < n_tokens; ++t) {
        if (pos[t] < 0 || pos[t] >= g_rope_cos_sin_cache.max_len) {
            std::fprintf(stderr,
                         "rope_with_global_cossin_cache: pos[%d]=%d out of cache range [0, %d)\n",
                         t, pos[t], g_rope_cos_sin_cache.max_len);
            return;
        }
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
        static_cast<int>(s0_2), n_dims, d_pos, g_rope_cos_sin_cache.d_cos_sin);
    LAUNCH_CHECK();

    CUDA_CHECK(
        cudaMemcpyAsync(output, d_y, n_elem * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, stream));
    CUDA_CHECK(cudaFreeAsync(d_y, stream));
    CUDA_CHECK(cudaFreeAsync(d_pos, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
