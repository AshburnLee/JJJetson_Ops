#include "rope_cossin_cache.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_utils.h"

struct RopeCosSinCache {
    float *d_cos_sin{nullptr}; // Device ptr，layout: [max_len][n_dims/2][2]
    int max_len{0};
    int n_dims{0};
    float freq_base{10000.f};
};

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

extern "C" RopeCosSinCache *rope_cossin_cache_create(int max_len, int n_dims, float freq_base) {
    if (max_len <= 0 || n_dims <= 0 || (n_dims % 2) != 0) {
        std::fprintf(stderr,
                     "rope_cossin_cache_create: invalid max_len=%d n_dims=%d (n_dims must be "
                     "even)\n",
                     max_len, n_dims);
        return nullptr;
    }

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

    auto *cache = new RopeCosSinCache{};
    cache->d_cos_sin = d_cos_sin;
    cache->max_len = max_len;
    cache->n_dims = n_dims;
    cache->freq_base = freq_base;
    return cache;
}

extern "C" void rope_cossin_cache_destroy(RopeCosSinCache *cache) {
    if (cache == nullptr) {
        return;
    }
    if (cache->d_cos_sin != nullptr) {
        cudaFree(cache->d_cos_sin);
        cache->d_cos_sin = nullptr;
    }
    delete cache;
}

extern "C" const float *rope_cossin_cache_device_ptr(const RopeCosSinCache *cache) {
    return cache != nullptr ? cache->d_cos_sin : nullptr;
}

extern "C" int rope_cossin_cache_max_len(const RopeCosSinCache *cache) {
    return cache != nullptr ? cache->max_len : 0;
}

extern "C" int rope_cossin_cache_n_dims(const RopeCosSinCache *cache) {
    return cache != nullptr ? cache->n_dims : 0;
}

// RoPE forward 前校验 pos：kernel 用 pos[t] 作为 global cache 行号
// cs_idx = (pos[t] * half + id_fast) * 2，pos 越界会读到非法 GPU 内存
// 返回 0 表示通过，-1 表示失败（错误信息写 stderr）
extern "C" int rope_cossin_cache_check_pos(const RopeCosSinCache *cache, const int *pos,
                                           int n_tokens) {
    // 1. cache 必须已在 model load 时创建并完成 H2D
    if (cache == nullptr || cache->d_cos_sin == nullptr) {
        std::fprintf(stderr, "rope_cossin_cache_check_pos: cache is null or not initialized\n");
        return -1;
    }
    // 2. pos 长度须与本步 n_tokens 数一致
    /*
    +---------------+-----------+---------------------+
    |    阶段       | n_tokens  |        pos          |
    +---------------+-----------+---------------------+
    | Prefill       |    13     | [0,1,2,...,12]      |
    | 第 1 次 Decode|     1     | [13]                |
    | 第 2 次 Decode|     1     | [14]                |
    +---------------+-----------+---------------------+
    */
    if (pos == nullptr || n_tokens <= 0) {
        std::fprintf(stderr, "rope_cossin_cache_check_pos: invalid pos or n_tokens=%d\n", n_tokens);
        return -1;
    }
    // 3. 每个 pos[t] 为绝对位置，要在 初始化时 预填范围 [0, max_len) 内
    for (int t = 0; t < n_tokens; ++t) {
        if (pos[t] < 0 || pos[t] >= cache->max_len) {
            std::fprintf(stderr, "rope_cossin_cache_check_pos: pos[%d]=%d out of range [0, %d)\n",
                         t, pos[t], cache->max_len);
            return -1;
        }
    }
    return 0;
}
