#include "kv_cache.h"

#include <cstdio>
#include <vector>

#include "cuda_utils.h"

extern "C" void kv_cache_append_launch_device(void *stream, const float *d_k, const float *d_v,
                                              float *d_k_cache, float *d_v_cache, int head_dim,
                                              int num_kv_heads, int max_seq, int offset,
                                              int n_tokens);

struct KVCacheLayerBuffers {
    float *d_k = nullptr;
    float *d_v = nullptr;
};

struct KVCache {
    int max_seq = 0;
    int head_dim = 0;
    int num_kv_heads = 0;
    int num_layers = 0;
    int cache_len = 0;
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
    std::vector<KVCacheLayerBuffers> layers;
};

static size_t kv_cache_layer_bytes(int head_dim, int max_seq, int num_kv_heads) {
    return static_cast<size_t>(head_dim) * max_seq * num_kv_heads * sizeof(float);
}

static int kv_cache_layer_checker(const KVCache *cache, int layer) {
    if (cache == nullptr) {
        std::fprintf(stderr, "kv_cache: cache is null\n");
        return -1;
    }
    if (layer < 0 || layer >= cache->num_layers) {
        std::fprintf(stderr, "kv_cache: invalid layer=%d (num_layers=%d)\n", layer,
                     cache->num_layers);
        return -1;
    }
    return 0;
}

static int kv_cache_append_checker(const KVCache *cache, int n_tokens, int offset) {
    if (cache == nullptr) {
        std::fprintf(stderr, "kv_cache_append: cache is null\n");
        return -1;
    }
    if (n_tokens <= 0) {
        std::fprintf(stderr, "kv_cache_append: invalid n_tokens=%d\n", n_tokens);
        return -1;
    }
    if (offset < 0 || offset > cache->cache_len) {
        std::fprintf(stderr, "kv_cache_append: invalid offset=%d (cache_len=%d)\n", offset,
                     cache->cache_len);
        return -1;
    }
    if (offset + n_tokens > cache->max_seq) {
        std::fprintf(stderr, "kv_cache_append: offset=%d + n_tokens=%d exceeds max_seq=%d\n",
                     offset, n_tokens, cache->max_seq);
        return -1;
    }
    return 0;
}

extern "C" KVCache *kv_cache_create(int max_seq, int head_dim, int num_kv_heads, int num_layers) {
    if (max_seq <= 0 || head_dim <= 0 || num_kv_heads <= 0 || num_layers <= 0) {
        std::fprintf(
            stderr,
            "kv_cache_create: invalid max_seq=%d head_dim=%d num_kv_heads=%d num_layers=%d\n",
            max_seq, head_dim, num_kv_heads, num_layers);
        return nullptr;
    }

    auto *cache = new KVCache{};
    cache->max_seq = max_seq;
    cache->head_dim = head_dim;
    cache->num_kv_heads = num_kv_heads;
    cache->num_layers = num_layers;
    cache->cache_len = 0;
    cache->owns_stream = true;
    CUDA_CHECK(cudaStreamCreateWithFlags(&cache->stream, cudaStreamNonBlocking));

    const size_t layer_bytes = kv_cache_layer_bytes(head_dim, max_seq, num_kv_heads);
    cache->layers.resize(static_cast<size_t>(num_layers));
    // 初始化为每一层 Device 的 k/v 值为 0
    for (int layer = 0; layer < num_layers; ++layer) {
        CUDA_CHECK(cudaMalloc(&cache->layers[layer].d_k, layer_bytes));
        CUDA_CHECK(cudaMalloc(&cache->layers[layer].d_v, layer_bytes));
        CUDA_CHECK(cudaMemset(cache->layers[layer].d_k, 0, layer_bytes));
        CUDA_CHECK(cudaMemset(cache->layers[layer].d_v, 0, layer_bytes));
    }
    // 返回指向 Device 的指针
    return cache;
}

extern "C" void kv_cache_destroy(KVCache *cache) {
    if (cache == nullptr) {
        return;
    }

    for (auto &layer : cache->layers) {
        cudaFree(layer.d_k);
        cudaFree(layer.d_v);
    }
    cache->layers.clear();

    if (cache->owns_stream && cache->stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(cache->stream));
    }

    delete cache;
}

extern "C" void kv_cache_reset(KVCache *cache) {
    if (cache == nullptr) {
        return;
    }
    cache->cache_len = 0;
}

extern "C" int kv_cache_get_len(const KVCache *cache) {
    return cache != nullptr ? cache->cache_len : 0;
}

extern "C" int kv_cache_get_max_seq(const KVCache *cache) {
    return cache != nullptr ? cache->max_seq : 0;
}

extern "C" int kv_cache_get_head_dim(const KVCache *cache) {
    return cache != nullptr ? cache->head_dim : 0;
}

extern "C" int kv_cache_get_num_kv_heads(const KVCache *cache) {
    return cache != nullptr ? cache->num_kv_heads : 0;
}

extern "C" int kv_cache_get_num_layers(const KVCache *cache) {
    return cache != nullptr ? cache->num_layers : 0;
}

extern "C" const float *kv_cache_get_k_device_ptr(const KVCache *cache, int layer) {
    if (kv_cache_layer_checker(cache, layer) != 0) {
        return nullptr;
    }
    return cache->layers[layer].d_k;
}

extern "C" const float *kv_cache_get_v_device_ptr(const KVCache *cache, int layer) {
    if (kv_cache_layer_checker(cache, layer) != 0) {
        return nullptr;
    }
    return cache->layers[layer].d_v;
}

// -========================-- 生产（device）--========================-

extern "C" int kv_cache_append_device(void *stream, KVCache *cache, int layer, const float *d_k,
                                      const float *d_v, int n_tokens) {
    const int offset = cache != nullptr ? cache->cache_len : 0;
    if (kv_cache_layer_checker(cache, layer) != 0 ||
        kv_cache_append_checker(cache, n_tokens, offset) != 0) {
        return -1;
    }
    if (d_k == nullptr || d_v == nullptr) {
        std::fprintf(stderr, "kv_cache_append_device: d_k/d_v is null\n");
        return -1;
    }

    cudaStream_t s = stream != nullptr ? static_cast<cudaStream_t>(stream) : cache->stream;

    kv_cache_append_launch_device(s, d_k, d_v, cache->layers[layer].d_k, cache->layers[layer].d_v,
                                  cache->head_dim, cache->num_kv_heads, cache->max_seq, offset,
                                  n_tokens);

    CUDA_CHECK(cudaStreamSynchronize(s));
    return 0;
}

/*
cache_len 是 全模型共享的序列长度, 所有layer append kv后 才会 advance_len

layer 0: append K/V  ->  offset = cache_len（如 0）
layer 1: append K/V  ->  仍是 offset = 0
...
layer N-1: append K/V
advance_len(T)       ->  cache_len += T
*/
extern "C" int kv_cache_advance_len(KVCache *cache, int n_tokens) {
    if (cache == nullptr) {
        std::fprintf(stderr, "kv_cache_advance_len: cache is null\n");
        return -1;
    }
    if (n_tokens <= 0) {
        std::fprintf(stderr, "kv_cache_advance_len: invalid n_tokens=%d\n", n_tokens);
        return -1;
    }
    if (cache->cache_len + n_tokens > cache->max_seq) {
        std::fprintf(stderr,
                     "kv_cache_advance_len: cache_len=%d + n_tokens=%d exceeds max_seq=%d\n",
                     cache->cache_len, n_tokens, cache->max_seq);
        return -1;
    }
    cache->cache_len += n_tokens;
    return 0;
}
