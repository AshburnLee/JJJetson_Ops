#include "inference_engine.h"

#include <stdio.h>
#include <unordered_map>

#include "cublas_utils.cuh"
#include "cuda_utils.h"
#include "kv_cache.h"
#include "transformer_model.h"

struct InferenceEngineBufferPool {
    int max_seq_len = 0;
    int head_dim = 0;
    int num_kv_heads = 0;

    // FA staging，全 layer 共用（与 Phase 1 Runner 同尺寸）
    uint16_t *d_k_fa_fp16 = nullptr;
    uint16_t *d_v_fa_fp16 = nullptr;

    // 按 T 懒分配；forward 实现后用于 hidden/logits/d_pos
    std::unordered_map<int, int *> d_pos_by_tokens;
};

struct InferenceEngineSessionState {
    int next_pos = 0;
};

struct InferenceEngine {
    TransformerModel *model = nullptr;
    KVCache *kv_cache = nullptr;
    InferenceEngineBufferPool pool{};
    InferenceEngineSessionState session{};
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    bool owns_stream = false;
};

static void inference_engine_buffer_pool_destroy(InferenceEngineBufferPool *pool) {
    if (pool == nullptr) {
        return;
    }
    for (auto &entry : pool->d_pos_by_tokens) {
        CUDA_CHECK(cudaFree(entry.second));
    }
    pool->d_pos_by_tokens.clear();
    if (pool->d_k_fa_fp16 != nullptr) {
        CUDA_CHECK(cudaFree(pool->d_k_fa_fp16));
        pool->d_k_fa_fp16 = nullptr;
    }
    if (pool->d_v_fa_fp16 != nullptr) {
        CUDA_CHECK(cudaFree(pool->d_v_fa_fp16));
        pool->d_v_fa_fp16 = nullptr;
    }
}

extern "C" InferenceEngine *inference_engine_create(TransformerModel *model, void *stream_in) {
    if (model == nullptr) {
        std::fprintf(stderr, "inference_engine_create: null model\n");
        return nullptr;
    }
    const ModelConfig *cfg = transformer_model_get_config(model);
    if (cfg == nullptr || model_config_validate(cfg) != 0) {
        std::fprintf(stderr, "inference_engine_create: invalid model config\n");
        return nullptr;
    }

    auto *engine = new InferenceEngine{};
    engine->model = model;
    engine->owns_stream = (stream_in == nullptr);
    engine->stream = engine->owns_stream ? nullptr : static_cast<cudaStream_t>(stream_in);
    if (engine->owns_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&engine->stream, cudaStreamNonBlocking));
    }

    CUBLAS_CHECK(cublasCreate(&engine->cublas_handle));

    engine->kv_cache =
        kv_cache_create(cfg->max_seq_len, cfg->head_dim, cfg->num_kv_heads, cfg->num_layers);
    if (engine->kv_cache == nullptr) {
        inference_engine_destroy(engine);
        return nullptr;
    }

    engine->pool.max_seq_len = cfg->max_seq_len;
    engine->pool.head_dim = cfg->head_dim;
    engine->pool.num_kv_heads = cfg->num_kv_heads;

    const size_t fa_fp16_bytes = static_cast<size_t>(cfg->head_dim) * cfg->max_seq_len *
                                 cfg->num_kv_heads * sizeof(uint16_t);
    CUDA_CHECK(cudaMalloc(&engine->pool.d_k_fa_fp16, fa_fp16_bytes));
    CUDA_CHECK(cudaMalloc(&engine->pool.d_v_fa_fp16, fa_fp16_bytes));

    engine->session.next_pos = 0;
    return engine;
}

extern "C" void inference_engine_destroy(InferenceEngine *engine) {
    if (engine == nullptr) {
        return;
    }
    inference_engine_buffer_pool_destroy(&engine->pool);
    kv_cache_destroy(engine->kv_cache);
    engine->kv_cache = nullptr;
    if (engine->cublas_handle != nullptr) {
        cublasDestroy(engine->cublas_handle);
        engine->cublas_handle = nullptr;
    }
    if (engine->owns_stream && engine->stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(engine->stream));
        engine->stream = nullptr;
    }
    delete engine;
}

extern "C" void inference_engine_reset(InferenceEngine *engine) {
    if (engine == nullptr) {
        return;
    }
    kv_cache_reset(engine->kv_cache);
    engine->session.next_pos = 0;
}

extern "C" const TransformerModel *inference_engine_get_model(const InferenceEngine *engine) {
    return engine != nullptr ? engine->model : nullptr;
}

extern "C" KVCache *inference_engine_get_kv_cache(InferenceEngine *engine) {
    return engine != nullptr ? engine->kv_cache : nullptr;
}

extern "C" int inference_engine_kv_cache_len(const InferenceEngine *engine) {
    if (engine == nullptr || engine->kv_cache == nullptr) {
        return 0;
    }
    return kv_cache_get_len(engine->kv_cache);
}

extern "C" int inference_engine_next_pos(const InferenceEngine *engine) {
    if (engine == nullptr) {
        return 0;
    }
    return engine->session.next_pos;
}
