#include "inference_engine.h"

#include <cstdio>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#include "cublas_utils.cuh"
#include "cuda_utils.h"
#include "kv_cache.h"
#include "nvtx_range.h"
#include "transformer_model.h"
#include "transformer_runner.h"

struct InferenceEngineBufferPool {
    int max_seq_len = 0;
    int head_dim = 0;
    int num_kv_heads = 0;
    int vocab_size = 0;
    int hidden_size = 0;
    int last_num_tokens = 0;

    uint16_t *d_k_fa_fp16 = nullptr;
    uint16_t *d_v_fa_fp16 = nullptr;

    int *d_token_ids = nullptr;
    float *d_logits = nullptr;
    float *d_hidden_out = nullptr;
    int *d_out_token = nullptr;

    std::unordered_map<int, TransformerLayerLinearDeviceBuffers *> layer_buffers_by_tokens;
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

static size_t inference_col_major_bytes(int features, int num_tokens) {
    return static_cast<size_t>(features) * num_tokens * sizeof(float);
}

static void inference_engine_buffer_pool_destroy(InferenceEngineBufferPool *pool) {
    if (pool == nullptr) {
        return;
    }
    for (auto &entry : pool->layer_buffers_by_tokens) {
        transformer_layer_linear_buffers_destroy(entry.second);
    }
    pool->layer_buffers_by_tokens.clear();
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
    if (pool->d_token_ids != nullptr) {
        CUDA_CHECK(cudaFree(pool->d_token_ids));
        pool->d_token_ids = nullptr;
    }
    if (pool->d_logits != nullptr) {
        CUDA_CHECK(cudaFree(pool->d_logits));
        pool->d_logits = nullptr;
    }
    if (pool->d_hidden_out != nullptr) {
        CUDA_CHECK(cudaFree(pool->d_hidden_out));
        pool->d_hidden_out = nullptr;
    }
    if (pool->d_out_token != nullptr) {
        CUDA_CHECK(cudaFree(pool->d_out_token));
        pool->d_out_token = nullptr;
    }
}

static TransformerLayerLinearDeviceBuffers *
inference_engine_layer_buffers_get(InferenceEngine *engine, int num_tokens, int hidden_size,
                                   int q_dim, int kv_dim, int intermediate_size, int head_dim,
                                   int num_q_heads, int num_kv_heads) {
    const auto it = engine->pool.layer_buffers_by_tokens.find(num_tokens);
    if (it != engine->pool.layer_buffers_by_tokens.end()) {
        return it->second;
    }
    TransformerLayerLinearDeviceBuffers *buffers = transformer_layer_linear_buffers_create(
        num_tokens, hidden_size, q_dim, kv_dim, intermediate_size, head_dim, num_q_heads,
        num_kv_heads);
    engine->pool.layer_buffers_by_tokens[num_tokens] = buffers;
    return buffers;
}

static int *inference_engine_d_pos_get(InferenceEngine *engine, int num_tokens, int pos_offset,
                                       cudaStream_t stream) {
    int *d_pos = nullptr;
    const auto it = engine->pool.d_pos_by_tokens.find(num_tokens);
    if (it != engine->pool.d_pos_by_tokens.end()) {
        d_pos = it->second;
    } else {
        CUDA_CHECK(cudaMalloc(&d_pos, static_cast<size_t>(num_tokens) * sizeof(int)));
        engine->pool.d_pos_by_tokens[num_tokens] = d_pos;
    }

    std::vector<int> pos_host(static_cast<size_t>(num_tokens));
    for (int t = 0; t < num_tokens; ++t) {
        pos_host[static_cast<size_t>(t)] = pos_offset + t;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_pos, pos_host.data(),
                               static_cast<size_t>(num_tokens) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    return d_pos;
}

// 创建 Phase 2 推理 session：在已有 Model 上挂 KV、buffer pool、stream，准备 prefill/decode。
//
// Big picture 里它在哪？
//   Model(create+load 权重) -> [本函数 engine_create] -> forward / reset / destroy
//   本函数 **借用** model 指针，不 load 权重、不 free Model；destroy Engine 后 Model 仍可复用。
//   对照 Phase 1：类似 transformer_runner_create，但权重在 Model 里，KV 是 num_layers 层。
//   图纸：doc/design/phase2_lifecycle.md §3.2
//
// 内部分配顺序（例：num_layers=2, max_seq_len=256, head_dim=32, num_kv_heads=2）：
//   INPUT：有效 Model*，cfg 已校验
//   step 1) Engine 壳 + 记录 model*（不拷贝权重）
//   step 2) stream：stream_in 为空则自建 non-blocking stream
//   step 3) cublasHandle
//   step 4) KVCache(256, 32, 2, num_layers=2)，cache_len=0
//   step 5) FA staging：d_k/d_v_fa_fp16 各 32*256*2*sizeof(fp16) bytes
//   step 6) session token/logits/hidden/out_token：按 max_seq 一次 malloc，decode 复用
//   step 7) 预热 T=1 layer workspace + d_pos[1]，避免第一次 decode 在 forward 里 cudaMalloc
//   OUTPUT：InferenceEngine*，session.next_pos=0；失败返回 nullptr 并释放已分配部分
//
// 调用方：Model 须先于 Engine create；Engine destroy 后 Model 才能 destroy。
extern "C" InferenceEngine *inference_engine_create(TransformerModel *model, void *stream_in) {
    // step 0：Model 与 cfg 必须合法
    if (model == nullptr) {
        std::fprintf(stderr, "inference_engine_create: null model\n");
        return nullptr;
    }
    const ModelConfig *cfg = transformer_model_get_config(model);
    if (cfg == nullptr || model_config_validate(cfg) != 0) {
        std::fprintf(stderr, "inference_engine_create: invalid model config\n");
        return nullptr;
    }

    // 最外层组件：Engine session（stream / cublas / KV / FA staging）。不拆子分配。
    name_engine_thread();
    NVTX_RANGE("create_engine");

    // step 1：Engine 壳，只存 model 引用
    auto *engine = new InferenceEngine{};
    engine->model = model;
    // step 2：stream（可外部传入，否则 Engine 自己建）
    engine->owns_stream = (stream_in == nullptr);
    engine->stream = engine->owns_stream ? nullptr : static_cast<cudaStream_t>(stream_in);
    if (engine->owns_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&engine->stream, cudaStreamNonBlocking));
    }

    // step 3：cublas（forward 里 layer / lm_head GEMM 用）
    CUBLAS_CHECK(cublasCreate(&engine->cublas_handle));

    // step 4：多 layer KV cache，create 后 cache_len=0
    engine->kv_cache =
        kv_cache_create(cfg->max_seq_len, cfg->head_dim, cfg->num_kv_heads, cfg->num_layers);
    if (engine->kv_cache == nullptr) {
        inference_engine_destroy(engine);
        return nullptr;
    }

    engine->pool.max_seq_len = cfg->max_seq_len;
    engine->pool.head_dim = cfg->head_dim;
    engine->pool.num_kv_heads = cfg->num_kv_heads;
    engine->pool.vocab_size = cfg->vocab_size;
    engine->pool.hidden_size = cfg->hidden_size;
    engine->pool.last_num_tokens = 0;

    // step 5：FA fp16 staging（全 layer 共用，尺寸按 max_seq）
    const size_t fa_fp16_bytes = static_cast<size_t>(cfg->head_dim) * cfg->max_seq_len *
                                 cfg->num_kv_heads * sizeof(uint16_t);
    CUDA_CHECK(cudaMalloc(&engine->pool.d_k_fa_fp16, fa_fp16_bytes));
    CUDA_CHECK(cudaMalloc(&engine->pool.d_v_fa_fp16, fa_fp16_bytes));

    // step 6：token / logits / hidden / 采样槽。例：vocab=512, hidden=128, max_seq=256
    //   d_logits = 512*256*4B；decode T=1 三次都写同一块，不再 cudaMallocAsync。
    CUDA_CHECK(
        cudaMalloc(&engine->pool.d_token_ids, static_cast<size_t>(cfg->max_seq_len) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&engine->pool.d_logits,
                          inference_col_major_bytes(cfg->vocab_size, cfg->max_seq_len)));
    CUDA_CHECK(cudaMalloc(&engine->pool.d_hidden_out,
                          inference_col_major_bytes(cfg->hidden_size, cfg->max_seq_len)));
    CUDA_CHECK(cudaMalloc(&engine->pool.d_out_token, sizeof(int)));

    // step 7：decode 永远 T=1。例：hidden=128 -> d_hidden 只占 128*1*4B；
    //   现在就把桶[1]填上。reset 不清这套；prefill T=3 仍走懒分配。
    const int q_dim = cfg->num_q_heads * cfg->head_dim;
    const int kv_dim = cfg->num_kv_heads * cfg->head_dim;
    inference_engine_layer_buffers_get(engine, /*num_tokens=*/1, cfg->hidden_size, q_dim, kv_dim,
                                       cfg->intermediate_size, cfg->head_dim, cfg->num_q_heads,
                                       cfg->num_kv_heads);
    int *d_pos_t1 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pos_t1, sizeof(int)));
    engine->pool.d_pos_by_tokens[1] = d_pos_t1;

    engine->session.next_pos = 0; // 与 kv cache_len=0 对齐，forward 后推进
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

extern "C" void *inference_engine_get_stream(InferenceEngine *engine) {
    if (engine == nullptr) {
        return nullptr;
    }
    return engine->stream;
}

// Phase 2 InferenceEngine 的单步生产入口：把 Model 里的权重和本 session 的 KV 拼成一次 forward。
//
// Big picture 里它在哪？
//   Loader -> Model(权重) -> [Engine 本函数] -> (将来 Sampler 拿 logits 采样)
//   Engine 只管 session：KVCache、buffer pool、prefill/decode 步进；不 own 权重，不 destroy Model。
//   对照 Phase 1：N=1 时等价于 TransformerRunner 的 forward_device + final_norm；
//   差别是权重从 Model 按 layer_idx 取，KV 是 N 层一块 cache。
//   图纸：doc/design/phase2_lifecycle.md §3.3
//
// 一步 prefill 里内部顺序（decode 同样，只是 T=1、pos 从 cache_len 起）：
//   例：T=3，改前 cache_len=0，ctx->d_pos=[0,1,2]
//   1) 入口 hidden：有 d_token_ids 则 embed -> buffers->d_hidden [hidden,T]；
//      否则 D2D d_hidden_in -> buffers->d_hidden（测试跳过 embed）
//   2) layer 0..N-1：每层 transformer_layer_linears_forward_device
//      - 层内 append KV，不改 cache_len；layer>0 时上一层输出 D2D 到本层输入
//   3) 全 layer 跑完后一次 kv_cache_advance_len(T) -> cache_len 变为 3
//   4) final_norm(Model) 作用在最后一层 d_hidden_out 上
//   5) D2D 到 ctx->d_hidden_out；若 ctx->d_logits 非空再 lm_head(Model)
//
// ctx 栈上构造，不持久化；调用方保证 d_pos 已在 GPU 上。
extern "C" int inference_engine_forward_device(InferenceEngine *engine,
                                               const InferenceForwardCtx *ctx) {
    if (engine == nullptr || ctx == nullptr) {
        return -1;
    }
    if (ctx->num_tokens <= 0 || ctx->d_hidden_out == nullptr || ctx->d_pos == nullptr) {
        std::fprintf(stderr, "inference_engine_forward_device: invalid ctx\n");
        return -1;
    }
    if (ctx->d_token_ids == nullptr && ctx->d_hidden_in == nullptr) {
        std::fprintf(stderr, "inference_engine_forward_device: need d_token_ids or d_hidden_in\n");
        return -1;
    }
    if (transformer_model_is_weights_loaded(engine->model) != 1) {
        std::fprintf(stderr, "inference_engine_forward_device: model weights not loaded\n");
        return -1;
    }

    const ModelConfig *cfg = transformer_model_get_config(engine->model);
    const int num_layers = cfg->num_layers;
    const int hidden_size = cfg->hidden_size;
    const int q_dim = cfg->num_q_heads * cfg->head_dim;
    const int kv_dim = cfg->num_kv_heads * cfg->head_dim;
    const int T = ctx->num_tokens;

    // step 1：会话边界检查——本步写入后不能超过 max_seq（例：L=250,T=10,max=256 -> 拒绝）
    const int cache_len_before = kv_cache_get_len(engine->kv_cache);
    if (cache_len_before + T > cfg->max_seq_len) {
        std::fprintf(stderr, "inference_engine_forward_device: exceeds max_seq_len\n");
        return -1;
    }

    cudaStream_t stream =
        ctx->stream != nullptr ? static_cast<cudaStream_t>(ctx->stream) : engine->stream;

    TransformerLayerLinearDeviceBuffers *buffers = inference_engine_layer_buffers_get(
        engine, T, hidden_size, q_dim, kv_dim, cfg->intermediate_size, cfg->head_dim,
        cfg->num_q_heads, cfg->num_kv_heads);

    // step 2：得到本步 layer 输入 hidden [hidden_size, T]（col-major）
    if (ctx->d_token_ids != nullptr) {
        if (transformer_model_embed_forward_device(stream, engine->model, ctx->d_token_ids,
                                                   buffers->d_hidden, T) != 0) {
            std::fprintf(stderr, "inference_engine_forward_device: embed failed\n");
            return -1;
        }
    } else {
        transformer_layer_copy_hidden_in(stream, buffers, ctx->d_hidden_in, hidden_size, T);
    }

    RopeCosSinCache *rope_cache = transformer_model_get_rope_cache(engine->model);

    // step 3：N 个 Pre-LN block；每层写 KV[layer_idx]，此处仍不 advance_len
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        if (layer_idx > 0) {
            transformer_layer_copy_hidden_in(stream, buffers, buffers->d_hidden_out, hidden_size,
                                             T);
        }
        const TransformerLayerWeights *layer_weights =
            transformer_model_get_layer_weights(engine->model, layer_idx);
        if (layer_weights == nullptr) {
            std::fprintf(stderr, "inference_engine_forward_device: null layer %d weights\n",
                         layer_idx);
            return -1;
        }
        transformer_layer_linears_forward_device(
            stream, engine->cublas_handle, buffers, layer_weights->d_w_input_layernorm,
            layer_weights->d_w_post_attention_layernorm, layer_weights->d_w_q, layer_weights->d_w_k,
            layer_weights->d_w_v, layer_weights->d_w_o, layer_weights->d_w_gate,
            layer_weights->d_w_up, layer_weights->d_w_down, rope_cache, ctx->d_pos,
            engine->kv_cache, layer_idx, engine->pool.d_k_fa_fp16, engine->pool.d_v_fa_fp16,
            cfg->max_seq_len, cfg->head_dim, cfg->num_q_heads, cfg->num_kv_heads,
            cfg->rms_norm_epsilon);
    }

    // step 4：全 layer KV append 完毕，推进 cache_len（例：L=0,T=3 -> L=3）
    if (kv_cache_advance_len(engine->kv_cache, T) != 0) {
        std::fprintf(stderr, "inference_engine_forward_device: kv_cache_advance_len failed\n");
        return -1;
    }
    engine->session.next_pos = kv_cache_get_len(engine->kv_cache);

    // step 5：最后一层 block 输出做 final RMSNorm（权重在 Model 上）
    if (transformer_model_final_norm_forward_device(stream, engine->model, buffers->d_hidden_out,
                                                    buffers->d_hidden_out, T) != 0) {
        std::fprintf(stderr, "inference_engine_forward_device: final_norm failed\n");
        return -1;
    }

    // step 6：写出 hidden；可选 lm_head -> logits [vocab, T]（GenerateLoop 将来用末 token slice）
    transformer_layer_copy_hidden_out(stream, ctx->d_hidden_out, buffers, hidden_size, T);

    if (ctx->d_logits != nullptr) {
        if (transformer_model_lm_head_forward_device(stream, engine->cublas_handle, engine->model,
                                                     buffers->d_hidden_out, ctx->d_logits,
                                                     T) != 0) {
            std::fprintf(stderr, "inference_engine_forward_device: lm_head failed\n");
            return -1;
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}

// _hidden_ 表示：这一步从 hidden 状态进，不走 embed
extern "C" int inference_engine_forward_hidden_host(InferenceEngine *engine,
                                                    const float *hidden_in_host,
                                                    float *hidden_out_host, int num_tokens,
                                                    int pos_offset) {
    if (engine == nullptr || hidden_in_host == nullptr || hidden_out_host == nullptr ||
        num_tokens <= 0) {
        return -1;
    }

    const ModelConfig *cfg = transformer_model_get_config(engine->model);
    if (cfg == nullptr) {
        return -1;
    }

    cudaStream_t stream = engine->stream;
    const int hidden_size = cfg->hidden_size;
    const size_t hidden_bytes = inference_col_major_bytes(hidden_size, num_tokens);

    float *d_hidden_in = nullptr;
    float *d_hidden_out = nullptr;
    int *d_pos = inference_engine_d_pos_get(engine, num_tokens, pos_offset, stream);

    CUDA_CHECK(cudaMallocAsync(&d_hidden_in, hidden_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_hidden_out, hidden_bytes, stream));

    CUDA_CHECK(
        cudaMemcpyAsync(d_hidden_in, hidden_in_host, hidden_bytes, cudaMemcpyHostToDevice, stream));

    InferenceForwardCtx ctx{};
    ctx.num_tokens = num_tokens;
    ctx.stream = stream;
    ctx.d_hidden_in = d_hidden_in;
    ctx.d_hidden_out = d_hidden_out;
    ctx.d_pos = d_pos;
    ctx.d_logits = nullptr;

    const int rc = inference_engine_forward_device(engine, &ctx);

    if (rc == 0) {
        CUDA_CHECK(cudaMemcpyAsync(hidden_out_host, d_hidden_out, hidden_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaFreeAsync(d_hidden_in, stream));
    CUDA_CHECK(cudaFreeAsync(d_hidden_out, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return rc;
}

// 编排/production：token embed + N×layer + lm_head；d_token_ids / d_logits 已在 GPU。
// d_hidden_out 用 create 时的 pool，不每步 malloc。
extern "C" int inference_engine_forward_token_device(InferenceEngine *engine,
                                                     const int *d_token_ids, float *d_logits,
                                                     int num_tokens, int pos_offset) {
    if (engine == nullptr || d_token_ids == nullptr || d_logits == nullptr || num_tokens <= 0) {
        return -1;
    }

    const ModelConfig *cfg = transformer_model_get_config(engine->model);
    if (cfg == nullptr) {
        return -1;
    }
    if (num_tokens > engine->pool.max_seq_len || engine->pool.d_hidden_out == nullptr) {
        return -1;
    }

    // 钉在 callee：GenerateLoop 的 prefill/decode、测试的 last_logits / forward_token_host
    // 都进这里。 例：token_ids=[3,17,42]、pos_offset=0 -> 一块 forward，盖住 embed + N 层 +
    // lm_head。
    NVTX_RANGE("forward");

    cudaStream_t stream = engine->stream;
    int *d_pos = inference_engine_d_pos_get(engine, num_tokens, pos_offset, stream);

    InferenceForwardCtx ctx{};
    ctx.num_tokens = num_tokens;
    ctx.stream = stream;
    ctx.d_token_ids = d_token_ids;
    ctx.d_hidden_out = engine->pool.d_hidden_out;
    ctx.d_pos = d_pos;
    ctx.d_logits = d_logits;

    return inference_engine_forward_device(engine, &ctx);
}

// host token -> pool H2D -> forward_token_device。末列留在 GPU 给 sampler。
// 例：prefill [3,17,42] T=3；随后 decode T=1 三次，同一块 pool.d_token_ids / d_logits。
// T=1 layer workspace 已在 create 预热，decode 不再 malloc。
extern "C" int inference_engine_forward_token_last_logits(InferenceEngine *engine,
                                                          const int *token_ids_host, int num_tokens,
                                                          int pos_offset) {
    if (engine == nullptr || token_ids_host == nullptr || num_tokens <= 0) {
        return -1;
    }
    if (num_tokens > engine->pool.max_seq_len || engine->pool.d_token_ids == nullptr ||
        engine->pool.d_logits == nullptr) {
        return -1;
    }

    cudaStream_t stream = engine->stream;
    CUDA_CHECK(cudaMemcpyAsync(engine->pool.d_token_ids, token_ids_host,
                               static_cast<size_t>(num_tokens) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    const int rc = inference_engine_forward_token_device(
        engine, engine->pool.d_token_ids, engine->pool.d_logits, num_tokens, pos_offset);
    if (rc == 0) {
        engine->pool.last_num_tokens = num_tokens;
    }
    return rc;
}

extern "C" const float *inference_engine_d_logits_last(const InferenceEngine *engine) {
    if (engine == nullptr || engine->pool.d_logits == nullptr ||
        engine->pool.last_num_tokens <= 0) {
        return nullptr;
    }
    return engine->pool.d_logits + static_cast<size_t>(engine->pool.vocab_size) *
                                       static_cast<size_t>(engine->pool.last_num_tokens - 1);
}

extern "C" int *inference_engine_d_out_token(InferenceEngine *engine) {
    if (engine == nullptr) {
        return nullptr;
    }
    return engine->pool.d_out_token;
}

// 测试：H2D token_ids -> last_logits -> D2H logits [vocab, T] col-major
extern "C" int inference_engine_forward_token_host(InferenceEngine *engine,
                                                   const int *token_ids_host,
                                                   float *logits_out_host, int num_tokens,
                                                   int pos_offset) {
    if (engine == nullptr || token_ids_host == nullptr || logits_out_host == nullptr ||
        num_tokens <= 0) {
        return -1;
    }

    const int rc =
        inference_engine_forward_token_last_logits(engine, token_ids_host, num_tokens, pos_offset);
    if (rc != 0) {
        return rc;
    }

    cudaStream_t stream = engine->stream;
    const size_t logits_bytes = static_cast<size_t>(engine->pool.vocab_size) *
                                static_cast<size_t>(num_tokens) * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(logits_out_host, engine->pool.d_logits, logits_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}
