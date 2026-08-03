#pragma once

#include <stddef.h>
#include <stdint.h>

struct KVCache;
typedef struct TransformerModel TransformerModel;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InferenceEngine InferenceEngine;

// 每步 forward 的入参视图；栈上构造，不持久化。forward 编排尚未实现。
typedef struct InferenceForwardCtx {
    int num_tokens;
    void *stream;
    const int *d_token_ids;
    const float *d_hidden_in;
    float *d_hidden_out;
    const int *d_pos;
    float *d_logits;
} InferenceForwardCtx;

// 借用 model 指针（不持有所有权）；分配 KVCache(N)、BufferPool、stream/cublas。
// stream_in 为 nullptr 时 Engine 自建 non-blocking stream。
InferenceEngine *inference_engine_create(TransformerModel *model, void *stream_in);

void inference_engine_destroy(InferenceEngine *engine);

// 新对话：kv_cache_reset，next_pos 归零；不释放 pool、不碰 Model。
void inference_engine_reset(InferenceEngine *engine);

const TransformerModel *inference_engine_get_model(const InferenceEngine *engine);

KVCache *inference_engine_get_kv_cache(InferenceEngine *engine);

int inference_engine_kv_cache_len(const InferenceEngine *engine);

// SessionState：下一 token 的绝对位置；reset 后为 0，forward 实现后随 prefill/decode 推进。
int inference_engine_next_pos(const InferenceEngine *engine);

#ifdef __cplusplus
}
#endif
