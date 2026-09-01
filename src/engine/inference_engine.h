#pragma once

#include <stddef.h>
#include <stdint.h>

struct KVCache;
typedef struct TransformerModel TransformerModel;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InferenceEngine InferenceEngine;

// 每步 forward 入参；栈上构造，不持久化。
// d_token_ids 与 d_hidden_in 二选一：有 token_ids 走 embed，否则用 hidden_in（测试路径）。
typedef struct InferenceForwardCtx {
    int num_tokens;
    void *stream;
    const int *d_token_ids;
    const float *d_hidden_in;
    float *d_hidden_out;
    const int *d_pos;
    float *d_logits;
} InferenceForwardCtx;

InferenceEngine *inference_engine_create(TransformerModel *model, void *stream_in);

void inference_engine_destroy(InferenceEngine *engine);

void inference_engine_reset(InferenceEngine *engine);

const TransformerModel *inference_engine_get_model(const InferenceEngine *engine);

KVCache *inference_engine_get_kv_cache(InferenceEngine *engine);

int inference_engine_kv_cache_len(const InferenceEngine *engine);

int inference_engine_next_pos(const InferenceEngine *engine);

void *inference_engine_get_stream(InferenceEngine *engine);

// embed（可选）-> N x layer -> advance_len(T) -> final_norm -> lm_head（可选 d_logits）
int inference_engine_forward_device(InferenceEngine *engine, const InferenceForwardCtx *ctx);

// 编排/production：d_token_ids、d_logits 已在 GPU；d_hidden_out 用 session pool
int inference_engine_forward_token_device(InferenceEngine *engine, const int *d_token_ids,
                                          float *d_logits, int num_tokens, int pos_offset);

// 生产：last_logits + 末列采样 + D2H 一个 token id。Python 不暴露。
int inference_engine_forward_token_sample(InferenceEngine *engine, const int *token_ids_host,
                                          int num_tokens, int pos_offset, int top_k,
                                          float temperature, float top_p, uint64_t seed,
                                          int *out_token_host);

const float *inference_engine_d_logits_last(const InferenceEngine *engine);

int *inference_engine_d_out_token(InferenceEngine *engine);

// 测试：pos_offset 生成本步 d_pos；H2D hidden -> forward_device -> D2H hidden_out
// _hidden_ 表示：这一步从 hidden 状态进，不走 embed
int inference_engine_forward_hidden_host(InferenceEngine *engine, const float *hidden_in_host,
                                         float *hidden_out_host, int num_tokens, int pos_offset);

// 测试：H2D token_ids -> embed + forward + lm_head -> D2H logits [vocab, T] col-major
int inference_engine_forward_token_host(InferenceEngine *engine, const int *token_ids_host,
                                        float *logits_out_host, int num_tokens, int pos_offset);

#ifdef __cplusplus
}
#endif
