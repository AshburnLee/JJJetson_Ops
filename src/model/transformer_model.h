#pragma once

#include "model_config.h"
#include "rope_cossin_cache.h"
#include "weight_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

// TransformerLayerWeights 需要放在 header，因为 Engine 要直接拿 d_w_* 传给 layer 链，
// 它是对外可见的数据视图，不是内部状态
typedef struct TransformerLayerWeights {
    float *d_w_q;
    float *d_w_k;
    float *d_w_v;
    float *d_w_o;
    float *d_w_gate;
    float *d_w_up;
    float *d_w_down;
    float *d_w_input_layernorm;
    float *d_w_post_attention_layernorm;
} TransformerLayerWeights;

typedef struct TransformerModel TransformerModel;

// 骨架：按 ModelConfig 分配 GPU 权重容器与 RopeCosSinCache。
TransformerModel *transformer_model_create(const ModelConfig *cfg);

// Loader 产出 host 权重后 H2D 拷贝至 device buffer；仅可调用一次（immutable）。0 成功，-1 失败。
// model_destroy 须在引用该 Model 的全部 Engine destroy 之后（由调用方保证）。
int transformer_model_load_weights(TransformerModel *model, const WeightLoadResult *loaded);

int transformer_model_is_weights_loaded(const TransformerModel *model);

void transformer_model_destroy(TransformerModel *model);

const ModelConfig *transformer_model_get_config(const TransformerModel *model);

int transformer_model_get_num_layers(const TransformerModel *model);

RopeCosSinCache *transformer_model_get_rope_cache(TransformerModel *model);

const TransformerLayerWeights *transformer_model_get_layer_weights(const TransformerModel *model,
                                                                   int layer_idx);

const float *transformer_model_get_d_embed(const TransformerModel *model);

const float *transformer_model_get_d_lm_head(const TransformerModel *model);

const float *transformer_model_get_d_final_norm(const TransformerModel *model);

int transformer_model_is_tied_embeddings(const TransformerModel *model);

// embed / lm_head 的乘加、查表在 src/cuda/embed.cu、lm_head.cu 里做，和 linear、rms_norm
// 同级，是通用算子。 下面两个 transformer_model_*_forward 只是薄壳，别把它当成算子本体：帮你从
// model 取出 d_embed、 d_lm_head，先看 weights 有没有 load；lm_head 还要再看
// tie_word_embeddings——为 1 就走 tied 路径 （权重用 d_embed），为 0 就走 untied 路径（用单独的
// d_lm_head）。细节见 doc/design/phase2_lifecycle.md §2.1.1、§2.4.1。
//
// Engine 理论上也能跳过这层，直接 embed_forward_device(stream, get_d_embed(model), ...)，
// 但 tied 时函数和指针都要自己配对，load 状态也要自己查，一般还是调这里省心。
// 和 layer 链的差别：Phase 1 要求 Engine 把每层 9 个 d_w_* 显式传给 layer 链，所以 Model 用
// get_layer_weights 暴露指针；embed/lm_head 没有这套老接口，权重又挂在 Model 上，就在 Model 包一层
// forward。

// embed: d_token_ids -> d_hidden（col-major [hidden, T]）。需 weights_loaded。
int transformer_model_embed_forward_device(void *stream, const TransformerModel *model,
                                           const int *d_token_ids, float *d_hidden, int num_tokens);

// lm_head: d_hidden -> d_logits（col-major [vocab, T]）；tied 时用 embed^T GEMM。需
// weights_loaded。
int transformer_model_lm_head_forward_device(void *stream, void *cublas_handle,
                                             const TransformerModel *model, const float *d_hidden,
                                             float *d_logits, int num_tokens);

// 测试薄封装：H2D I/O + 上述 device 路径（权重已在 Model GPU 上）
int transformer_model_embed_forward_host(const TransformerModel *model, const int *token_ids_host,
                                         float *hidden_host, int num_tokens);

int transformer_model_lm_head_forward_host(const TransformerModel *model, const float *hidden_host,
                                           float *logits_host, int num_tokens);

// final norm：N 层 block 跑完后的 RMSNorm，权重 d_w_final_norm，epsilon 来自 ModelConfig。
// 薄壳内部调 rms_norm_forward_device；d_hidden_out 可与 d_hidden_in 相同（in-place）。
int transformer_model_final_norm_forward_device(void *stream, const TransformerModel *model,
                                                const float *d_hidden_in, float *d_hidden_out,
                                                int num_tokens);

int transformer_model_final_norm_forward_host(const TransformerModel *model,
                                              const float *hidden_in_host, float *hidden_out_host,
                                              int num_tokens);

#ifdef __cplusplus
}
#endif
