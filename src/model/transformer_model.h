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

#ifdef __cplusplus
}
#endif
