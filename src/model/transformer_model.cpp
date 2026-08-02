#include "transformer_model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_utils.h"

// 按 ModelConfig 在 device 上为 embed、各层 Linear/RMSNorm、final norm、lm_head
// 等分配并持有显存地址 是只读的 GPU 权重容器
struct TransformerModel {
    ModelConfig config{};
    RopeCosSinCache *rope_cache = nullptr;
    TransformerLayerWeights *layers = nullptr;
    float *d_embed = nullptr;
    float *d_lm_head = nullptr;
    float *d_w_final_norm = nullptr;
    bool lm_head_tied = false;
};

static int compute_layer_weight_bytes(const ModelConfig *cfg, size_t *out_layer_bytes) {
    if (cfg == nullptr || out_layer_bytes == nullptr) {
        return -1;
    }
    const int q_dim = cfg->num_q_heads * cfg->head_dim;
    const int kv_dim = cfg->num_kv_heads * cfg->head_dim;
    const size_t w_q = static_cast<size_t>(q_dim) * cfg->hidden_size * sizeof(float);
    const size_t w_kv = static_cast<size_t>(kv_dim) * cfg->hidden_size * sizeof(float);
    const size_t w_o = static_cast<size_t>(cfg->hidden_size) * q_dim * sizeof(float);
    const size_t w_gu =
        static_cast<size_t>(cfg->intermediate_size) * cfg->hidden_size * sizeof(float);
    const size_t w_d =
        static_cast<size_t>(cfg->hidden_size) * cfg->intermediate_size * sizeof(float);
    const size_t w_norm = static_cast<size_t>(cfg->hidden_size) * sizeof(float);
    out_layer_bytes[0] = w_q;
    out_layer_bytes[1] = w_kv;
    out_layer_bytes[2] = w_kv;
    out_layer_bytes[3] = w_o;
    out_layer_bytes[4] = w_gu;
    out_layer_bytes[5] = w_gu;
    out_layer_bytes[6] = w_d;
    out_layer_bytes[7] = w_norm;
    out_layer_bytes[8] = w_norm;
    return 0;
}

static void free_layer_weights(TransformerLayerWeights *layer) {
    if (layer == nullptr) {
        return;
    }
    cudaFree(layer->d_w_q);
    cudaFree(layer->d_w_k);
    cudaFree(layer->d_w_v);
    cudaFree(layer->d_w_o);
    cudaFree(layer->d_w_gate);
    cudaFree(layer->d_w_up);
    cudaFree(layer->d_w_down);
    cudaFree(layer->d_w_input_layernorm);
    cudaFree(layer->d_w_post_attention_layernorm);
    std::memset(layer, 0, sizeof(*layer));
}

static int allocate_layer_weights(const ModelConfig *cfg, TransformerLayerWeights *layer) {
    size_t bytes[9]{};
    if (compute_layer_weight_bytes(cfg, bytes) != 0) {
        return -1;
    }

    CUDA_CHECK(cudaMalloc(&layer->d_w_q, bytes[0]));
    CUDA_CHECK(cudaMalloc(&layer->d_w_k, bytes[1]));
    CUDA_CHECK(cudaMalloc(&layer->d_w_v, bytes[2]));
    CUDA_CHECK(cudaMalloc(&layer->d_w_o, bytes[3]));
    CUDA_CHECK(cudaMalloc(&layer->d_w_gate, bytes[4]));
    CUDA_CHECK(cudaMalloc(&layer->d_w_up, bytes[5]));
    CUDA_CHECK(cudaMalloc(&layer->d_w_down, bytes[6]));
    CUDA_CHECK(cudaMalloc(&layer->d_w_input_layernorm, bytes[7]));
    CUDA_CHECK(cudaMalloc(&layer->d_w_post_attention_layernorm, bytes[8]));
    return 0;
}

TransformerModel *transformer_model_create(const ModelConfig *cfg) {
    if (cfg == nullptr || model_config_validate(cfg) != 0) {
        std::fprintf(stderr, "transformer_model_create: invalid ModelConfig\n");
        return nullptr;
    }
    //
    auto *model = new TransformerModel{};
    model->config = *cfg;
    model->lm_head_tied = (cfg->tie_word_embeddings != 0);

    model->rope_cache = rope_cossin_cache_create(cfg->max_seq_len, cfg->head_dim, cfg->freq_base);
    if (model->rope_cache == nullptr) {
        transformer_model_destroy(model);
        return nullptr;
    }

    model->layers = static_cast<TransformerLayerWeights *>(
        std::calloc(static_cast<size_t>(cfg->num_layers), sizeof(TransformerLayerWeights)));
    if (model->layers == nullptr) {
        transformer_model_destroy(model);
        return nullptr;
    }

    for (int i = 0; i < cfg->num_layers; ++i) {
        if (allocate_layer_weights(cfg, &model->layers[i]) != 0) {
            transformer_model_destroy(model);
            return nullptr;
        }
    }

    const size_t embed_bytes =
        static_cast<size_t>(cfg->vocab_size) * cfg->hidden_size * sizeof(float);
    const size_t final_norm_bytes = static_cast<size_t>(cfg->hidden_size) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&model->d_embed, embed_bytes));
    CUDA_CHECK(cudaMalloc(&model->d_w_final_norm, final_norm_bytes));

    if (model->lm_head_tied) {
        model->d_lm_head = model->d_embed;
    } else {
        CUDA_CHECK(cudaMalloc(&model->d_lm_head, embed_bytes));
    }

    return model;
}

void transformer_model_destroy(TransformerModel *model) {
    if (model == nullptr) {
        return;
    }

    if (model->layers != nullptr) {
        for (int i = 0; i < model->config.num_layers; ++i) {
            free_layer_weights(&model->layers[i]);
        }
        std::free(model->layers);
        model->layers = nullptr;
    }

    rope_cossin_cache_destroy(model->rope_cache);
    model->rope_cache = nullptr;

    cudaFree(model->d_embed);
    model->d_embed = nullptr;

    if (!model->lm_head_tied) {
        cudaFree(model->d_lm_head);
    }
    model->d_lm_head = nullptr;

    cudaFree(model->d_w_final_norm);
    model->d_w_final_norm = nullptr;

    delete model;
}

const ModelConfig *transformer_model_get_config(const TransformerModel *model) {
    if (model == nullptr) {
        return nullptr;
    }
    return &model->config;
}

int transformer_model_get_num_layers(const TransformerModel *model) {
    if (model == nullptr) {
        return -1;
    }
    return model->config.num_layers;
}

RopeCosSinCache *transformer_model_get_rope_cache(TransformerModel *model) {
    if (model == nullptr) {
        return nullptr;
    }
    return model->rope_cache;
}

const TransformerLayerWeights *transformer_model_get_layer_weights(const TransformerModel *model,
                                                                   int layer_idx) {
    if (model == nullptr || model->layers == nullptr || layer_idx < 0 ||
        layer_idx >= model->config.num_layers) {
        return nullptr;
    }
    return &model->layers[layer_idx];
}

const float *transformer_model_get_d_embed(const TransformerModel *model) {
    if (model == nullptr) {
        return nullptr;
    }
    return model->d_embed;
}

const float *transformer_model_get_d_lm_head(const TransformerModel *model) {
    if (model == nullptr) {
        return nullptr;
    }
    return model->d_lm_head;
}

const float *transformer_model_get_d_final_norm(const TransformerModel *model) {
    if (model == nullptr) {
        return nullptr;
    }
    return model->d_w_final_norm;
}

int transformer_model_is_tied_embeddings(const TransformerModel *model) {
    if (model == nullptr) {
        return -1;
    }
    return model->lm_head_tied ? 1 : 0;
}
