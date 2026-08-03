#include "transformer_model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cublas_v2.h>
#include <cublas_utils.cuh>
#include "cuda_utils.h"
#include "embed.h"
#include "lm_head.h"
#include "weight_loader.h"

// 按 ModelConfig 在 device 上为 embed、各层 Linear/RMSNorm、final norm、lm_head
// 等分配并持有显存地址；immutable GPU 权重容器
struct TransformerModel {
    ModelConfig config{};
    RopeCosSinCache *rope_cache = nullptr;
    TransformerLayerWeights *layers = nullptr;
    float *d_embed = nullptr;
    float *d_lm_head = nullptr;
    float *d_w_final_norm = nullptr;
    bool lm_head_tied = false;
    bool weights_loaded = false;
};

static int model_config_matches(const ModelConfig *a, const ModelConfig *b) {
    if (a == nullptr || b == nullptr) {
        return 0;
    }
    if (a->hidden_size != b->hidden_size || a->intermediate_size != b->intermediate_size ||
        a->num_layers != b->num_layers || a->num_q_heads != b->num_q_heads ||
        a->num_kv_heads != b->num_kv_heads || a->head_dim != b->head_dim ||
        a->vocab_size != b->vocab_size || a->max_seq_len != b->max_seq_len ||
        a->tie_word_embeddings != b->tie_word_embeddings) {
        return 0;
    }
    if (a->freq_base != b->freq_base) {
        return 0;
    }
    const float eps_diff = a->rms_norm_epsilon - b->rms_norm_epsilon;
    if (eps_diff < 0.f) {
        if (-eps_diff > 1e-6f) {
            return 0;
        }
    } else if (eps_diff > 1e-6f) {
        return 0;
    }
    return 1;
}

static size_t host_tensor_num_bytes(const HostTensor *tensor) {
    if (tensor == nullptr || tensor->data == nullptr || tensor->dims == nullptr ||
        tensor->ndim <= 0) {
        return 0;
    }
    int64_t numel = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
        numel *= tensor->dims[i];
    }
    return static_cast<size_t>(numel) * sizeof(float);
}

static int copy_host_tensor_to_device(float *d_dst, size_t dst_bytes, const HostTensor *host_tensor,
                                      const char *name_for_error) {
    if (d_dst == nullptr || host_tensor == nullptr) {
        return -1;
    }
    const size_t src_bytes = host_tensor_num_bytes(host_tensor);
    if (src_bytes == 0 || src_bytes != dst_bytes) {
        std::fprintf(stderr, "transformer_model_load_weights: byte mismatch for %s\n",
                     name_for_error != nullptr ? name_for_error : "tensor");
        return -1;
    }
    CUDA_CHECK(cudaMemcpy(d_dst, host_tensor->data, dst_bytes, cudaMemcpyHostToDevice));
    return 0;
}

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

static int load_layer_weight_from_result(const WeightLoadResult *loaded, int layer_idx,
                                         const char *suffix, float *d_dst, size_t dst_bytes) {
    char name[128];
    std::snprintf(name, sizeof(name), "layer%d.%s", layer_idx, suffix);
    const HostTensor *tensor = weight_load_result_find(loaded, name);
    if (tensor == nullptr) {
        std::fprintf(stderr, "transformer_model_load_weights: missing tensor %s\n", name);
        return -1;
    }
    return copy_host_tensor_to_device(d_dst, dst_bytes, tensor, name);
}

static int load_one_layer_from_result(TransformerModel *model, const WeightLoadResult *loaded,
                                      int layer_idx) {
    size_t bytes[9]{};
    if (compute_layer_weight_bytes(&model->config, bytes) != 0) {
        return -1;
    }
    TransformerLayerWeights *layer = &model->layers[layer_idx];
    static const char *k_suffixes[9] = {"w_q",
                                        "w_k",
                                        "w_v",
                                        "w_o",
                                        "w_gate",
                                        "w_up",
                                        "w_down",
                                        "w_input_layernorm",
                                        "w_post_attention_layernorm"};
    float *d_ptrs[9] = {layer->d_w_q,
                        layer->d_w_k,
                        layer->d_w_v,
                        layer->d_w_o,
                        layer->d_w_gate,
                        layer->d_w_up,
                        layer->d_w_down,
                        layer->d_w_input_layernorm,
                        layer->d_w_post_attention_layernorm};
    for (int i = 0; i < 9; ++i) {
        if (load_layer_weight_from_result(loaded, layer_idx, k_suffixes[i], d_ptrs[i], bytes[i]) !=
            0) {
            return -1;
        }
    }
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

int transformer_model_is_weights_loaded(const TransformerModel *model) {
    if (model == nullptr) {
        return -1;
    }
    return model->weights_loaded ? 1 : 0;
}

int transformer_model_load_weights(TransformerModel *model, const WeightLoadResult *loaded) {
    if (model == nullptr || loaded == nullptr) {
        return -1;
    }
    if (model->weights_loaded) {
        std::fprintf(stderr,
                     "transformer_model_load_weights: weights already loaded (immutable)\n");
        return -1;
    }
    if (!model_config_matches(&model->config, &loaded->config)) {
        std::fprintf(stderr, "transformer_model_load_weights: ModelConfig mismatch\n");
        return -1;
    }

    for (int layer = 0; layer < model->config.num_layers; ++layer) {
        if (load_one_layer_from_result(model, loaded, layer) != 0) {
            return -1;
        }
    }

    const size_t embed_bytes =
        static_cast<size_t>(model->config.vocab_size) * model->config.hidden_size * sizeof(float);
    const size_t final_norm_bytes = static_cast<size_t>(model->config.hidden_size) * sizeof(float);

    const HostTensor *embed = weight_load_result_find(loaded, "embed");
    if (embed == nullptr ||
        copy_host_tensor_to_device(model->d_embed, embed_bytes, embed, "embed") != 0) {
        return -1;
    }

    const HostTensor *final_norm = weight_load_result_find(loaded, "final_norm");
    if (final_norm == nullptr || copy_host_tensor_to_device(model->d_w_final_norm, final_norm_bytes,
                                                            final_norm, "final_norm") != 0) {
        return -1;
    }

    if (!model->lm_head_tied) {
        const HostTensor *lm_head = weight_load_result_find(loaded, "lm_head");
        if (lm_head == nullptr ||
            copy_host_tensor_to_device(model->d_lm_head, embed_bytes, lm_head, "lm_head") != 0) {
            return -1;
        }
    }

    model->weights_loaded = true;
    return 0;
}

static int check_weights_loaded_for_forward(const TransformerModel *model, const char *op) {
    if (model == nullptr) {
        return -1;
    }
    if (!model->weights_loaded) {
        std::fprintf(stderr, "%s: weights not loaded\n", op != nullptr ? op : "transformer_model");
        return -1;
    }
    return 0;
}

int transformer_model_embed_forward_device(void *stream, const TransformerModel *model,
                                           const int *d_token_ids, float *d_hidden,
                                           int num_tokens) {
    if (check_weights_loaded_for_forward(model, "transformer_model_embed_forward_device") != 0) {
        return -1;
    }
    return embed_forward_device(stream, model->d_embed, d_token_ids, d_hidden,
                                model->config.hidden_size, num_tokens);
}

int transformer_model_lm_head_forward_device(void *stream, void *cublas_handle,
                                             const TransformerModel *model, const float *d_hidden,
                                             float *d_logits, int num_tokens) {
    if (check_weights_loaded_for_forward(model, "transformer_model_lm_head_forward_device") != 0) {
        return -1;
    }
    const int hidden_size = model->config.hidden_size;
    const int vocab_size = model->config.vocab_size;
    if (model->lm_head_tied) {
        tied_lm_head_forward_device(stream, cublas_handle, model->d_embed, d_hidden, d_logits,
                                    hidden_size, vocab_size, num_tokens);
        return 0;
    }
    untied_lm_head_forward_device(stream, cublas_handle, model->d_lm_head, d_hidden, d_logits,
                                  hidden_size, vocab_size, num_tokens);
    return 0;
}

// ======================  test only ========================
int transformer_model_embed_forward_host(const TransformerModel *model, const int *token_ids_host,
                                         float *hidden_host, int num_tokens) {
    if (check_weights_loaded_for_forward(model, "transformer_model_embed_forward_host") != 0) {
        return -1;
    }
    if (token_ids_host == nullptr || hidden_host == nullptr || num_tokens <= 0) {
        return -1;
    }

    const int hidden_size = model->config.hidden_size;
    const int vocab_size = model->config.vocab_size;
    const int64_t n_hidden = static_cast<int64_t>(hidden_size) * num_tokens;

    int *d_token_ids = nullptr;
    float *d_hidden = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(
        cudaMallocAsync(&d_token_ids, static_cast<size_t>(num_tokens) * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_hidden, static_cast<size_t>(n_hidden) * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_token_ids, token_ids_host,
                               static_cast<size_t>(num_tokens) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    if (transformer_model_embed_forward_device(stream, model, d_token_ids, d_hidden, num_tokens) !=
        0) {
        cudaFreeAsync(d_token_ids, stream);
        cudaFreeAsync(d_hidden, stream);
        cudaStreamDestroy(stream);
        return -1;
    }

    CUDA_CHECK(cudaMemcpyAsync(hidden_host, d_hidden, static_cast<size_t>(n_hidden) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_token_ids, stream));
    CUDA_CHECK(cudaFreeAsync(d_hidden, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}

// ======================  test only ========================
int transformer_model_lm_head_forward_host(const TransformerModel *model, const float *hidden_host,
                                           float *logits_host, int num_tokens) {
    if (check_weights_loaded_for_forward(model, "transformer_model_lm_head_forward_host") != 0) {
        return -1;
    }
    if (hidden_host == nullptr || logits_host == nullptr || num_tokens <= 0) {
        return -1;
    }

    const int hidden_size = model->config.hidden_size;
    const int vocab_size = model->config.vocab_size;
    const int64_t n_hidden = static_cast<int64_t>(hidden_size) * num_tokens;
    const int64_t n_logits = static_cast<int64_t>(vocab_size) * num_tokens;

    float *d_hidden = nullptr;
    float *d_logits = nullptr;
    cudaStream_t stream;
    cublasHandle_t handle;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaMallocAsync(&d_hidden, static_cast<size_t>(n_hidden) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_logits, static_cast<size_t>(n_logits) * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_hidden, hidden_host, static_cast<size_t>(n_hidden) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    if (transformer_model_lm_head_forward_device(stream, handle, model, d_hidden, d_logits,
                                                 num_tokens) != 0) {
        cudaFreeAsync(d_hidden, stream);
        cudaFreeAsync(d_logits, stream);
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        return -1;
    }

    CUDA_CHECK(cudaMemcpyAsync(logits_host, d_logits, static_cast<size_t>(n_logits) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_hidden, stream));
    CUDA_CHECK(cudaFreeAsync(d_logits, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
