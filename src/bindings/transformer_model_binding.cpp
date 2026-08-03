#include "transformer_model.h"
#include "weight_loader.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cublas_utils.cuh>
#include <cuda_runtime.h>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(transformer_model_me, m) {
    m.doc() = "TransformerModel: immutable GPU weights container";

    m.def(
        "create_model",
        [](int hidden_size, int intermediate_size, int num_layers, int num_q_heads,
           int num_kv_heads, int head_dim, int vocab_size, int max_seq_len, float freq_base,
           float rms_norm_epsilon, int tie_word_embeddings) -> uintptr_t {
            ModelConfig cfg{};
            cfg.hidden_size = hidden_size;
            cfg.intermediate_size = intermediate_size;
            cfg.num_layers = num_layers;
            cfg.num_q_heads = num_q_heads;
            cfg.num_kv_heads = num_kv_heads;
            cfg.head_dim = head_dim;
            cfg.vocab_size = vocab_size;
            cfg.max_seq_len = max_seq_len;
            cfg.freq_base = freq_base;
            cfg.rms_norm_epsilon = rms_norm_epsilon;
            cfg.tie_word_embeddings = tie_word_embeddings;

            TransformerModel *model = transformer_model_create(&cfg);
            if (model == nullptr) {
                throw std::runtime_error("transformer_model_create failed");
            }
            return reinterpret_cast<uintptr_t>(model);
        },
        py::arg("hidden_size"), py::arg("intermediate_size"), py::arg("num_layers"),
        py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("head_dim"), py::arg("vocab_size"),
        py::arg("max_seq_len"), py::arg("freq_base") = 10000.f, py::arg("rms_norm_epsilon") = 1e-5f,
        py::arg("tie_word_embeddings") = 0);

    m.def(
        "destroy_model",
        [](uintptr_t model_handle) {
            transformer_model_destroy(reinterpret_cast<TransformerModel *>(model_handle));
        },
        py::arg("model_handle"));

    m.def(
        "get_num_layers",
        [](uintptr_t model_handle) {
            const TransformerModel *model =
                reinterpret_cast<const TransformerModel *>(model_handle);
            return transformer_model_get_num_layers(model);
        },
        py::arg("model_handle"));

    m.def(
        "is_tied_embeddings",
        [](uintptr_t model_handle) {
            const TransformerModel *model =
                reinterpret_cast<const TransformerModel *>(model_handle);
            return transformer_model_is_tied_embeddings(model) == 1;
        },
        py::arg("model_handle"));

    m.def(
        "is_weights_loaded",
        [](uintptr_t model_handle) {
            const TransformerModel *model =
                reinterpret_cast<const TransformerModel *>(model_handle);
            return transformer_model_is_weights_loaded(model) == 1;
        },
        py::arg("model_handle"));

    m.def(
        "load_weights_from_fixture",
        [](uintptr_t model_handle, const std::string &fixture_path) {
            TransformerModel *model = reinterpret_cast<TransformerModel *>(model_handle);
            WeightLoadResult loaded{};
            weight_load_result_init(&loaded);
            if (weight_loader_load_fixture(fixture_path.c_str(), &loaded) != 0) {
                throw std::runtime_error("weight_loader_load_fixture failed");
            }
            if (transformer_model_load_weights(model, &loaded) != 0) {
                weight_load_result_destroy(&loaded);
                throw std::runtime_error("transformer_model_load_weights failed");
            }
            weight_load_result_destroy(&loaded);
        },
        py::arg("model_handle"), py::arg("fixture_path"));

    m.def(
        "read_layer_w_q_host",
        [](uintptr_t model_handle, int layer_idx, int hidden_size, int q_dim) {
            const TransformerModel *model =
                reinterpret_cast<const TransformerModel *>(model_handle);
            if (transformer_model_is_weights_loaded(model) != 1) {
                throw std::runtime_error("weights not loaded");
            }
            const TransformerLayerWeights *layer =
                transformer_model_get_layer_weights(model, layer_idx);
            if (layer == nullptr || layer->d_w_q == nullptr) {
                throw std::runtime_error("invalid layer or d_w_q");
            }
            py::array_t<float> out({hidden_size, q_dim});
            const size_t bytes = static_cast<size_t>(hidden_size) * q_dim * sizeof(float);
            cudaError_t err =
                cudaMemcpy(out.mutable_data(), layer->d_w_q, bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(err));
            }
            return out;
        },
        py::arg("model_handle"), py::arg("layer_idx"), py::arg("hidden_size"), py::arg("q_dim"));

    m.def(
        "embed_forward_host",
        [](uintptr_t model_handle,
           py::array_t<int, py::array::c_style | py::array::forcecast> token_ids, int num_tokens) {
            const TransformerModel *model =
                reinterpret_cast<const TransformerModel *>(model_handle);
            const ModelConfig *cfg = transformer_model_get_config(model);
            if (cfg == nullptr) {
                throw std::runtime_error("invalid model");
            }
            if (static_cast<int>(token_ids.size()) != num_tokens) {
                throw std::runtime_error("token_ids size mismatch");
            }
            py::array_t<float> hidden(
                {cfg->hidden_size, num_tokens},
                {static_cast<ssize_t>(sizeof(float)),
                 static_cast<ssize_t>(cfg->hidden_size) * static_cast<ssize_t>(sizeof(float))});
            if (transformer_model_embed_forward_host(model, token_ids.data(), hidden.mutable_data(),
                                                     num_tokens) != 0) {
                throw std::runtime_error("transformer_model_embed_forward_host failed");
            }
            return hidden;
        },
        py::arg("model_handle"), py::arg("token_ids"), py::arg("num_tokens"));

    m.def(
        "lm_head_forward_host",
        [](uintptr_t model_handle,
           py::array_t<float, py::array::f_style | py::array::forcecast> hidden, int num_tokens) {
            const TransformerModel *model =
                reinterpret_cast<const TransformerModel *>(model_handle);
            const ModelConfig *cfg = transformer_model_get_config(model);
            if (cfg == nullptr) {
                throw std::runtime_error("invalid model");
            }
            auto hidden_buf = hidden.request();
            if (hidden_buf.ndim != 2 || static_cast<int>(hidden_buf.shape[1]) != num_tokens) {
                throw std::runtime_error("hidden must be [hidden_size, num_tokens] Fortran order");
            }
            py::array_t<float> logits(
                {cfg->vocab_size, num_tokens},
                {static_cast<ssize_t>(sizeof(float)),
                 static_cast<ssize_t>(cfg->vocab_size) * static_cast<ssize_t>(sizeof(float))});
            if (transformer_model_lm_head_forward_host(model, hidden.data(), logits.mutable_data(),
                                                       num_tokens) != 0) {
                throw std::runtime_error("transformer_model_lm_head_forward_host failed");
            }
            return logits;
        },
        py::arg("model_handle"), py::arg("hidden"), py::arg("num_tokens"));
}
