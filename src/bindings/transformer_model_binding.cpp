#include "transformer_model.h"
#include "weight_loader.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cublas_utils.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

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

    // load_safetensors_hf_llama（改名 + 2D 转置 + 读 config）再 H2D；与 fixture 入口共用
    // transformer_model_load_weights。path 是 .safetensors 文件，不是目录。
    m.def(
        "load_weights_from_safetensors_hf_llama",
        [](uintptr_t model_handle, const std::string &safetensors_path) {
            TransformerModel *model = reinterpret_cast<TransformerModel *>(model_handle);
            WeightLoadResult loaded{};
            weight_load_result_init(&loaded);
            if (weight_loader_load_safetensors_hf_llama(safetensors_path.c_str(), &loaded) != 0) {
                throw std::runtime_error("weight_loader_load_safetensors_hf_llama failed");
            }
            if (transformer_model_load_weights(model, &loaded) != 0) {
                weight_load_result_destroy(&loaded);
                throw std::runtime_error("transformer_model_load_weights failed");
            }
            weight_load_result_destroy(&loaded);
        },
        py::arg("model_handle"), py::arg("safetensors_path"),
        "Test wrapper: host I/O around load_safetensors_hf_llama then "
        "transformer_model_load_weights");

    auto read_device_floats = [](const float *d_ptr, const std::vector<ssize_t> &shape) {
        if (d_ptr == nullptr) {
            throw std::runtime_error("null device pointer");
        }
        int64_t numel = 1;
        for (ssize_t dim : shape) {
            numel *= dim;
        }
        py::array_t<float> out(shape);
        const size_t bytes = static_cast<size_t>(numel) * sizeof(float);
        cudaError_t err = cudaMemcpy(out.mutable_data(), d_ptr, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        return out;
    };

    auto layer_weight_device_ptr = [](const TransformerLayerWeights *layer,
                                      const std::string &suffix) -> const float * {
        if (layer == nullptr) {
            return nullptr;
        }
        if (suffix == "w_q") {
            return layer->d_w_q;
        }
        if (suffix == "w_k") {
            return layer->d_w_k;
        }
        if (suffix == "w_v") {
            return layer->d_w_v;
        }
        if (suffix == "w_o") {
            return layer->d_w_o;
        }
        if (suffix == "w_gate") {
            return layer->d_w_gate;
        }
        if (suffix == "w_up") {
            return layer->d_w_up;
        }
        if (suffix == "w_down") {
            return layer->d_w_down;
        }
        if (suffix == "w_input_layernorm") {
            return layer->d_w_input_layernorm;
        }
        if (suffix == "w_post_attention_layernorm") {
            return layer->d_w_post_attention_layernorm;
        }
        return nullptr;
    };

    m.def(
        "read_layer_w_q_host",
        [&](uintptr_t model_handle, int layer_idx, int q_dim, int hidden_size) {
            const TransformerModel *model =
                reinterpret_cast<const TransformerModel *>(model_handle);
            if (transformer_model_is_weights_loaded(model) != 1) {
                throw std::runtime_error("weights not loaded");
            }
            const TransformerLayerWeights *layer =
                transformer_model_get_layer_weights(model, layer_idx);
            return read_device_floats(layer_weight_device_ptr(layer, "w_q"), {q_dim, hidden_size});
        },
        py::arg("model_handle"), py::arg("layer_idx"), py::arg("q_dim"), py::arg("hidden_size"));

    m.def(
        "read_layer_weight_host",
        [&](uintptr_t model_handle, int layer_idx, const std::string &suffix,
            const std::vector<ssize_t> &shape) {
            const TransformerModel *model =
                reinterpret_cast<const TransformerModel *>(model_handle);
            if (transformer_model_is_weights_loaded(model) != 1) {
                throw std::runtime_error("weights not loaded");
            }
            const TransformerLayerWeights *layer =
                transformer_model_get_layer_weights(model, layer_idx);
            if (layer == nullptr) {
                throw std::runtime_error("invalid layer_idx");
            }
            const float *d_ptr = layer_weight_device_ptr(layer, suffix);
            if (d_ptr == nullptr) {
                throw std::runtime_error("unknown layer weight suffix: " + suffix);
            }
            return read_device_floats(d_ptr, shape);
        },
        py::arg("model_handle"), py::arg("layer_idx"), py::arg("suffix"), py::arg("shape"),
        "Test only: D2H one layer weight tensor (row-major shape as in fixture)");

    m.def(
        "read_global_weight_host",
        [&](uintptr_t model_handle, const std::string &name, const std::vector<ssize_t> &shape) {
            const TransformerModel *model =
                reinterpret_cast<const TransformerModel *>(model_handle);
            if (transformer_model_is_weights_loaded(model) != 1) {
                throw std::runtime_error("weights not loaded");
            }
            const float *d_ptr = nullptr;
            if (name == "embed") {
                d_ptr = transformer_model_get_d_embed(model);
            } else if (name == "lm_head") {
                d_ptr = transformer_model_get_d_lm_head(model);
            } else if (name == "final_norm") {
                d_ptr = transformer_model_get_d_final_norm(model);
            } else {
                throw std::runtime_error("unknown global weight name: " + name);
            }
            return read_device_floats(d_ptr, shape);
        },
        py::arg("model_handle"), py::arg("name"), py::arg("shape"),
        "Test only: D2H embed / lm_head / final_norm (row-major shape as in fixture)");

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

    m.def(
        "final_norm_forward_host",
        [](uintptr_t model_handle,
           py::array_t<float, py::array::f_style | py::array::forcecast> hidden_in,
           int num_tokens) {
            const TransformerModel *model =
                reinterpret_cast<const TransformerModel *>(model_handle);
            const ModelConfig *cfg = transformer_model_get_config(model);
            if (cfg == nullptr) {
                throw std::runtime_error("invalid model");
            }
            auto in_buf = hidden_in.request();
            if (in_buf.ndim != 2 || static_cast<int>(in_buf.shape[1]) != num_tokens) {
                throw std::runtime_error(
                    "hidden_in must be [hidden_size, num_tokens] Fortran order");
            }
            py::array_t<float> hidden_out(
                {cfg->hidden_size, num_tokens},
                {static_cast<ssize_t>(sizeof(float)),
                 static_cast<ssize_t>(cfg->hidden_size) * static_cast<ssize_t>(sizeof(float))});
            if (transformer_model_final_norm_forward_host(
                    model, hidden_in.data(), hidden_out.mutable_data(), num_tokens) != 0) {
                throw std::runtime_error("transformer_model_final_norm_forward_host failed");
            }
            return hidden_out;
        },
        py::arg("model_handle"), py::arg("hidden_in"), py::arg("num_tokens"),
        "Test wrapper: host I/O around transformer_model_final_norm_forward_device");
}
