#include "model_config.h"
#include "weight_loader.h"

#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(weight_loader_me, m) {
    m.doc() = "WeightLoader: model file to host tensors (no GPU session)";

    m.def(
        "validate_config",
        [](int hidden_size, int intermediate_size, int num_layers, int num_q_heads,
           int num_kv_heads, int head_dim, int vocab_size, int max_seq_len, float freq_base,
           float rms_norm_epsilon, int tie_word_embeddings) {
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
            if (model_config_validate(&cfg) != 0) {
                throw std::invalid_argument("invalid ModelConfig");
            }
            return true;
        },
        py::arg("hidden_size"), py::arg("intermediate_size"), py::arg("num_layers"),
        py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("head_dim"), py::arg("vocab_size"),
        py::arg("max_seq_len"), py::arg("freq_base") = 10000.f, py::arg("rms_norm_epsilon") = 1e-5f,
        py::arg("tie_word_embeddings") = 0);

    m.def(
        "load_fixture",
        [](const std::string &path) {
            WeightLoadResult result{};
            weight_load_result_init(&result);
            if (weight_loader_load_fixture(path.c_str(), &result) != 0) {
                throw std::runtime_error("weight_loader_load_fixture not implemented");
            }
            return result.num_tensors;
        },
        py::arg("path"));

    m.def(
        "load_safetensors",
        [](const std::string &path) {
            WeightLoadResult result{};
            weight_load_result_init(&result);
            if (weight_loader_load_safetensors(path.c_str(), &result) != 0) {
                throw std::runtime_error("weight_loader_load_safetensors not implemented");
            }
            return result.num_tensors;
        },
        py::arg("path"));
}
