#include "model_config.h"
#include "weight_loader.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

static py::dict model_config_to_py(const ModelConfig &cfg) {
    py::dict out;
    out["hidden_size"] = cfg.hidden_size;
    out["intermediate_size"] = cfg.intermediate_size;
    out["num_layers"] = cfg.num_layers;
    out["num_q_heads"] = cfg.num_q_heads;
    out["num_kv_heads"] = cfg.num_kv_heads;
    out["head_dim"] = cfg.head_dim;
    out["vocab_size"] = cfg.vocab_size;
    out["max_seq_len"] = cfg.max_seq_len;
    out["freq_base"] = cfg.freq_base;
    out["rms_norm_epsilon"] = cfg.rms_norm_epsilon;
    out["tie_word_embeddings"] = cfg.tie_word_embeddings;
    return out;
}

static py::dict weight_load_result_to_py(const WeightLoadResult &result) {
    py::dict tensors;
    for (int i = 0; i < result.num_tensors; ++i) {
        const HostTensor &tensor = result.tensors[i];
        if (tensor.name == nullptr || tensor.data == nullptr || tensor.dims == nullptr) {
            throw std::runtime_error("invalid tensor entry in WeightLoadResult");
        }
        std::vector<pybind11::ssize_t> shape(static_cast<size_t>(tensor.ndim));
        int64_t numel = 1;
        for (int d = 0; d < tensor.ndim; ++d) {
            shape[static_cast<size_t>(d)] = static_cast<pybind11::ssize_t>(tensor.dims[d]);
            numel *= tensor.dims[d];
        }
        py::array_t<float> arr(shape);
        std::memcpy(arr.mutable_data(), tensor.data, static_cast<size_t>(numel) * sizeof(float));
        tensors[tensor.name] = arr;
    }

    py::dict out;
    out["config"] = model_config_to_py(result.config);
    out["tensors"] = tensors;
    out["num_tensors"] = result.num_tensors;
    return out;
}

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
                throw std::runtime_error("weight_loader_load_fixture failed");
            }
            py::dict py_result = weight_load_result_to_py(result);
            weight_load_result_destroy(&result);
            return py_result;
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
            py::dict py_result = weight_load_result_to_py(result);
            weight_load_result_destroy(&result);
            return py_result;
        },
        py::arg("path"));
}
