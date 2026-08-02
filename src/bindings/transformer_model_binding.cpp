#include "transformer_model.h"

#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(transformer_model_me, m) {
    m.doc() = "TransformerModel: immutable GPU weights container (skeleton)";

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
}
