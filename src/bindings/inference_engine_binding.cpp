#include "inference_engine.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "kv_cache.h"

namespace py = pybind11;

PYBIND11_MODULE(inference_engine_me, m) {
    m.doc() = "InferenceEngine: multi-layer session (Phase 2 Module 3)";

    m.def(
        "create_engine",
        [](uintptr_t model_handle) -> uintptr_t {
            TransformerModel *model = reinterpret_cast<TransformerModel *>(model_handle);
            InferenceEngine *engine = inference_engine_create(model, nullptr);
            if (engine == nullptr) {
                throw std::runtime_error("inference_engine_create failed");
            }
            return reinterpret_cast<uintptr_t>(engine);
        },
        py::arg("model_handle"));

    m.def(
        "destroy_engine",
        [](uintptr_t engine_handle) {
            inference_engine_destroy(reinterpret_cast<InferenceEngine *>(engine_handle));
        },
        py::arg("engine_handle"));

    m.def(
        "reset_engine",
        [](uintptr_t engine_handle) {
            inference_engine_reset(reinterpret_cast<InferenceEngine *>(engine_handle));
        },
        py::arg("engine_handle"));

    m.def(
        "kv_cache_len",
        [](uintptr_t engine_handle) -> int {
            return inference_engine_kv_cache_len(
                reinterpret_cast<InferenceEngine *>(engine_handle));
        },
        py::arg("engine_handle"));

    m.def(
        "next_pos",
        [](uintptr_t engine_handle) -> int {
            return inference_engine_next_pos(reinterpret_cast<InferenceEngine *>(engine_handle));
        },
        py::arg("engine_handle"));

    m.def(
        "kv_cache_num_layers",
        [](uintptr_t engine_handle) -> int {
            InferenceEngine *engine = reinterpret_cast<InferenceEngine *>(engine_handle);
            KVCache *cache = inference_engine_get_kv_cache(engine);
            if (cache == nullptr) {
                return 0;
            }
            return kv_cache_get_num_layers(cache);
        },
        py::arg("engine_handle"));

    m.def(
        "forward_hidden_host",
        [](uintptr_t engine_handle, int num_tokens, int pos_offset,
           py::array_t<float, py::array::f_style | py::array::forcecast> hidden_in,
           py::array_t<float, py::array::f_style | py::array::forcecast> hidden_out) {
            auto in_buf = hidden_in.request();
            auto out_buf = hidden_out.request();
            InferenceEngine *engine = reinterpret_cast<InferenceEngine *>(engine_handle);
            if (inference_engine_forward_hidden_host(engine, static_cast<float *>(in_buf.ptr),
                                                     static_cast<float *>(out_buf.ptr), num_tokens,
                                                     pos_offset) != 0) {
                throw std::runtime_error("inference_engine_forward_hidden_host failed");
            }
        },
        py::arg("engine_handle"), py::arg("num_tokens"), py::arg("pos_offset") = 0,
        py::arg("hidden_in"), py::arg("hidden_out"),
        "Test wrapper: hidden H2D + N-layer forward + final_norm + D2H");

    m.def(
        "forward_token_host",
        [](uintptr_t engine_handle, int num_tokens, int pos_offset,
           py::array_t<int, py::array::c_style | py::array::forcecast> token_ids,
           py::array_t<float, py::array::f_style | py::array::forcecast> logits_out) {
            auto tok_buf = token_ids.request();
            auto log_buf = logits_out.request();
            if (tok_buf.ndim != 1 || static_cast<int>(tok_buf.shape[0]) != num_tokens ||
                num_tokens <= 0) {
                throw std::runtime_error("token_ids must be 1-D length num_tokens");
            }
            if (log_buf.ndim != 2 || static_cast<int>(log_buf.shape[1]) != num_tokens) {
                throw std::runtime_error("logits_out must be [vocab_size, num_tokens] Fortran");
            }
            InferenceEngine *engine = reinterpret_cast<InferenceEngine *>(engine_handle);
            if (inference_engine_forward_token_host(engine, static_cast<const int *>(tok_buf.ptr),
                                                    static_cast<float *>(log_buf.ptr), num_tokens,
                                                    pos_offset) != 0) {
                throw std::runtime_error("inference_engine_forward_token_host failed");
            }
        },
        py::arg("engine_handle"), py::arg("num_tokens"), py::arg("pos_offset") = 0,
        py::arg("token_ids"), py::arg("logits_out"),
        "Test wrapper: token H2D + embed + N-layer forward + lm_head + D2H logits [vocab,T]");
}
