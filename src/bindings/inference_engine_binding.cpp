#include "inference_engine.h"

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
}
