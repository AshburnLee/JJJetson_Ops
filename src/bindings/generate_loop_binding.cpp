#include "generate_loop.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

#include "inference_engine.h"
#include "sampler_top_k.h"

namespace py = pybind11;

PYBIND11_MODULE(generate_loop_me, m) {
    m.doc() = "GenerateLoop: prefill + decode orchestration (Phase 2 Module 4)";

    m.def(
        "sampler_top_k_host",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> logits, int top_k,
           uint64_t seed) -> int {
            auto buf = logits.request();
            if (buf.ndim != 1 || buf.shape[0] <= 0) {
                throw std::runtime_error("logits must be non-empty 1-D float32");
            }
            const int vocab_size = static_cast<int>(buf.shape[0]);
            const int token =
                sampler_top_k_host(static_cast<const float *>(buf.ptr), vocab_size, top_k, seed);
            if (token < 0) {
                throw std::runtime_error("sampler_top_k_host failed");
            }
            return token;
        },
        py::arg("logits"), py::arg("top_k"), py::arg("seed") = 0,
        "Test wrapper: host top-k sample on logits [vocab]");

    m.def(
        "generate",
        [](uintptr_t engine_handle,
           py::array_t<int, py::array::c_style | py::array::forcecast> prompt_token_ids,
           int max_new_tokens, int eos_token_id, int top_k, uint64_t seed) -> py::list {
            auto buf = prompt_token_ids.request();
            if (buf.ndim != 1 || buf.shape[0] <= 0) {
                throw std::runtime_error("prompt_token_ids must be non-empty 1-D");
            }
            if (max_new_tokens <= 0) {
                return py::list();
            }
            if (top_k <= 0) {
                throw std::runtime_error("top_k must be positive");
            }
            InferenceEngine *engine = reinterpret_cast<InferenceEngine *>(engine_handle);
            const int prompt_len = static_cast<int>(buf.shape[0]);
            std::vector<int> out(static_cast<size_t>(max_new_tokens));
            const int n = generate_loop_run(engine, static_cast<const int *>(buf.ptr), prompt_len,
                                            max_new_tokens, eos_token_id, top_k, seed, out.data(),
                                            max_new_tokens);
            if (n < 0) {
                throw std::runtime_error("generate_loop_run failed");
            }
            py::list result;
            for (int i = 0; i < n; ++i) {
                result.append(out[static_cast<size_t>(i)]);
            }
            return result;
        },
        py::arg("engine_handle"), py::arg("prompt_token_ids"), py::arg("max_new_tokens"),
        py::arg("eos_token_id") = -1, py::arg("top_k") = 1, py::arg("seed") = 0);
}
