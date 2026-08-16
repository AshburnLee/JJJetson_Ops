#include "generate_loop.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

#include "inference_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(generate_loop_me, m) {
    m.doc() = "GenerateLoop: prefill + decode orchestration (Phase 2 Module 4)";

    m.def(
        "generate",
        [](uintptr_t engine_handle,
           py::array_t<int, py::array::c_style | py::array::forcecast> prompt_token_ids,
           int max_new_tokens, int eos_token_id) -> py::list {
            auto buf = prompt_token_ids.request();
            if (buf.ndim != 1 || buf.shape[0] <= 0) {
                throw std::runtime_error("prompt_token_ids must be non-empty 1-D");
            }
            if (max_new_tokens <= 0) {
                return py::list();
            }
            InferenceEngine *engine = reinterpret_cast<InferenceEngine *>(engine_handle);
            const int prompt_len = static_cast<int>(buf.shape[0]);
            std::vector<int> out(static_cast<size_t>(max_new_tokens));
            const int n =
                generate_loop_run(engine, static_cast<const int *>(buf.ptr), prompt_len,
                                  max_new_tokens, eos_token_id, out.data(), max_new_tokens);
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
        py::arg("eos_token_id") = -1);
}
