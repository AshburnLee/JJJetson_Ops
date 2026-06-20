#include "moe_runner.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(moe_runner_me, m) {
    m.doc() = "MoE runner: dispatch eager forward_device or CUDA graph replay";

    py::enum_<MoeRunnerDispatch>(m, "Dispatch")
        .value("AUTO", MOE_RUNNER_DISPATCH_AUTO)
        .value("EAGER", MOE_RUNNER_DISPATCH_EAGER)
        .value("GRAPH", MOE_RUNNER_DISPATCH_GRAPH)
        .export_values();

    m.def(
        "create_runner",
        [](int hidden_size, int intermediate_size, int num_experts, int top_k,
           bool enable_graph) -> uintptr_t {
            MoeRunner *runner = moe_runner_create(hidden_size, intermediate_size, num_experts,
                                                  top_k, enable_graph ? 1 : 0, nullptr);
            if (runner == nullptr) {
                throw std::runtime_error("moe_runner_create failed");
            }
            return reinterpret_cast<uintptr_t>(runner);
        },
        py::arg("hidden_size"), py::arg("intermediate_size"), py::arg("num_experts"),
        py::arg("top_k"), py::arg("enable_graph") = true);

    m.def(
        "destroy_runner",
        [](uintptr_t runner_handle) {
            moe_runner_destroy(reinterpret_cast<MoeRunner *>(runner_handle));
        },
        py::arg("runner_handle"));

    m.def(
        "set_dispatch",
        [](uintptr_t runner_handle, MoeRunnerDispatch dispatch) {
            moe_runner_set_dispatch(reinterpret_cast<MoeRunner *>(runner_handle), dispatch);
        },
        py::arg("runner_handle"), py::arg("dispatch"));

    m.def(
        "capture_graph",
        [](uintptr_t runner_handle, int num_tokens) {
            if (moe_runner_capture_graph(reinterpret_cast<MoeRunner *>(runner_handle),
                                         num_tokens) != 0) {
                throw std::runtime_error("moe_runner_capture_graph failed");
            }
        },
        py::arg("runner_handle"), py::arg("num_tokens"));

    m.def(
        "has_graph",
        [](uintptr_t runner_handle, int num_tokens) {
            return moe_runner_has_graph(reinterpret_cast<MoeRunner *>(runner_handle), num_tokens) !=
                   0;
        },
        py::arg("runner_handle"), py::arg("num_tokens"));

    m.def(
        "forward_host",
        [](uintptr_t runner_handle, int num_tokens, bool is_decode,
           py::array_t<float, py::array::c_style | py::array::forcecast> x,
           py::array_t<float, py::array::c_style | py::array::forcecast> logits,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_gate,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_up,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_down,
           py::array_t<float, py::array::c_style> y) {
            auto bx = x.request();
            auto blog = logits.request();
            auto by = y.request();

            if (static_cast<int>(bx.shape[0]) != num_tokens) {
                throw std::invalid_argument("x.shape[0] must equal num_tokens");
            }
            if (by.shape[0] != num_tokens || by.shape[1] != static_cast<py::ssize_t>(bx.shape[1])) {
                throw std::invalid_argument("y shape mismatch");
            }

            MoeRunnerForwardResult result{};
            MoeRunner *runner = reinterpret_cast<MoeRunner *>(runner_handle);
            if (moe_runner_forward_host(
                    runner, static_cast<float *>(bx.ptr), static_cast<float *>(blog.ptr),
                    static_cast<float *>(w_gate.request().ptr),
                    static_cast<float *>(w_up.request().ptr),
                    static_cast<float *>(w_down.request().ptr), static_cast<float *>(by.ptr),
                    num_tokens, is_decode ? 1 : 0, &result) != 0) {
                throw std::runtime_error("moe_runner_forward_host failed");
            }
            return result.used_graph != 0;
        },
        py::arg("runner_handle"), py::arg("num_tokens"), py::arg("is_decode"), py::arg("x"),
        py::arg("logits"), py::arg("w_gate"), py::arg("w_up"), py::arg("w_down"), py::arg("y"));
}
