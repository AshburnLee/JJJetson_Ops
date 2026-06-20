#include "moe_cuda_graph_cache.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(moe_cuda_graph_cache_me, m) {
    m.doc() = "CUDA Graph capture/replay wrapper for moe_pipeline_sota_forward_device";

    m.def(
        "create_cache",
        [](int hidden_size, int intermediate_size, int num_experts, int top_k) -> uintptr_t {
            MoECudaGraphCache *cache = moe_cuda_graph_cache_create(hidden_size, intermediate_size,
                                                                   num_experts, top_k, nullptr);
            if (cache == nullptr) {
                throw std::runtime_error("moe_cuda_graph_cache_create failed");
            }
            return reinterpret_cast<uintptr_t>(cache);
        },
        py::arg("hidden_size"), py::arg("intermediate_size"), py::arg("num_experts"),
        py::arg("top_k"));

    m.def(
        "destroy_cache",
        [](uintptr_t cache_handle) {
            moe_cuda_graph_cache_destroy(reinterpret_cast<MoECudaGraphCache *>(cache_handle));
        },
        py::arg("cache_handle"));

    m.def(
        "has_graph",
        [](uintptr_t cache_handle, int num_tokens) {
            return moe_cuda_graph_cache_has(reinterpret_cast<MoECudaGraphCache *>(cache_handle),
                                            num_tokens) != 0;
        },
        py::arg("cache_handle"), py::arg("num_tokens"));

    m.def(
        "capture",
        [](uintptr_t cache_handle, int num_tokens) {
            if (moe_cuda_graph_cache_capture(reinterpret_cast<MoECudaGraphCache *>(cache_handle),
                                             num_tokens) != 0) {
                throw std::runtime_error("moe_cuda_graph_cache_capture failed");
            }
        },
        py::arg("cache_handle"), py::arg("num_tokens"));

    m.def(
        "replay",
        [](uintptr_t cache_handle, int num_tokens) {
            if (moe_cuda_graph_cache_replay(reinterpret_cast<MoECudaGraphCache *>(cache_handle),
                                            num_tokens) != 0) {
                throw std::runtime_error("moe_cuda_graph_cache_replay failed");
            }
        },
        py::arg("cache_handle"), py::arg("num_tokens"));

    m.def(
        "run_graph",
        [](uintptr_t cache_handle, int num_tokens,
           py::array_t<float, py::array::c_style | py::array::forcecast> x,
           py::array_t<float, py::array::c_style | py::array::forcecast> logits,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_gate,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_up,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_down,
           py::array_t<float, py::array::c_style> y, int num_experts) {
            auto bx = x.request();
            auto blog = logits.request();
            auto by = y.request();

            if (static_cast<int>(bx.shape[0]) != num_tokens) {
                throw std::invalid_argument("x.shape[0] must equal num_tokens");
            }
            if (blog.shape[0] != num_tokens ||
                blog.shape[1] != static_cast<py::ssize_t>(num_experts)) {
                throw std::invalid_argument("logits shape mismatch");
            }
            if (by.shape[0] != num_tokens || by.shape[1] != static_cast<py::ssize_t>(bx.shape[1])) {
                throw std::invalid_argument("y shape mismatch");
            }

            MoECudaGraphCache *cache = reinterpret_cast<MoECudaGraphCache *>(cache_handle);
            if (moe_cuda_graph_cache_run(cache, num_tokens, static_cast<float *>(bx.ptr),
                                         static_cast<float *>(blog.ptr),
                                         static_cast<float *>(w_gate.request().ptr),
                                         static_cast<float *>(w_up.request().ptr),
                                         static_cast<float *>(w_down.request().ptr),
                                         static_cast<float *>(by.ptr)) != 0) {
                throw std::runtime_error("moe_cuda_graph_cache_run failed");
            }
        },
        py::arg("cache_handle"), py::arg("num_tokens"), py::arg("x"), py::arg("logits"),
        py::arg("w_gate"), py::arg("w_up"), py::arg("w_down"), py::arg("y"),
        py::arg("num_experts"));
}
