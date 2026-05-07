#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <sstream>

namespace py = pybind11;

extern "C" void moe_swiglu_experts_forward(const float* x_host,
                                           const int* expert_ids_host,
                                           const float* route_weights_host,
                                           const float* w_gate_host,
                                           const float* w_up_host,
                                           const float* w_down_host,
                                           float* y_host,
                                           int num_tokens,
                                           int hidden_size,
                                           int intermediate_size,
                                           int num_experts,
                                           int top_k,
                                           cudaStream_t stream);

PYBIND11_MODULE(moe_pipeline_me, m) {
    m.doc() =
        "Full MoE SwiGLU expert pipeline (dispatch -> gate/up/down GEMM -> silu multiply -> combine)";
    m.def(
        "moe_swiglu_experts_forward",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> x,
           py::array_t<int, py::array::c_style | py::array::forcecast> expert_ids,
           py::array_t<float, py::array::c_style | py::array::forcecast> route_weights,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_gate,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_up,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_down,
           py::array_t<float, py::array::c_style> y,
           int intermediate_size,
           int num_experts,
           int top_k) {
            auto bx = x.request();
            auto bid = expert_ids.request();
            auto brw = route_weights.request();
            auto bwg = w_gate.request();
            auto bwu = w_up.request();
            auto bwd = w_down.request();
            auto by = y.request();

            if (bx.ndim != 2) {
                throw std::invalid_argument("x must be 2-D (num_tokens, hidden_size)");
            }
            const int num_tokens = static_cast<int>(bx.shape[0]);
            const int hidden_size = static_cast<int>(bx.shape[1]);
            const int num_routes = num_tokens * top_k;

            if (bid.ndim != 1 || bid.shape[0] != num_routes) {
                throw std::invalid_argument("expert_ids must be flat length num_tokens * top_k");
            }
            if (brw.ndim != 2 || brw.shape[0] != num_tokens || brw.shape[1] != top_k) {
                throw std::invalid_argument("route_weights must be (num_tokens, top_k)");
            }
            const size_t w_gu_elems = static_cast<size_t>(num_experts) * intermediate_size * hidden_size;
            if (bwg.ndim != 1 || static_cast<size_t>(bwg.shape[0]) != w_gu_elems) {
                std::ostringstream oss;
                oss << "w_gate must be flat length num_experts*I*H = " << w_gu_elems;
                throw std::invalid_argument(oss.str());
            }
            if (bwu.ndim != 1 || static_cast<size_t>(bwu.shape[0]) != w_gu_elems) {
                throw std::invalid_argument("w_up must match w_gate flat length");
            }
            const size_t w_d_elems = static_cast<size_t>(num_experts) * hidden_size * intermediate_size;
            if (bwd.ndim != 1 || static_cast<size_t>(bwd.shape[0]) != w_d_elems) {
                std::ostringstream oss;
                oss << "w_down must be flat length num_experts*H*I = " << w_d_elems;
                throw std::invalid_argument(oss.str());
            }
            if (by.ndim != 2 || by.shape[0] != num_tokens || by.shape[1] != hidden_size) {
                throw std::invalid_argument("y must be (num_tokens, hidden_size)");
            }

            float* x_ptr = static_cast<float*>(bx.ptr);
            int* ids_ptr = static_cast<int*>(bid.ptr);
            float* rw_ptr = static_cast<float*>(brw.ptr);
            float* wg_ptr = static_cast<float*>(bwg.ptr);
            float* wu_ptr = static_cast<float*>(bwu.ptr);
            float* wd_ptr = static_cast<float*>(bwd.ptr);
            float* y_ptr = static_cast<float*>(by.ptr);

            moe_swiglu_experts_forward(x_ptr, ids_ptr, rw_ptr, wg_ptr, wu_ptr, wd_ptr, y_ptr,
                                       num_tokens, hidden_size, intermediate_size, num_experts,
                                       top_k, nullptr);
        },
        py::arg("x"),
        py::arg("expert_ids"),
        py::arg("route_weights"),
        py::arg("w_gate"),
        py::arg("w_up"),
        py::arg("w_down"),
        py::arg("y"),
        py::arg("intermediate_size"),
        py::arg("num_experts"),
        py::arg("top_k"));
}
