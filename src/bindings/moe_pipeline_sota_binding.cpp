#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

extern "C" void moe_pipeline_sota_forward(const float *x_host, const float *logits_host,
                                          const float *w_gate_host, const float *w_up_host,
                                          const float *w_down_host, float *y_host, int num_tokens,
                                          int hidden_size, int intermediate_size, int num_experts,
                                          int top_k, void *stream);

PYBIND11_MODULE(moe_pipeline_sota_me, m) {
    m.doc() = "Full MoE sota pipeline: top-k + dispatch_sota + grouped GEMM + combine_sota";
    m.def(
        "moe_pipeline_sota_forward",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> x,
           py::array_t<float, py::array::c_style | py::array::forcecast> logits,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_gate,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_up,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_down,
           py::array_t<float, py::array::c_style> y, int intermediate_size, int num_experts,
           int top_k) {
            auto bx = x.request();
            auto blog = logits.request();
            auto bwg = w_gate.request();
            auto bwu = w_up.request();
            auto bwd = w_down.request();
            auto by = y.request();

            if (bx.ndim != 2) {
                throw std::invalid_argument("x must be 2-D (num_tokens, hidden_size)");
            }
            const int num_tokens = static_cast<int>(bx.shape[0]);
            const int hidden_size = static_cast<int>(bx.shape[1]);

            if (blog.ndim != 2 || blog.shape[0] != num_tokens || blog.shape[1] != num_experts) {
                std::ostringstream oss;
                oss << "logits must be shape (" << num_tokens << ", " << num_experts << ")";
                throw std::invalid_argument(oss.str());
            }
            const size_t w_gu_elems =
                static_cast<size_t>(num_experts) * intermediate_size * hidden_size;
            if (bwg.ndim != 1 || static_cast<size_t>(bwg.shape[0]) != w_gu_elems) {
                throw std::invalid_argument("w_gate flat length must be num_experts*I*H");
            }
            if (bwu.ndim != 1 || static_cast<size_t>(bwu.shape[0]) != w_gu_elems) {
                throw std::invalid_argument("w_up flat length must match w_gate");
            }
            const size_t w_d_elems =
                static_cast<size_t>(num_experts) * hidden_size * intermediate_size;
            if (bwd.ndim != 1 || static_cast<size_t>(bwd.shape[0]) != w_d_elems) {
                throw std::invalid_argument("w_down flat length must be num_experts*H*I");
            }
            if (by.ndim != 2 || by.shape[0] != num_tokens || by.shape[1] != hidden_size) {
                throw std::invalid_argument("y must be (num_tokens, hidden_size)");
            }

            moe_pipeline_sota_forward(static_cast<float *>(bx.ptr), static_cast<float *>(blog.ptr),
                                      static_cast<float *>(bwg.ptr), static_cast<float *>(bwu.ptr),
                                      static_cast<float *>(bwd.ptr), static_cast<float *>(by.ptr),
                                      num_tokens, hidden_size, intermediate_size, num_experts,
                                      top_k, nullptr);
        },
        py::arg("x"), py::arg("logits"), py::arg("w_gate"), py::arg("w_up"), py::arg("w_down"),
        py::arg("y"), py::arg("intermediate_size"), py::arg("num_experts"), py::arg("top_k"));
}
