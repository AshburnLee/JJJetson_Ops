#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <sstream>
#include <string>

namespace py = pybind11;

extern "C" void moe_combine_sota(const float *expert_out_host, const int *source_token_host,
                                 const int *source_k_host, const float *route_weights_host,
                                 float *y_host, int num_routes, int hidden_size, int num_tokens,
                                 int top_k);

PYBIND11_MODULE(moe_combine_sota_me, m) {
    m.doc() = "MoE combine (vLLM unpermute style): token-block weighted reduce, no atomics";
    m.def(
        "moe_combine_sota",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> expert_out,
           py::array_t<int, py::array::c_style | py::array::forcecast> source_token,
           py::array_t<int, py::array::c_style | py::array::forcecast> source_k,
           py::array_t<float, py::array::c_style | py::array::forcecast> route_weights,
           py::array_t<float, py::array::c_style> y, int num_tokens, int top_k) {
            auto be = expert_out.request();
            auto bst = source_token.request();
            auto bsk = source_k.request();
            auto brw = route_weights.request();
            auto by = y.request();

            if (be.ndim != 2) {
                throw std::invalid_argument("expert_out must be 2-D (num_routes, hidden_size)");
            }
            const int num_routes = static_cast<int>(be.shape[0]);
            const int hidden_size = static_cast<int>(be.shape[1]);

            if (bst.ndim != 1 || bst.shape[0] != num_routes) {
                throw std::invalid_argument("source_token must be 1-D of length num_routes");
            }
            if (bsk.ndim != 1 || bsk.shape[0] != num_routes) {
                throw std::invalid_argument("source_k must be 1-D of length num_routes");
            }
            if (num_tokens * top_k != num_routes) {
                throw std::invalid_argument("num_tokens * top_k must equal expert_out.shape[0]");
            }
            if (brw.ndim != 2 || brw.shape[0] != num_tokens || brw.shape[1] != top_k) {
                throw std::invalid_argument("route_weights must be (num_tokens, top_k)");
            }
            if (by.ndim != 2 || by.shape[0] != num_tokens || by.shape[1] != hidden_size) {
                std::ostringstream oss;
                oss << "y must be shape (" << num_tokens << ", " << hidden_size << ")";
                throw std::invalid_argument(oss.str());
            }

            moe_combine_sota(static_cast<float *>(be.ptr), static_cast<int *>(bst.ptr),
                             static_cast<int *>(bsk.ptr), static_cast<float *>(brw.ptr),
                             static_cast<float *>(by.ptr), num_routes, hidden_size, num_tokens,
                             top_k);
        },
        py::arg("expert_out"), py::arg("source_token"), py::arg("source_k"),
        py::arg("route_weights"), py::arg("y"), py::arg("num_tokens"), py::arg("top_k"),
        "Build inv_permuted_idx from dispatch_sota metadata, then unpermute+reduce into y.");
}
