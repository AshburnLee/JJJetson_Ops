#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <sstream>
#include <string>

namespace py = pybind11;

extern "C" size_t moe_dispatch_sota_sort_workspace_bytes(int num_tokens,
                                                         int top_k,
                                                         int num_experts);

extern "C" void moe_dispatch_sota(const float* x_host,
                                  const int* expert_ids_host,
                                  int num_tokens,
                                  int top_k,
                                  int hidden_size,
                                  int num_experts,
                                  float* permuted_x_host,
                                  int* source_token_host,
                                  int* source_k_host,
                                  int* expert_offsets_host);

PYBIND11_MODULE(moe_dispatch_sota_me, m) {
    m.doc() = "MoE dispatch (vLLM-style permute): CUB sort by expert + row expand (host I/O)";
    m.def(
        "moe_dispatch_sota_sort_workspace_bytes",
        [](int num_tokens, int top_k, int num_experts) {
            return moe_dispatch_sota_sort_workspace_bytes(num_tokens, top_k, num_experts);
        },
        py::arg("num_tokens"),
        py::arg("top_k"),
        py::arg("num_experts"),
        "Temp storage bytes for DeviceRadixSort (padded), for device launch.");
    m.def(
        "moe_dispatch_sota",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> x,
           py::array_t<int, py::array::c_style | py::array::forcecast> expert_ids,
           int top_k,
           int num_experts,
           py::array_t<float, py::array::c_style> permuted_x,
           py::array_t<int, py::array::c_style> source_token,
           py::array_t<int, py::array::c_style> source_k,
           py::array_t<int, py::array::c_style> expert_offsets) {
            auto bx = x.request();
            auto bid = expert_ids.request();
            auto bperm = permuted_x.request();
            auto bst = source_token.request();
            auto bsk = source_k.request();
            auto boff = expert_offsets.request();

            if (bx.ndim != 2) {
                throw std::invalid_argument("x must be 2-D (num_tokens, hidden_size)");
            }
            const int num_tokens = static_cast<int>(bx.shape[0]);
            const int hidden_size = static_cast<int>(bx.shape[1]);
            const int num_routes = num_tokens * top_k;

            if (bid.ndim != 1 || bid.shape[0] != num_routes) {
                throw std::invalid_argument("expert_ids must be 1-D of length num_tokens * top_k");
            }
            if (bperm.ndim != 2 || bperm.shape[0] != num_routes ||
                bperm.shape[1] != hidden_size) {
                std::ostringstream oss;
                oss << "permuted_x must be shape (" << num_routes << ", " << hidden_size << ")";
                throw std::invalid_argument(oss.str());
            }
            if (bst.ndim != 1 || bst.shape[0] != num_routes) {
                throw std::invalid_argument("source_token must be 1-D length num_tokens * top_k");
            }
            if (bsk.ndim != 1 || bsk.shape[0] != num_routes) {
                throw std::invalid_argument("source_k must be 1-D length num_tokens * top_k");
            }
            if (boff.ndim != 1 || boff.shape[0] != num_experts + 1) {
                throw std::invalid_argument("expert_offsets must be 1-D length num_experts + 1");
            }

            float* x_ptr = static_cast<float*>(bx.ptr);
            int* ids_ptr = static_cast<int*>(bid.ptr);
            float* perm_ptr = static_cast<float*>(bperm.ptr);
            int* st_ptr = static_cast<int*>(bst.ptr);
            int* sk_ptr = static_cast<int*>(bsk.ptr);
            int* off_ptr = static_cast<int*>(boff.ptr);

            moe_dispatch_sota(x_ptr, ids_ptr, num_tokens, top_k, hidden_size, num_experts,
                              perm_ptr, st_ptr, sk_ptr, off_ptr);
        },
        py::arg("x"),
        py::arg("expert_ids"),
        py::arg("top_k"),
        py::arg("num_experts"),
        py::arg("permuted_x"),
        py::arg("source_token"),
        py::arg("source_k"),
        py::arg("expert_offsets"),
        "Sort routes by expert (CUB), write permuted_x and expert_offsets like vLLM moe_permute.");
}
