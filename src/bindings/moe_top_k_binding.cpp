// moe_top_k_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // std::vector<int>

namespace py = pybind11;

extern "C" void moe_top_k(const float* logits, 
                          const int topk, 
                          float* weights, 
                          int* ids, 
                          const std::vector<int>& input_dims);

PYBIND11_MODULE(moe_top_k_me, m) {
    m.doc() = "Python binding for CUDA moe_top_k (MoE expert routing top-k)";
    m.def("moe_top_k", [](py::array_t<float> input, 
                         int topk,
                         py::array_t<float> weights, 
                         py::array_t<int> ids, 
                         const std::vector<int> input_dims) {
        auto input_buf   = input.request();
        auto weights_buf = weights.request();
        auto ids_buf     = ids.request();

        float* input_ptr  = static_cast<float*>(input_buf.ptr);
        float* output_ptr = static_cast<float*>(weights_buf.ptr);
        int* ids_ptr      = static_cast<int*>(ids_buf.ptr);

        moe_top_k(input_ptr, topk, output_ptr, ids_ptr, input_dims);
    }, 
    py::arg("logits"), py::arg("topk"),
    py::arg("weights"), py::arg("ids"),
    py::arg("input_dims"),
    "Fused kernel: top-k expert ids and softmax weights from logits (MoE gate)"
);
}
