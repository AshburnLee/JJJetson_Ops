// q8_1_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // std::vector<int>

namespace py = pybind11;

extern "C" void top_k_moe(const float* logits, 
                          const int topk, 
                          float* weights, 
                          int* ids, 
                          const std::vector<int>& input_dims);

PYBIND11_MODULE(top_k_moe_me, m) {
    m.doc() = "Python binding for CUDA top_k_moe";
    m.def("top_k_moe", [](py::array_t<float> input, 
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

        top_k_moe(input_ptr, topk, output_ptr, ids_ptr, input_dims);
    }, 
    py::arg("logits"), py::arg("topk"),
    py::arg("weights"), py::arg("ids"),
    py::arg("input_dims"),
    "a fused kernel to get top-k experts"
);
}
