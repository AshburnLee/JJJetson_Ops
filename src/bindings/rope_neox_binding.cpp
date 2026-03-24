// q8_1_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // std::vector<int>

namespace py = pybind11;

extern "C" void rope(float* input, int* pos, float* output, std::vector<int>& input_dims);

PYBIND11_MODULE(rope_me, m) {
    m.doc() = "Python binding for CUDA RoPE";
    m.def("RoPE", [](py::array_t<float> input, py::array_t<int> pos,py::array_t<float> output, std::vector<int> dims) {
        auto input_buf = input.request();
        auto pos_buf = pos.request();
        auto output_buf = output.request();

        float* input_ptr = static_cast<float*>(input_buf.ptr);
        int* pos_ptr = static_cast<int*>(pos_buf.ptr);
        float* output_ptr = static_cast<float*>(output_buf.ptr);

        rope(input_ptr, pos_ptr, output_ptr, dims);
    }, "rope intput to get output");
}
