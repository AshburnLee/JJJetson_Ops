// q8_1_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // std::vector<int>

namespace py = pybind11;

extern "C" void roll(float* input, float* output, std::vector<int>& input_dims, std::vector<int>& shifts);

PYBIND11_MODULE(roll_me, m) {
    m.doc() = "Python binding for CUDA roll";
    m.def("roll", [](py::array_t<float> input, py::array_t<float> output, std::vector<int> dims, std::vector<int> shifts) {
        auto input_buf = input.request();
        auto output_buf = output.request();

        float* input_ptr = static_cast<float*>(input_buf.ptr);
        float* output_ptr = static_cast<float*>(output_buf.ptr);

        roll(input_ptr, output_ptr, dims, shifts);
    }, "rope intput to get output");
}
