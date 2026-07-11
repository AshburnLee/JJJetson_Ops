#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

extern "C" void linear(float *input, float *weight, float *output, std::vector<int> &input_dims,
                       int out_features);

PYBIND11_MODULE(linear_me, m) {
    m.doc() = "Python binding for cuBLAS Linear (GEMM) for Q/K/V GEMM, up/gate/down GEMM";
    m.def(
        "linear",
        [](py::array_t<float> input, py::array_t<float> weight, py::array_t<float> output,
           std::vector<int> input_dims, int out_features) {
            auto input_buf = input.request();
            auto weight_buf = weight.request();
            auto output_buf = output.request();

            float *input_ptr = static_cast<float *>(input_buf.ptr);
            float *weight_ptr = static_cast<float *>(weight_buf.ptr);
            float *output_ptr = static_cast<float *>(output_buf.ptr);

            linear(input_ptr, weight_ptr, output_ptr, input_dims, out_features);
        },
        py::arg("input"), py::arg("weight"), py::arg("output"), py::arg("input_dims"),
        py::arg("out_features"),
        "Linear GEMM: col-major input [in, tokens, 1, 1], weight [out, in] row-major");
}
