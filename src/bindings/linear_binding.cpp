#include "linear.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(linear_me, m) {
    m.doc() = "Python binding for cuBLAS Linear (GEMM) for Q/K/V GEMM, up/gate/down GEMM";
    m.def(
        "forward_host",
        [](py::array_t<float> input, py::array_t<float> weight, py::array_t<float> output,
           int in_features, int num_tokens, int out_features) {
            float *input_ptr = static_cast<float *>(input.request().ptr);
            float *weight_ptr = static_cast<float *>(weight.request().ptr);
            float *output_ptr = static_cast<float *>(output.request().ptr);
            linear_forward_host(input_ptr, weight_ptr, output_ptr, in_features, num_tokens,
                                out_features);
        },
        py::arg("input"), py::arg("weight"), py::arg("output"), py::arg("in_features"),
        py::arg("num_tokens"), py::arg("out_features"),
        "Test wrapper: host I/O around linear_forward_device");
}
