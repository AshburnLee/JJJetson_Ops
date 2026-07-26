#include "layer_norm.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(layer_norm_me, m) {
    m.doc() = "Python binding for CUDA LayerNorm kernel";
    m.def(
        "forward_host",
        [](py::array_t<float> input, py::array_t<float> weight, py::array_t<float> bias,
           py::array_t<float> output, int hidden_size, int num_tokens, float epsilon) {
            float *input_ptr = static_cast<float *>(input.request().ptr);
            float *weight_ptr = static_cast<float *>(weight.request().ptr);
            float *bias_ptr = static_cast<float *>(bias.request().ptr);
            float *output_ptr = static_cast<float *>(output.request().ptr);
            layer_norm_forward_host(input_ptr, weight_ptr, bias_ptr, output_ptr, hidden_size,
                                    num_tokens, epsilon);
        },
        py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("output"),
        py::arg("hidden_size"), py::arg("num_tokens"), py::arg("epsilon") = 1e-6f,
        "Test wrapper: host I/O around layer_norm kernel");
}
