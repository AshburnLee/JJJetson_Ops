#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

extern "C" void layer_norm(float *input, float *weight, float *bias, float *output,
                           std::vector<int> &input_dims, float epsilon);

PYBIND11_MODULE(layer_norm_me, m) {
    m.doc() = "Python binding for CUDA LayerNorm kernel";
    m.def(
        "layer_norm",
        [](py::array_t<float> input, py::array_t<float> weight, py::array_t<float> bias,
           py::array_t<float> output, std::vector<int> dims, float epsilon) {
            auto input_buf = input.request();
            auto weight_buf = weight.request();
            auto bias_buf = bias.request();
            auto output_buf = output.request();

            float *input_ptr = static_cast<float *>(input_buf.ptr);
            float *weight_ptr = static_cast<float *>(weight_buf.ptr);
            float *bias_ptr = static_cast<float *>(bias_buf.ptr);
            float *output_ptr = static_cast<float *>(output_buf.ptr);

            layer_norm(input_ptr, weight_ptr, bias_ptr, output_ptr, dims, epsilon);
        },
        py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("output"),
        py::arg("input_dims"), py::arg("epsilon") = 1e-6f,
        "LayerNorm on col-major [hidden_size, num_tokens, 1, 1]");
}
