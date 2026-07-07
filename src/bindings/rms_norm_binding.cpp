#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

extern "C" void rms_norm(float *input, float *weight, float *output, std::vector<int> &input_dims,
                         float epsilon);

PYBIND11_MODULE(rms_norm_me, m) {
    m.doc() = "Python binding for CUDA RMSNorm kernel";
    m.def(
        "rms_norm",
        [](py::array_t<float> input, py::array_t<float> weight, py::array_t<float> output,
           std::vector<int> dims, float epsilon) {
            auto input_buf = input.request();
            auto weight_buf = weight.request();
            auto output_buf = output.request();

            float *input_ptr = static_cast<float *>(input_buf.ptr);
            float *weight_ptr = static_cast<float *>(weight_buf.ptr);
            float *output_ptr = static_cast<float *>(output_buf.ptr);

            rms_norm(input_ptr, weight_ptr, output_ptr, dims, epsilon);
        },
        py::arg("input"), py::arg("weight"), py::arg("output"), py::arg("input_dims"),
        py::arg("epsilon") = 1e-6f, "RMSNorm on col-major [hidden_size, num_tokens, 1, 1]");
}
