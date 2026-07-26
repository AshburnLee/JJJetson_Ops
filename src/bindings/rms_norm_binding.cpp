#include "rms_norm.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(rms_norm_me, m) {
    m.doc() = "Python binding for CUDA RMSNorm kernel";
    m.def(
        "forward_host",
        [](py::array_t<float> input, py::array_t<float> weight, py::array_t<float> output,
           int hidden_size, int num_tokens, float epsilon) {
            const float *input_ptr = static_cast<const float *>(input.request().ptr);
            const float *weight_ptr = static_cast<const float *>(weight.request().ptr);
            float *output_ptr = static_cast<float *>(output.request().ptr);
            rms_norm_forward_host(input_ptr, weight_ptr, output_ptr, hidden_size, num_tokens,
                                  epsilon);
        },
        py::arg("input"), py::arg("weight"), py::arg("output"), py::arg("hidden_size"),
        py::arg("num_tokens"), py::arg("epsilon") = 1e-6f,
        "Test wrapper: host I/O around rms_norm_forward_device");
}
