#include "rms_norm.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(rms_norm_fused_add_me, m) {
    m.doc() = "Python binding for CUDA fused add RMSNorm kernel";
    m.def(
        "forward_host",
        [](py::array_t<float> input, py::array_t<float> residual, py::array_t<float> weight,
           int hidden_size, int num_tokens, float epsilon) {
            float *input_ptr = static_cast<float *>(input.request().ptr);
            float *residual_ptr = static_cast<float *>(residual.request().ptr);
            const float *weight_ptr = static_cast<const float *>(weight.request().ptr);
            rms_norm_fused_add_forward_host(input_ptr, residual_ptr, weight_ptr, hidden_size,
                                            num_tokens, epsilon);
        },
        py::arg("input"), py::arg("residual"), py::arg("weight"), py::arg("hidden_size"),
        py::arg("num_tokens"), py::arg("epsilon") = 1e-6f,
        "Test wrapper: host I/O around rms_norm_fused_add_forward_device");
}
