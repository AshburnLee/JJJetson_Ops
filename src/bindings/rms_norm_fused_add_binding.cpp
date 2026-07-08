#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

extern "C" void rms_norm_fused_add(float *input, float *residual, float *weight,
                                   std::vector<int> &input_dims, float epsilon);

PYBIND11_MODULE(rms_norm_fused_add_me, m) {
    m.doc() = "Python binding for CUDA fused add RMSNorm kernel";
    m.def(
        "rms_norm_fused_add",
        [](py::array_t<float> input, py::array_t<float> residual, py::array_t<float> weight,
           std::vector<int> dims, float epsilon) {
            auto input_buf = input.request();
            auto residual_buf = residual.request();
            auto weight_buf = weight.request();

            float *input_ptr = static_cast<float *>(input_buf.ptr);
            float *residual_ptr = static_cast<float *>(residual_buf.ptr);
            float *weight_ptr = static_cast<float *>(weight_buf.ptr);

            rms_norm_fused_add(input_ptr, residual_ptr, weight_ptr, dims, epsilon);
        },
        py::arg("input"), py::arg("residual"), py::arg("weight"), py::arg("input_dims"),
        py::arg("epsilon") = 1e-6f,
        "Fused add + RMSNorm on col-major [hidden_size, num_tokens, 1, 1]; in-place "
        "input/residual");
}
