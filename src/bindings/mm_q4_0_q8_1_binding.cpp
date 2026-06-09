#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

extern "C" void mat_mul_q4_0_q8_1(const void *weight_q4_0, float *input, uint8_t *quant_workspace,
                                  float *output, std::vector<int> &input_dims, int nrows_out,
                                  int ncols_k);

PYBIND11_MODULE(mm_q4_0_q8_1_me, m) {
    m.def(
        "mmq",
        [](py::array_t<uint8_t> weight, py::array_t<float> input, py::array_t<uint8_t> quant_buf,
           py::array_t<float> output, std::vector<int> input_dims, int nrows_out, int ncols_k) {
            mat_mul_q4_0_q8_1(weight.data(), input.mutable_data(), quant_buf.mutable_data(),
                              output.mutable_data(), input_dims, nrows_out, ncols_k);
        },
        py::arg("weight_q4_0"), py::arg("input"), py::arg("quant_workspace"), py::arg("output"),
        py::arg("input_dims"), py::arg("nrows_out"), py::arg("ncols_k"),
        "Q4_0 weight x float activation (quantize via q8_1) -> float output");
}
