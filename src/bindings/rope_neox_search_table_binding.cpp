#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

extern "C" void rope_with_search_table(float *input, int *pos, float *output,
                                       std::vector<int> &input_dims);

PYBIND11_MODULE(rope_search_table_me, m) {
    m.doc() = "Python binding for CUDA RoPE (Host cos/sin lookup table)";
    m.def(
        "RoPE",
        [](py::array_t<float> input, py::array_t<int> pos, py::array_t<float> output,
           std::vector<int> dims) {
            auto input_buf = input.request();
            auto pos_buf = pos.request();
            auto output_buf = output.request();

            float *input_ptr = static_cast<float *>(input_buf.ptr);
            int *pos_ptr = static_cast<int *>(pos_buf.ptr);
            float *output_ptr = static_cast<float *>(output_buf.ptr);

            rope_with_search_table(input_ptr, pos_ptr, output_ptr, dims);
        },
        py::arg("input"), py::arg("pos"), py::arg("output"), py::arg("input_dims"),
        "RoPE with Host-precomputed cos/sin table");
}
