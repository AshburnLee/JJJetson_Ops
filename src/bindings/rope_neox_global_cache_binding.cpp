#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

extern "C" void init_cossin_cache(int max_len, int n_dims, float freq_base);
extern "C" void destroy_cossin_cache();
extern "C" void rope_with_global_cossin_cache(float *input, int *pos, float *output,
                                              std::vector<int> &input_dims);

PYBIND11_MODULE(rope_global_cache_me, m) {
    m.doc() = "Python binding for CUDA RoPE (global cos/sin lookup table)";

    m.def(
        "init_cossin_cache",
        [](int max_len, int n_dims, float freq_base) {
            init_cossin_cache(max_len, n_dims, freq_base);
        },
        py::arg("max_len"), py::arg("n_dims"), py::arg("freq_base") = 10000.f,
        "Precompute and upload global cos/sin cache for positions [0, max_len)");

    m.def("destroy_cossin_cache", []() { destroy_cossin_cache(); }, "Release global cos/sin cache");

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

            rope_with_global_cossin_cache(input_ptr, pos_ptr, output_ptr, dims);
        },
        py::arg("input"), py::arg("pos"), py::arg("output"), py::arg("input_dims"),
        "RoPE forward using global cos/sin cache (requires init_cossin_cache first)");
}
