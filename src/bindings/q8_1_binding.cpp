// q8_1_binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // std::vector<int>

namespace py = pybind11;

extern "C" void q8_1(float *, uint8_t *, std::vector<int> &);
extern "C" void parse_q8_1_output(const uint8_t *, int8_t *, float *, float *, int64_t);

PYBIND11_MODULE(q8_1_me, m) {
    m.def(
        "quantize",
        [](py::array_t<float> input, py::array_t<uint8_t> output, std::vector<int> dims) {
            auto input_buf = input.request();
            auto output_buf = output.request();
            q8_1((float *)input_buf.ptr, (uint8_t *)output_buf.ptr, dims);
        },
        "Quantize float32 to Q8_1");

    m.def(
        "parse",
        [](py::array_t<uint8_t> raw, int64_t n_blocks) {
            auto raw_buf = raw.request();

            // 分配输出
            std::vector<size_t> shape_qs = {static_cast<size_t>(n_blocks), 128};
            std::vector<size_t> shape_scale = {static_cast<size_t>(n_blocks), 4};

            auto qs = py::array_t<int8_t>(shape_qs);
            auto scale = py::array_t<float>(shape_scale);
            auto sum = py::array_t<float>(shape_scale);

            auto qs_buf = qs.request();
            auto scale_buf = scale.request();
            auto sum_buf = sum.request();

            parse_q8_1_output((uint8_t *)raw_buf.ptr, (int8_t *)qs_buf.ptr, (float *)scale_buf.ptr,
                              (float *)sum_buf.ptr, n_blocks);

            return py::make_tuple(qs, scale, sum);
        },
        "Parse raw Q8_1 output");
}
