#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" void cpy_transpose(const char *csrc, char *cdst, const std::vector<int> &src_dims);

PYBIND11_MODULE(cpy_transpose_me, m) {
    m.doc() = "python binding for CUDA copy transpose kernel";
    m.def(
        "cpy_trans",
        [](py::buffer src, py::buffer dst, const std::vector<int> src_dims) {
            auto src_buff = src.request();
            auto dst_buff = dst.request();

            cpy_transpose(static_cast<const char *>(src_buff.ptr),
                          static_cast<char *>(dst_buff.ptr), src_dims);
        },
        py::arg("src"), py::arg("dst"), py::arg("src_dims"));
}
