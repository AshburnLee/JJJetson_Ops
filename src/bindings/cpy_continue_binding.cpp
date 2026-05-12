#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // std::vector<int>
#include "utils.h"        // data_type

namespace py = pybind11;

extern "C" void cpy_continue(const char *src, char *dst, const std::vector<int> &src_dims,
                             const std::vector<int> &dst_dims,
                             const std::vector<int> &src_byte_stride,
                             const std::vector<int> &dst_byte_stride, data_type src_dt,
                             data_type dst_dt);

PYBIND11_MODULE(cpy_continue_me, m) {
    m.doc() = "Python binding for CUDA copy continue";
    py::enum_<data_type>(m, "data_type")
        .value("f32", data_type::DT_F32)
        .value("bf16", data_type::DT_BF16)
        .value("f16", data_type::DT_F16)
        .value("i32", data_type::DT_I32);
    m.def(
        "cpy_con",
        [](py::buffer src, // 不指明 dtye
           py::buffer dst, const std::vector<int> src_dims, const std::vector<int> dst_dims,
           const std::vector<int> src_byte_stride, const std::vector<int> dst_byte_stride,
           data_type src_dt, data_type dst_dt) {
            auto src_buf = src.request();
            auto dst_buf = dst.request();

            cpy_continue(static_cast<const char *>(src_buf.ptr), static_cast<char *>(dst_buf.ptr),
                         src_dims, dst_dims, src_byte_stride, dst_byte_stride, src_dt, dst_dt);
        },
        py::arg("src"), py::arg("dst"), py::arg("src_dims"), py::arg("dst_dims"),
        py::arg("src_byte_stride"), py::arg("dst_byte_stride"), py::arg("src_dt"),
        py::arg("dst_dt"));
}
