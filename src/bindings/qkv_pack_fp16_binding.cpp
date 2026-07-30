#include "qkv_pack_fp16.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(qkv_pack_fp16_me, m) {
    m.doc() = "Python binding for qkv_pack_fp16 (Linear flat fp32 -> FA layout fp16)";

    m.def(
        "forward_host",
        [](py::array_t<float, py::array::f_style> src,
           py::array_t<uint16_t, py::array::f_style> dst, int head_dim, int num_tokens,
           int num_heads) {
            const py::buffer_info src_info = src.request();
            const py::buffer_info dst_info = dst.request();
            const int feat_dim = head_dim * num_heads;
            const size_t expected_src = static_cast<size_t>(feat_dim) * num_tokens;
            const size_t expected_dst = static_cast<size_t>(head_dim) * num_tokens * num_heads;
            if (static_cast<size_t>(src_info.size) != expected_src) {
                throw std::invalid_argument("src size mismatch for head_dim/num_tokens/num_heads");
            }
            if (static_cast<size_t>(dst_info.size) != expected_dst) {
                throw std::invalid_argument("dst size mismatch for head_dim/num_tokens/num_heads");
            }
            qkv_pack_fp16_forward_host(static_cast<const float *>(src_info.ptr),
                                       static_cast<uint16_t *>(dst_info.ptr), head_dim, num_tokens,
                                       num_heads);
        },
        py::arg("src"), py::arg("dst"), py::arg("head_dim"), py::arg("num_tokens"),
        py::arg("num_heads"), "Test wrapper: host I/O around qkv_pack_fp16_forward_device");
}
