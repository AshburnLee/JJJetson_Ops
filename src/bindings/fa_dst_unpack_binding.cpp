#include "fa_dst_unpack.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(fa_dst_unpack_me, m) {
    m.doc() = "Unpack FA dst layout to Linear flat (fp32)";
    m.def(
        "forward_host",
        [](py::array_t<float> src, py::array_t<float> dst, int head_dim, int num_tokens,
           int num_heads) {
            const float *src_ptr = static_cast<const float *>(src.request().ptr);
            float *dst_ptr = static_cast<float *>(dst.request().ptr);
            fa_dst_unpack_forward_host(src_ptr, dst_ptr, head_dim, num_tokens, num_heads);
        },
        py::arg("src"), py::arg("dst"), py::arg("head_dim"), py::arg("num_tokens"),
        py::arg("num_heads"), "Test wrapper: host I/O around fa_dst_unpack_forward_device");
}
