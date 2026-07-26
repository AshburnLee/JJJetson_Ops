#include "elementwise.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

static ElementwiseBinaryOp parse_op(const std::string &op_name) {
    static const std::unordered_map<std::string, ElementwiseBinaryOp> kOps = {
        {"add", ELEMENTWISE_ADD},
        {"sub", ELEMENTWISE_SUB},
        {"mul", ELEMENTWISE_MUL},
        {"div", ELEMENTWISE_DIV},
    };
    const auto it = kOps.find(op_name);
    if (it == kOps.end()) {
        throw std::invalid_argument("unknown elementwise op: " + op_name);
    }
    return it->second;
}

PYBIND11_MODULE(elementwise_me, m) {
    m.doc() = "Python binding for CUDA elementwise binary kernels";

    py::enum_<ElementwiseBinaryOp>(m, "Op")
        .value("ADD", ELEMENTWISE_ADD)
        .value("SUB", ELEMENTWISE_SUB)
        .value("MUL", ELEMENTWISE_MUL)
        .value("DIV", ELEMENTWISE_DIV);

    m.def(
        "forward_host",
        [](const std::string &op, py::array_t<float> a, py::array_t<float> b,
           py::array_t<float> out) {
            const py::buffer_info a_info = a.request();
            const py::buffer_info b_info = b.request();
            const py::buffer_info out_info = out.request();
            if (a_info.size != b_info.size || a_info.size != out_info.size) {
                throw std::invalid_argument("a, b, out must have the same number of elements");
            }
            const float *a_ptr = static_cast<const float *>(a_info.ptr);
            const float *b_ptr = static_cast<const float *>(b_info.ptr);
            float *out_ptr = static_cast<float *>(out_info.ptr);
            elementwise_binary_forward_host(a_ptr, b_ptr, out_ptr, static_cast<int>(a_info.size),
                                            parse_op(op));
        },
        py::arg("op"), py::arg("a"), py::arg("b"), py::arg("out"),
        "Test wrapper: host I/O around elementwise_binary_forward_device (op: add/sub/mul/div)");
}
