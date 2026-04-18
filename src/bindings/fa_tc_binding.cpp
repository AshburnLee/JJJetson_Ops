#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" void fa_one_pass_parallel_tc(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* dst_host,
    float scale);

extern "C" void fa_one_pass_parallel_tc_true(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* dst_host,
    float scale);

extern "C" void fa_one_pass_parallel_tc_true_64(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* dst_host,
    float scale);

void fa_one_pass_parallel_tc_py(
    py::buffer q, py::buffer k, py::buffer v, py::array_t<float> dst, float scale) {

    auto q_buf = q.request();
    auto k_buf = k.request();
    auto v_buf = v.request();
    auto dst_buf = dst.request();

    auto* q_ptr = static_cast<const uint16_t*>(q_buf.ptr);
    auto* k_ptr = static_cast<const uint16_t*>(k_buf.ptr);
    auto* v_ptr = static_cast<const uint16_t*>(v_buf.ptr);
    auto* dst_ptr = static_cast<float*>(dst_buf.ptr);

    fa_one_pass_parallel_tc(q_ptr, k_ptr, v_ptr, dst_ptr, scale);
}

void fa_one_pass_parallel_tc_true_py(
    py::buffer q, py::buffer k, py::buffer v, py::array_t<float> dst, float scale) {

    auto q_buf = q.request();
    auto k_buf = k.request();
    auto v_buf = v.request();
    auto dst_buf = dst.request();

    auto* q_ptr = static_cast<const uint16_t*>(q_buf.ptr);
    auto* k_ptr = static_cast<const uint16_t*>(k_buf.ptr);
    auto* v_ptr = static_cast<const uint16_t*>(v_buf.ptr);
    auto* dst_ptr = static_cast<float*>(dst_buf.ptr);

    fa_one_pass_parallel_tc_true(q_ptr, k_ptr, v_ptr, dst_ptr, scale);
}

void fa_one_pass_parallel_tc_true_64_py(
    py::buffer q, py::buffer k, py::buffer v, py::array_t<float> dst, float scale) {

    auto q_buf = q.request();
    auto k_buf = k.request();
    auto v_buf = v.request();
    auto dst_buf = dst.request();

    auto* q_ptr = static_cast<const uint16_t*>(q_buf.ptr);
    auto* k_ptr = static_cast<const uint16_t*>(k_buf.ptr);
    auto* v_ptr = static_cast<const uint16_t*>(v_buf.ptr);
    auto* dst_ptr = static_cast<float*>(dst_buf.ptr);

    fa_one_pass_parallel_tc_true_64(q_ptr, k_ptr, v_ptr, dst_ptr, scale);
}

PYBIND11_MODULE(fa_tc_me, m) {
    m.doc() = "Fused attention one-pass parallel: launch_fa_one_pass_parallel_tc (WMMA); "
              "launch_fa_one_pass_parallel_tc_true; launch_fa_one_pass_parallel_tc_true_64 (KV tile 64, opt-in smem)";
    m.def(
        "launch_fa_one_pass_parallel_tc",
        &fa_one_pass_parallel_tc_py,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("dst"),
        py::arg("scale"));
    m.def(
        "launch_fa_one_pass_parallel_tc_true",
        &fa_one_pass_parallel_tc_true_py,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("dst"),
        py::arg("scale"));
    m.def(
        "launch_fa_one_pass_parallel_tc_true_64",
        &fa_one_pass_parallel_tc_true_64_py,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("dst"),
        py::arg("scale"));
}
