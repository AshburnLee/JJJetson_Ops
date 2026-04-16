
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

extern "C" void fa_two_pass(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* dst_host,
    float scale);

extern "C" void fa_one_pass(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* dst_host,
    float scale);

extern "C" void fa_one_pass_parallel(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* dst_host,
    float scale);

extern "C" void fa(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* dst_host,
    float scale);

#if defined(MY_OPS_DEBUG)
extern "C" void fa_debug(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* dst_host,
    float scale,
    float* m_host,
    float* l_host,
    float* s_host,
    float* row_sum_host,
    float* scale_old_host,
    float* scale_new_host,
    float* exp_val_host);
#endif

namespace {

void fa_launch_impl(
    void (*fn)(const uint16_t*, const uint16_t*, const uint16_t*, float*, float),
    py::buffer q,
    py::buffer k,
    py::buffer v,
    py::array_t<float> dst,
    float scale) {

    auto q_buf   = q.request();
    auto k_buf   = k.request();
    auto v_buf   = v.request();
    auto dst_buf = dst.request();

    auto* q_ptr   = static_cast<const uint16_t*>(q_buf.ptr);
    auto* k_ptr   = static_cast<const uint16_t*>(k_buf.ptr);
    auto* v_ptr   = static_cast<const uint16_t*>(v_buf.ptr);
    auto* dst_ptr = static_cast<float*>(dst_buf.ptr);

    fn(q_ptr, k_ptr, v_ptr, dst_ptr, scale);
}

}  // namespace

void fa_two_pass_py(py::buffer q, py::buffer k, py::buffer v, py::array_t<float> dst, float scale) {
    fa_launch_impl(fa_two_pass, q, k, v, dst, scale);
}

void fa_one_pass_py(py::buffer q, py::buffer k, py::buffer v, py::array_t<float> dst, float scale) {
    fa_launch_impl(fa_one_pass, q, k, v, dst, scale);
}

void fa_one_pass_parallel_py(py::buffer q, py::buffer k, py::buffer v, py::array_t<float> dst, float scale) {
    fa_launch_impl(fa_one_pass_parallel, q, k, v, dst, scale);
}

void fa_py(py::buffer q, py::buffer k, py::buffer v, py::array_t<float> dst, float scale) {
    fa_launch_impl(fa, q, k, v, dst, scale);
}

#if defined(MY_OPS_DEBUG)
void fa_debug_py(
    py::buffer q,
    py::buffer k,
    py::buffer v,
    py::array_t<float> dst,
    float scale,
    py::array_t<float> m_out,
    py::array_t<float> l_out,
    py::array_t<float> s_out,
    py::array_t<float> row_sum_out,
    py::array_t<float> scale_old_out,
    py::array_t<float> scale_new_out,
    py::array_t<float> exp_val_out) {

    auto q_buf   = q.request();
    auto k_buf   = k.request();
    auto v_buf   = v.request();
    auto dst_buf = dst.request();
    auto m_buf   = m_out.request();
    auto l_buf   = l_out.request();
    auto s_buf   = s_out.request();
    auto rs_buf  = row_sum_out.request();
    auto so_buf  = scale_old_out.request();
    auto sn_buf  = scale_new_out.request();
    auto ev_buf  = exp_val_out.request();

    auto* q_ptr   = static_cast<const uint16_t*>(q_buf.ptr);
    auto* k_ptr   = static_cast<const uint16_t*>(k_buf.ptr);
    auto* v_ptr   = static_cast<const uint16_t*>(v_buf.ptr);
    auto* dst_ptr = static_cast<float*>(dst_buf.ptr);
    auto* m_ptr   = static_cast<float*>(m_buf.ptr);
    auto* l_ptr   = static_cast<float*>(l_buf.ptr);
    auto* s_ptr   = static_cast<float*>(s_buf.ptr);
    auto* rs_ptr  = static_cast<float*>(rs_buf.ptr);
    auto* so_ptr  = static_cast<float*>(so_buf.ptr);
    auto* sn_ptr  = static_cast<float*>(sn_buf.ptr);
    auto* ev_ptr  = static_cast<float*>(ev_buf.ptr);

    fa_debug(q_ptr, k_ptr, v_ptr, dst_ptr, scale,
             m_ptr, l_ptr, s_ptr, rs_ptr, so_ptr, sn_ptr, ev_ptr);
}
#endif

PYBIND11_MODULE(fa_me, m) {
    m.doc() = "CUDA fused attention (fa): two-pass / one-pass / one-pass parallel";
    m.def("launch_fa_two_pass", &fa_two_pass_py, "Two-pass KV exact attention", py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("dst"), py::arg("scale"));
    m.def("launch_fa_one_pass", &fa_one_pass_py, "Fused streaming, 8 blocks (2 Q-heads/block)", py::arg("q"),
          py::arg("k"), py::arg("v"), py::arg("dst"), py::arg("scale"));
    m.def("launch_fa_one_pass_parallel", &fa_one_pass_parallel_py, "Fused streaming, 16 blocks", py::arg("q"),
          py::arg("k"), py::arg("v"), py::arg("dst"), py::arg("scale"));
    m.def("launch_fa", &fa_py, "Same as launch_fa_one_pass_parallel", py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("dst"), py::arg("scale"));

#if defined(MY_OPS_DEBUG)
    m.def("launch_fa_debug_ml", &fa_debug_py, "One-pass kernel with m/l/S dumps (debug build)", py::arg("q"),
          py::arg("k"), py::arg("v"), py::arg("dst"), py::arg("scale"), py::arg("m_out"), py::arg("l_out"),
          py::arg("s_out"), py::arg("row_sum_out"), py::arg("scale_old_out"), py::arg("scale_new_out"),
          py::arg("exp_val_out"));
#endif
}
