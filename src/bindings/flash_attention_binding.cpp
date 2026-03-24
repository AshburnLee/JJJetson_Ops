
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // std::vector<int>

namespace py = pybind11;

// 在 flash_attention.cu 中实现的 C 接口（负责 cudaMalloc/cudaMemcpy 和 kernel launch）
// 约定：Q/K/V 在 Python 端使用 np.float16（列主序，首维度最快），dst 使用 np.float32。
extern "C" void flash_attention(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* dst_host,
    float scale);

extern "C" void flash_attention_debug(
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

// Python 友好的包装函数
void flash_attention_py(
    py::buffer q,         // 期望 dtype=np.float16，列主序
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

    flash_attention(
        q_ptr,
        k_ptr,
        v_ptr,
        dst_ptr,
        scale);
}

void flash_attention_debug_py(
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

    flash_attention_debug(q_ptr, k_ptr, v_ptr, dst_ptr, scale,
                             m_ptr, l_ptr, s_ptr, rs_ptr, so_ptr, sn_ptr, ev_ptr);
}

PYBIND11_MODULE(flash_attention_me, m) {
    m.doc() = "Python binding for CUDA Flash Attention";
    m.def(
        "launch_flash_attention",
        &flash_attention_py,
        "Launch Flash Attention kernel",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("dst"),
        py::arg("scale"));

    m.def(
        "launch_flash_attention_debug_ml",
        &flash_attention_debug_py,
        "Launch Flash Attention kernel and dump m/l per block",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("dst"),
        py::arg("scale"),
        py::arg("m_out"),
        py::arg("l_out"),
        py::arg("s_out"),
        py::arg("row_sum_out"),
        py::arg("scale_old_out"),
        py::arg("scale_new_out"),
        py::arg("exp_val_out"));
}

