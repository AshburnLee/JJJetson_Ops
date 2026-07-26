#include "transformer_runner.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(transformer_runner_me, m) {
    m.doc() = "Transformer runner: Linear + RoPE + Attention placeholder + FFN on device";

    m.def(
        "create_runner",
        [](int hidden_size, int intermediate_size, int num_q_heads, int num_kv_heads, int head_dim,
           int max_seq_len, float freq_base,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_q,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_k,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_v,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_o,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_gate,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_up,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_down,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_input_layernorm,
           py::array_t<float, py::array::c_style | py::array::forcecast> w_post_attention_layernorm)
            -> uintptr_t {
            TransformerRunner *runner = transformer_runner_create(
                hidden_size, intermediate_size, num_q_heads, num_kv_heads, head_dim, max_seq_len,
                freq_base, static_cast<float *>(w_q.request().ptr),
                static_cast<float *>(w_k.request().ptr), static_cast<float *>(w_v.request().ptr),
                static_cast<float *>(w_o.request().ptr), static_cast<float *>(w_gate.request().ptr),
                static_cast<float *>(w_up.request().ptr),
                static_cast<float *>(w_down.request().ptr),
                static_cast<float *>(w_input_layernorm.request().ptr),
                static_cast<float *>(w_post_attention_layernorm.request().ptr), nullptr);
            if (runner == nullptr) {
                throw std::runtime_error("transformer_runner_create failed");
            }
            return reinterpret_cast<uintptr_t>(runner);
        },
        py::arg("hidden_size"), py::arg("intermediate_size"), py::arg("num_q_heads"),
        py::arg("num_kv_heads"), py::arg("head_dim"), py::arg("max_seq_len"),
        py::arg("freq_base") = 10000.f, py::arg("w_q"), py::arg("w_k"), py::arg("w_v"),
        py::arg("w_o"), py::arg("w_gate"), py::arg("w_up"), py::arg("w_down"),
        py::arg("w_input_layernorm"), py::arg("w_post_attention_layernorm"));

    m.def(
        "destroy_runner",
        [](uintptr_t runner_handle) {
            transformer_runner_destroy(reinterpret_cast<TransformerRunner *>(runner_handle));
        },
        py::arg("runner_handle"));

    m.def(
        "forward_host",
        [](uintptr_t runner_handle, int num_tokens, int pos_offset,
           py::array_t<float, py::array::f_style | py::array::forcecast> hidden_in,
           py::array_t<float, py::array::f_style | py::array::forcecast> hidden_out) {
            auto in_buf = hidden_in.request();
            auto out_buf = hidden_out.request();
            TransformerRunner *runner = reinterpret_cast<TransformerRunner *>(runner_handle);
            if (transformer_runner_test(runner, static_cast<float *>(in_buf.ptr),
                                        static_cast<float *>(out_buf.ptr), num_tokens,
                                        pos_offset) != 0) {
                throw std::runtime_error("transformer_runner_test failed");
            }
        },
        py::arg("runner_handle"), py::arg("num_tokens"), py::arg("pos_offset") = 0,
        py::arg("hidden_in"), py::arg("hidden_out"));
}
