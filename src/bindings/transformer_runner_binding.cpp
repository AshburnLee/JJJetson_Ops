#include "transformer_runner.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "cuda_utils.h"

namespace py = pybind11;

PYBIND11_MODULE(transformer_runner_me, m) {
    m.doc() = "Transformer runner: Pre-LN + FA Attention + FFN on device";

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
        py::arg("hidden_in"), py::arg("hidden_out"),
        "Test wrapper: H2D + auto d_pos from pos_offset + transformer_runner_test");

    m.def(
        "forward_device",
        [](uintptr_t runner_handle, int num_tokens,
           py::array_t<float, py::array::f_style | py::array::forcecast> hidden_in,
           py::array_t<float, py::array::f_style | py::array::forcecast> hidden_out,
           py::array_t<int, py::array::c_style | py::array::forcecast> pos) {
            auto in_buf = hidden_in.request();
            auto out_buf = hidden_out.request();
            auto pos_buf = pos.request();

            if (pos_buf.ndim != 1 || static_cast<int>(pos_buf.shape[0]) != num_tokens) {
                throw std::invalid_argument("pos must be 1-D with length num_tokens");
            }
            if (static_cast<int>(in_buf.shape[1]) != num_tokens ||
                static_cast<int>(out_buf.shape[1]) != num_tokens) {
                throw std::invalid_argument("hidden_in/out num_tokens mismatch");
            }
            if (in_buf.shape[0] != out_buf.shape[0]) {
                throw std::invalid_argument("hidden_in/out hidden_size mismatch");
            }

            const size_t hidden_bytes = static_cast<size_t>(in_buf.shape[0]) *
                                        static_cast<size_t>(num_tokens) * sizeof(float);
            const size_t pos_bytes = static_cast<size_t>(num_tokens) * sizeof(int);

            float *d_hidden_in = nullptr;
            float *d_hidden_out = nullptr;
            int *d_pos = nullptr;
            // cudaMalloc 是测试开销，生产路径上 hidden/pos 已在 GPU 上，不会cudaMalloc
            CUDA_CHECK(cudaMalloc(&d_hidden_in, hidden_bytes));
            CUDA_CHECK(cudaMalloc(&d_hidden_out, hidden_bytes));
            CUDA_CHECK(cudaMalloc(&d_pos, pos_bytes));

            cudaStream_t stream = nullptr;
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

            CUDA_CHECK(cudaMemcpyAsync(d_hidden_in, in_buf.ptr, hidden_bytes,
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(
                cudaMemcpyAsync(d_pos, pos_buf.ptr, pos_bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // 在python调用一次该API，就要重新创建ctx ，这是数据pack
            TransformerRunnerForwardCtx ctx{};
            ctx.num_tokens = num_tokens;
            ctx.stream = nullptr;
            ctx.d_hidden_in = d_hidden_in;
            ctx.d_hidden_out = d_hidden_out;
            ctx.d_pos = d_pos;
            // runner 生产上，整个session 只创建一次
            TransformerRunner *runner = reinterpret_cast<TransformerRunner *>(runner_handle);
            const int rc = transformer_runner_forward_device(runner, &ctx);

            if (rc == 0) {
                CUDA_CHECK(
                    cudaMemcpy(out_buf.ptr, d_hidden_out, hidden_bytes, cudaMemcpyDeviceToHost));
            }

            cudaFree(d_hidden_in);
            cudaFree(d_hidden_out);
            cudaFree(d_pos);
            CUDA_CHECK(cudaStreamDestroy(stream));

            if (rc != 0) {
                throw std::runtime_error("transformer_runner_forward_device failed");
            }
        },
        py::arg("runner_handle"), py::arg("num_tokens"), py::arg("hidden_in"),
        py::arg("hidden_out"), py::arg("pos"),
        "Production path: H2D I/O wrapper around transformer_runner_forward_device (caller "
        "supplies absolute d_pos)");

    m.def(
        "kv_cache_len",
        [](uintptr_t runner_handle) {
            return transformer_runner_kv_cache_len(
                reinterpret_cast<TransformerRunner *>(runner_handle));
        },
        py::arg("runner_handle"), "Current KV cache length after forward + advance_len");

    m.def(
        "kv_cache_reset",
        [](uintptr_t runner_handle) {
            transformer_runner_kv_cache_reset(reinterpret_cast<TransformerRunner *>(runner_handle));
        },
        py::arg("runner_handle"),
        "Reset session KV cache (cache_len=0); reuse GPU buffers for a new request, No destroy");
}
