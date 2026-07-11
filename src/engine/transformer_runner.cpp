#include "transformer_runner.h"

#include <stdio.h>
#include <unordered_map>

#include "cublas_utils.cuh"
#include "cuda_utils.h"
#include "linear.h"
#include "swiglu.h"

// Transformer 调度器
struct TransformerRunner {
    int hidden_size = 0;
    int intermediate_size = 0;
    int q_dim = 0;
    int kv_dim = 0;
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
    cublasHandle_t cublas_handle = nullptr;

    float *d_w_q = nullptr;
    float *d_w_k = nullptr;
    float *d_w_v = nullptr;
    float *d_w_o = nullptr;
    float *d_w_gate = nullptr;
    float *d_w_up = nullptr;
    float *d_w_down = nullptr;
    // buffer 按 shape 复用，num_tokens 直接体现 shape 的不同
    // device buffer 由Runner 管理，按num_tokens 保存、查找，找不到才 cudamalloc
    // 保存的是Device的指针，避免了 每 次forward 前的 cudaMalloc
    std::unordered_map<int, TransformerLayerLinearDeviceBuffers *> buffers_by_tokens;
};

static size_t transformer_col_major_bytes(int features, int num_tokens) {
    return static_cast<size_t>(features) * num_tokens * sizeof(float);
}

// cudaMalloc
static TransformerLayerLinearDeviceBuffers *
transformer_layer_linear_buffers_create(int num_tokens, int hidden_size, int q_dim, int kv_dim,
                                        int intermediate_size) {
    auto *buffers = new TransformerLayerLinearDeviceBuffers{};
    buffers->num_tokens = num_tokens;
    buffers->hidden_size = hidden_size;
    buffers->q_dim = q_dim;
    buffers->kv_dim = kv_dim;
    buffers->intermediate_size = intermediate_size;

    CUDA_CHECK(
        cudaMalloc(&buffers->d_hidden, transformer_col_major_bytes(hidden_size, num_tokens)));
    CUDA_CHECK(cudaMalloc(&buffers->d_q, transformer_col_major_bytes(q_dim, num_tokens)));
    CUDA_CHECK(cudaMalloc(&buffers->d_k, transformer_col_major_bytes(kv_dim, num_tokens)));
    CUDA_CHECK(cudaMalloc(&buffers->d_v, transformer_col_major_bytes(kv_dim, num_tokens)));
    CUDA_CHECK(cudaMalloc(&buffers->d_attn_out, transformer_col_major_bytes(q_dim, num_tokens)));
    CUDA_CHECK(
        cudaMalloc(&buffers->d_hidden_mid, transformer_col_major_bytes(hidden_size, num_tokens)));
    CUDA_CHECK(
        cudaMalloc(&buffers->d_gate, transformer_col_major_bytes(intermediate_size, num_tokens)));
    CUDA_CHECK(
        cudaMalloc(&buffers->d_up, transformer_col_major_bytes(intermediate_size, num_tokens)));
    CUDA_CHECK(cudaMalloc(&buffers->d_ffn_mid,
                          transformer_col_major_bytes(intermediate_size, num_tokens)));
    CUDA_CHECK(
        cudaMalloc(&buffers->d_hidden_out, transformer_col_major_bytes(hidden_size, num_tokens)));

    return buffers;
}

static void transformer_layer_linear_buffers_destroy(TransformerLayerLinearDeviceBuffers *buffers) {
    if (buffers == nullptr) {
        return;
    }

    cudaFree(buffers->d_hidden);
    cudaFree(buffers->d_q);
    cudaFree(buffers->d_k);
    cudaFree(buffers->d_v);
    cudaFree(buffers->d_attn_out);
    cudaFree(buffers->d_hidden_mid);
    cudaFree(buffers->d_gate);
    cudaFree(buffers->d_up);
    cudaFree(buffers->d_ffn_mid);
    cudaFree(buffers->d_hidden_out);
    delete buffers;
}

// D2D, 7 GEMM chain, D2D out，无 H2D/D2H
static void transformer_runner_copy_input(cudaStream_t stream,
                                          TransformerLayerLinearDeviceBuffers *buffers,
                                          const float *d_hidden_in, int hidden_size,
                                          int num_tokens) {
    const size_t bytes = transformer_col_major_bytes(hidden_size, num_tokens);
    if (d_hidden_in != buffers->d_hidden) {
        CUDA_CHECK(cudaMemcpyAsync(buffers->d_hidden, d_hidden_in, bytes, cudaMemcpyDeviceToDevice,
                                   stream));
    }
}

static void transformer_runner_copy_output(cudaStream_t stream, float *d_hidden_out,
                                           TransformerLayerLinearDeviceBuffers *buffers,
                                           int hidden_size, int num_tokens) {
    const size_t bytes = transformer_col_major_bytes(hidden_size, num_tokens);
    if (d_hidden_out != buffers->d_hidden_out) {
        CUDA_CHECK(cudaMemcpyAsync(d_hidden_out, buffers->d_hidden_out, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

// 生产 Transformer pipline (7 GEMM all on Device)
extern "C" void transformer_layer_linears_forward_device(
    void *stream, void *cublas_handle, TransformerLayerLinearDeviceBuffers *buffers,
    const float *d_w_q, const float *d_w_k, const float *d_w_v, const float *d_w_o,
    const float *d_w_gate, const float *d_w_up, const float *d_w_down) {
    const int H = buffers->hidden_size;
    const int T = buffers->num_tokens;
    const int Q = buffers->q_dim;
    const int KV = buffers->kv_dim;
    const int I = buffers->intermediate_size;

    linear_forward_device(stream, cublas_handle, buffers->d_hidden, d_w_q, buffers->d_q, H, Q, T);
    linear_forward_device(stream, cublas_handle, buffers->d_hidden, d_w_k, buffers->d_k, H, KV, T);
    linear_forward_device(stream, cublas_handle, buffers->d_hidden, d_w_v, buffers->d_v, H, KV, T);

    const size_t attn_bytes = transformer_col_major_bytes(Q, T);
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_attn_out, buffers->d_q, attn_bytes,
                               cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));

    linear_forward_device(stream, cublas_handle, buffers->d_attn_out, d_w_o, buffers->d_hidden_mid,
                          Q, H, T);
    linear_forward_device(stream, cublas_handle, buffers->d_hidden_mid, d_w_gate, buffers->d_gate,
                          H, I, T);
    linear_forward_device(stream, cublas_handle, buffers->d_hidden_mid, d_w_up, buffers->d_up, H, I,
                          T);

    swiglu_silu_mul_launch_device(stream, buffers->d_gate, buffers->d_up, buffers->d_ffn_mid,
                                  I * T);

    linear_forward_device(stream, cublas_handle, buffers->d_ffn_mid, d_w_down,
                          buffers->d_hidden_out, I, H, T);
}

// Weight 在 Runner create时，H2D 一次，存入Runner中
extern "C" TransformerRunner *
transformer_runner_create(int hidden_size, int intermediate_size, int num_q_heads, int num_kv_heads,
                          int head_dim, const float *w_q_host, const float *w_k_host,
                          const float *w_v_host, const float *w_o_host, const float *w_gate_host,
                          const float *w_up_host, const float *w_down_host, void *stream_in) {
    if (hidden_size <= 0 || intermediate_size <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0) {
        std::fprintf(stderr, "transformer_runner_create: invalid shape\n");
        return nullptr;
    }

    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;

    auto *runner = new TransformerRunner{};
    runner->hidden_size = hidden_size;
    runner->intermediate_size = intermediate_size;
    runner->q_dim = q_dim;
    runner->kv_dim = kv_dim;
    runner->owns_stream = (stream_in == nullptr);
    runner->stream = runner->owns_stream ? nullptr : static_cast<cudaStream_t>(stream_in);
    if (runner->owns_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&runner->stream, cudaStreamNonBlocking));
    }

    CUBLAS_CHECK(cublasCreate(&runner->cublas_handle));

    const size_t w_q_bytes = static_cast<size_t>(q_dim) * hidden_size * sizeof(float);
    const size_t w_kv_bytes = static_cast<size_t>(kv_dim) * hidden_size * sizeof(float);
    const size_t w_o_bytes = static_cast<size_t>(hidden_size) * q_dim * sizeof(float);
    const size_t w_gu_bytes = static_cast<size_t>(intermediate_size) * hidden_size * sizeof(float);
    const size_t w_d_bytes = static_cast<size_t>(hidden_size) * intermediate_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&runner->d_w_q, w_q_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_k, w_kv_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_v, w_kv_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_o, w_o_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_gate, w_gu_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_up, w_gu_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_down, w_d_bytes));

    CUDA_CHECK(cudaMemcpy(runner->d_w_q, w_q_host, w_q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_k, w_k_host, w_kv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_v, w_v_host, w_kv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_o, w_o_host, w_o_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_gate, w_gate_host, w_gu_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_up, w_up_host, w_gu_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_down, w_down_host, w_d_bytes, cudaMemcpyHostToDevice));

    return runner;
}

// 负责 cudaFree 释放资源
extern "C" void transformer_runner_destroy(TransformerRunner *runner) {
    if (runner == nullptr) {
        return;
    }

    for (auto &entry : runner->buffers_by_tokens) {
        transformer_layer_linear_buffers_destroy(entry.second);
    }
    runner->buffers_by_tokens.clear();

    cudaFree(runner->d_w_q);
    cudaFree(runner->d_w_k);
    cudaFree(runner->d_w_v);
    cudaFree(runner->d_w_o);
    cudaFree(runner->d_w_gate);
    cudaFree(runner->d_w_up);
    cudaFree(runner->d_w_down);

    if (runner->cublas_handle != nullptr) {
        CUBLAS_CHECK(cublasDestroy(runner->cublas_handle));
    }
    if (runner->owns_stream && runner->stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(runner->stream));
    }

    delete runner;
}

// 按照 num_tokens 的大小 缓存不同的 buffer，避免每次都cudaMalloc
// 只有遇到未见过的 num_tokens 时，才会 cudaMalloc
extern "C" TransformerLayerLinearDeviceBuffers *
transformer_runner_buffers_get(TransformerRunner *runner, int num_tokens) {
    const auto it = runner->buffers_by_tokens.find(num_tokens);
    if (it != runner->buffers_by_tokens.end()) {
        return it->second;
    }

    TransformerLayerLinearDeviceBuffers *buffers = transformer_layer_linear_buffers_create(
        num_tokens, runner->hidden_size, runner->q_dim, runner->kv_dim, runner->intermediate_size);
    runner->buffers_by_tokens[num_tokens] = buffers;
    return buffers;
}

// 生产入口，为暴露给 python 端，暂不能测试到，ctx 暂时也用不到
// TODO：接入生产链路，构造 ctx
extern "C" int transformer_runner_forward_device(TransformerRunner *runner,
                                                 const TransformerRunnerForwardCtx *ctx) {
    if (runner == nullptr || ctx == nullptr) {
        return -1;
    }
    // 获取 Device buffer
    TransformerLayerLinearDeviceBuffers *buffers =
        transformer_runner_buffers_get(runner, ctx->num_tokens);
    cudaStream_t stream =
        ctx->stream != nullptr ? static_cast<cudaStream_t>(ctx->stream) : runner->stream;
    // 向 Device buffer D2D
    transformer_runner_copy_input(stream, buffers, ctx->d_hidden_in, runner->hidden_size,
                                  ctx->num_tokens);

    transformer_layer_linears_forward_device(stream, runner->cublas_handle, buffers, runner->d_w_q,
                                             runner->d_w_k, runner->d_w_v, runner->d_w_o,
                                             runner->d_w_gate, runner->d_w_up, runner->d_w_down);
    // 计算结束后 D2D
    transformer_runner_copy_output(stream, ctx->d_hidden_out, buffers, runner->hidden_size,
                                   ctx->num_tokens);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}

// Transformer test 入口
extern "C" int transformer_runner_test(TransformerRunner *runner, const float *hidden_in_host,
                                       float *hidden_out_host, int num_tokens) {
    if (runner == nullptr) {
        return -1;
    }

    // 按照 num_tokens 的大小 缓存不同的 buffer，避免每次都cudaMalloc
    TransformerLayerLinearDeviceBuffers *buffers =
        transformer_runner_buffers_get(runner, num_tokens);
    cudaStream_t stream = runner->stream;
    const size_t hidden_bytes = transformer_col_major_bytes(runner->hidden_size, num_tokens);
    // H2D
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_hidden, hidden_in_host, hidden_bytes,
                               cudaMemcpyHostToDevice, stream));
    // Linear * 7
    transformer_layer_linears_forward_device(stream, runner->cublas_handle, buffers, runner->d_w_q,
                                             runner->d_w_k, runner->d_w_v, runner->d_w_o,
                                             runner->d_w_gate, runner->d_w_up, runner->d_w_down);
    // D2H
    CUDA_CHECK(cudaMemcpyAsync(hidden_out_host, buffers->d_hidden_out, hidden_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}
