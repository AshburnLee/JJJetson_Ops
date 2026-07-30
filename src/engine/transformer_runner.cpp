#include "transformer_runner.h"

#include <stdio.h>
#include <unordered_map>
#include <vector>

#include "cublas_utils.cuh"
#include "cuda_utils.h"
#include "elementwise.h"
#include "linear.h"
#include "qkv_pack_fp16.h"
#include "rms_norm.h"
#include "rope.h"
#include "swiglu.h"

static constexpr float kTransformerRmsNormEps = 1e-6f;

// Transformer 调度器
struct TransformerRunner {
    int hidden_size = 0;
    int intermediate_size = 0;
    int head_dim = 0;
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int q_dim = 0;
    int kv_dim = 0;
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
    cublasHandle_t cublas_handle = nullptr;
    RopeCosSinCache *rope_cache = nullptr;

    float *d_w_q = nullptr;
    float *d_w_k = nullptr;
    float *d_w_v = nullptr;
    float *d_w_o = nullptr;
    float *d_w_gate = nullptr;
    float *d_w_up = nullptr;
    float *d_w_down = nullptr;
    float *d_w_input_layernorm = nullptr;
    float *d_w_post_attention_layernorm = nullptr;
    // buffer 按 shape 复用，num_tokens 直接体现 shape 的不同
    // device buffer 由Runner 管理，按num_tokens 保存、查找，找不到才 cudamalloc
    // 保存的是Device的指针，避免了 每 次forward 前的 cudaMalloc
    std::unordered_map<int, TransformerLayerLinearDeviceBuffers *> buffers_by_tokens;
    std::unordered_map<int, int *> d_pos_by_tokens;
};

static size_t transformer_col_major_bytes(int features, int num_tokens) {
    return static_cast<size_t>(features) * num_tokens * sizeof(float);
}

// cudaMalloc
static TransformerLayerLinearDeviceBuffers *
transformer_layer_linear_buffers_create(int num_tokens, int hidden_size, int q_dim, int kv_dim,
                                        int intermediate_size, int head_dim, int num_q_heads,
                                        int num_kv_heads) {
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
    CUDA_CHECK(
        cudaMalloc(&buffers->d_residual, transformer_col_major_bytes(hidden_size, num_tokens)));

    const size_t q_fp16_bytes =
        static_cast<size_t>(head_dim) * num_tokens * num_q_heads * sizeof(uint16_t);
    const size_t kv_fp16_bytes =
        static_cast<size_t>(head_dim) * num_tokens * num_kv_heads * sizeof(uint16_t);
    CUDA_CHECK(cudaMalloc(&buffers->d_q_fp16, q_fp16_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_k_fp16, kv_fp16_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_v_fp16, kv_fp16_bytes));

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
    cudaFree(buffers->d_residual);
    cudaFree(buffers->d_q_fp16);
    cudaFree(buffers->d_k_fp16);
    cudaFree(buffers->d_v_fp16);
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

// Pre-LN fused add + Linear + RoPE + Attention 占位 + FFN（Post-FFN residual add 见 40c）
extern "C" void transformer_layer_linears_forward_device(
    void *stream, void *cublas_handle, TransformerLayerLinearDeviceBuffers *buffers,
    const float *d_w_input_layernorm, const float *d_w_post_attention_layernorm, const float *d_w_q,
    const float *d_w_k, const float *d_w_v, const float *d_w_o, const float *d_w_gate,
    const float *d_w_up, const float *d_w_down, const RopeCosSinCache *rope_cache, const int *d_pos,
    int head_dim, int num_q_heads, int num_kv_heads, float rms_norm_epsilon) {
    const int H = buffers->hidden_size;
    const int T = buffers->num_tokens;
    const int Q = buffers->q_dim;
    const int KV = buffers->kv_dim;
    const int I = buffers->intermediate_size;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const size_t hidden_bytes = transformer_col_major_bytes(H, T);

    // Pre-Attn:
    /*
    调用前:
    d_hidden   = h        (层输入)
    d_residual = 0        (memset，尚无 skip)

    rms_norm_fused_add 内部:
    z = d_hidden + d_residual = h + 0 = h

    调用后（两个 buffer 都被 in-place 覆盖）:
    d_residual = z = h                    ← 存 skip，给后面 Pre-FFN 用
    d_hidden   = RMSNorm(h) * w_input      ← 供 Q/K/V Linear
    */
    CUDA_CHECK(cudaMemsetAsync(buffers->d_residual, 0, hidden_bytes, s));
    if (rms_norm_fused_add_forward_device(stream, buffers->d_hidden, buffers->d_residual,
                                          d_w_input_layernorm, H, T, rms_norm_epsilon) != 0) {
        std::fprintf(stderr, "transformer_layer: pre-attn rms_norm_fused_add failed\n");
        return;
    }
    // Linear * 3
    linear_forward_device(stream, cublas_handle, buffers->d_hidden, d_w_q, buffers->d_q, H, Q, T);
    linear_forward_device(stream, cublas_handle, buffers->d_hidden, d_w_k, buffers->d_k, H, KV, T);
    linear_forward_device(stream, cublas_handle, buffers->d_hidden, d_w_v, buffers->d_v, H, KV, T);
    // RoPE
    if (rope_cache != nullptr && d_pos != nullptr) {
        // RoPE 仅作用于 Q/K；V 不旋转。layout 与 d_q/d_k 的 [q_dim/kv_dim, T] col-major 兼容
        if (rope_neox_forward_device(stream, rope_cache, buffers->d_q, buffers->d_q, d_pos,
                                     head_dim, num_q_heads, T, 1) != 0) {
            std::fprintf(stderr, "transformer_layer: rope on Q failed\n");
            return;
        }
        if (rope_neox_forward_device(stream, rope_cache, buffers->d_k, buffers->d_k, d_pos,
                                     head_dim, num_kv_heads, T, 1) != 0) {
            std::fprintf(stderr, "transformer_layer: rope on K failed\n");
            return;
        }
    }

    /*
    Q/K/V pack（RoPE 后）:
    调用前: d_q/d_k/d_v = fp32 flat [feat_dim, T]（Linear/RoPE 输出）
    调用后: d_q_fp16/d_k_fp16/d_v_fp16 = fp16 [head_dim, T, num_heads, 1]（FA/KV layout）
    Attention 占位仍读 fp32 d_q；FA / KV cache 接入时使用 fp16 buffer。
    */
    if (qkv_pack_fp16_forward_device(stream, buffers->d_q, buffers->d_q_fp16, head_dim, T,
                                     num_q_heads) != 0) {
        std::fprintf(stderr, "transformer_layer: q pack fp16 failed\n");
        return;
    }
    if (qkv_pack_fp16_forward_device(stream, buffers->d_k, buffers->d_k_fp16, head_dim, T,
                                     num_kv_heads) != 0) {
        std::fprintf(stderr, "transformer_layer: k pack fp16 failed\n");
        return;
    }
    if (qkv_pack_fp16_forward_device(stream, buffers->d_v, buffers->d_v_fp16, head_dim, T,
                                     num_kv_heads) != 0) {
        std::fprintf(stderr, "transformer_layer: v pack fp16 failed\n");
        return;
    }

    const size_t attn_bytes = transformer_col_major_bytes(Q, T);
    // Attention 占位：暂 D2D 拷贝 d_q -> d_attn_out；后续替换为 FA + KV cache
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_attn_out, buffers->d_q, attn_bytes,
                               cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));

    linear_forward_device(stream, cublas_handle, buffers->d_attn_out, d_w_o, buffers->d_hidden_mid,
                          Q, H, T);

    // Pre-FFN:
    /*
    调用前:
    d_hidden_mid = attn_out   (O Linear 输出)
    d_residual   = h          (Pre-Attn 存下的 skip)

    rms_norm_fused_add 内部:
    z = attn_out + h          ← 等价「Post-Attn residual add」

    调用后:
    d_residual   = z = h + attn_out       ← 更新 skip（40c Post-FFN add 还会用）
    d_hidden_mid = RMSNorm(z) * w_post  ← 供 gate/up Linear
    */
    if (rms_norm_fused_add_forward_device(stream, buffers->d_hidden_mid, buffers->d_residual,
                                          d_w_post_attention_layernorm, H, T,
                                          rms_norm_epsilon) != 0) {
        std::fprintf(stderr, "transformer_layer: pre-ffn rms_norm_fused_add failed\n");
        return;
    }

    // FFN
    linear_forward_device(stream, cublas_handle, buffers->d_hidden_mid, d_w_gate, buffers->d_gate,
                          H, I, T);
    linear_forward_device(stream, cublas_handle, buffers->d_hidden_mid, d_w_up, buffers->d_up, H, I,
                          T);

    swiglu_silu_mul_launch_device(stream, buffers->d_gate, buffers->d_up, buffers->d_ffn_mid,
                                  I * T);

    linear_forward_device(stream, cublas_handle, buffers->d_ffn_mid, d_w_down,
                          buffers->d_hidden_out, I, H, T);

    // Post-FFN residual add：层输出 = d_residual + ffn_out
    // 调用前: d_residual = h + attn_out（Pre-FFN fused add 写入）, d_hidden_out = ffn_out
    // 调用后: d_hidden_out 覆盖为 d_residual + ffn_out（in-place，d_out = d_b）
    const int n_hidden_elem = H * T;
    if (elementwise_add_forward_device(stream, buffers->d_residual, buffers->d_hidden_out,
                                       buffers->d_hidden_out, n_hidden_elem) != 0) {
        std::fprintf(stderr, "transformer_layer: post-ffn elementwise_add failed\n");
        return;
    }
}

// Weight 在 Runner create时，H2D 一次，存入Runner中
extern "C" TransformerRunner *
transformer_runner_create(int hidden_size, int intermediate_size, int num_q_heads, int num_kv_heads,
                          int head_dim, int max_seq_len, float freq_base, const float *w_q_host,
                          const float *w_k_host, const float *w_v_host, const float *w_o_host,
                          const float *w_gate_host, const float *w_up_host,
                          const float *w_down_host, const float *w_input_layernorm_host,
                          const float *w_post_attention_layernorm_host, void *stream_in) {
    if (hidden_size <= 0 || intermediate_size <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || max_seq_len <= 0) {
        std::fprintf(stderr, "transformer_runner_create: invalid shape\n");
        return nullptr;
    }
    if (w_q_host == nullptr || w_k_host == nullptr || w_v_host == nullptr || w_o_host == nullptr ||
        w_gate_host == nullptr || w_up_host == nullptr || w_down_host == nullptr ||
        w_input_layernorm_host == nullptr || w_post_attention_layernorm_host == nullptr) {
        std::fprintf(stderr, "transformer_runner_create: null weight pointer\n");
        return nullptr;
    }

    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;

    auto *runner = new TransformerRunner{};
    runner->hidden_size = hidden_size;
    runner->intermediate_size = intermediate_size;
    runner->head_dim = head_dim;
    runner->num_q_heads = num_q_heads;
    runner->num_kv_heads = num_kv_heads;
    runner->q_dim = q_dim;
    runner->kv_dim = kv_dim;
    runner->owns_stream = (stream_in == nullptr);
    runner->stream = runner->owns_stream ? nullptr : static_cast<cudaStream_t>(stream_in);
    if (runner->owns_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&runner->stream, cudaStreamNonBlocking));
    }

    CUBLAS_CHECK(cublasCreate(&runner->cublas_handle));

    // model 级 RoPE cos/sin cache，create 时 H2D 一次，forward 时查表
    runner->rope_cache = rope_cossin_cache_create(max_seq_len, head_dim, freq_base);
    if (runner->rope_cache == nullptr) {
        transformer_runner_destroy(runner);
        return nullptr;
    }

    const size_t w_q_bytes = static_cast<size_t>(q_dim) * hidden_size * sizeof(float);
    const size_t w_kv_bytes = static_cast<size_t>(kv_dim) * hidden_size * sizeof(float);
    const size_t w_o_bytes = static_cast<size_t>(hidden_size) * q_dim * sizeof(float);
    const size_t w_gu_bytes = static_cast<size_t>(intermediate_size) * hidden_size * sizeof(float);
    const size_t w_d_bytes = static_cast<size_t>(hidden_size) * intermediate_size * sizeof(float);
    const size_t w_norm_bytes = static_cast<size_t>(hidden_size) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&runner->d_w_q, w_q_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_k, w_kv_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_v, w_kv_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_o, w_o_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_gate, w_gu_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_up, w_gu_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_down, w_d_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_input_layernorm, w_norm_bytes));
    CUDA_CHECK(cudaMalloc(&runner->d_w_post_attention_layernorm, w_norm_bytes));

    CUDA_CHECK(cudaMemcpy(runner->d_w_q, w_q_host, w_q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_k, w_k_host, w_kv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_v, w_v_host, w_kv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_o, w_o_host, w_o_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_gate, w_gate_host, w_gu_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_up, w_up_host, w_gu_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_down, w_down_host, w_d_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_input_layernorm, w_input_layernorm_host, w_norm_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(runner->d_w_post_attention_layernorm, w_post_attention_layernorm_host,
                          w_norm_bytes, cudaMemcpyHostToDevice));

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

    for (auto &entry : runner->d_pos_by_tokens) {
        cudaFree(entry.second);
    }
    runner->d_pos_by_tokens.clear();

    if (runner->rope_cache != nullptr) {
        rope_cossin_cache_destroy(runner->rope_cache);
        runner->rope_cache = nullptr;
    }

    cudaFree(runner->d_w_q);
    cudaFree(runner->d_w_k);
    cudaFree(runner->d_w_v);
    cudaFree(runner->d_w_o);
    cudaFree(runner->d_w_gate);
    cudaFree(runner->d_w_up);
    cudaFree(runner->d_w_down);
    cudaFree(runner->d_w_input_layernorm);
    cudaFree(runner->d_w_post_attention_layernorm);

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
        num_tokens, runner->hidden_size, runner->q_dim, runner->kv_dim, runner->intermediate_size,
        runner->head_dim, runner->num_q_heads, runner->num_kv_heads);
    runner->buffers_by_tokens[num_tokens] = buffers;
    return buffers;
}

// 测试路径辅助：为当前 step 准备 device 侧 d_pos。
// - 按 num_tokens 复用已 cudaMalloc 的 buffer (见 d_pos_by_tokens)
// - 每步将 pos[t] 写为 pos_offset + t，再 H2D 到 d_pos
//   prefill:  pos_offset=0        -> [0, 1, ..., T-1]
//   decode:   pos_offset=cache_len -> [cache_len] (num_tokens=1)
// 生产路径 forward_device 由调用方直接提供 ctx->d_pos，不经过此函数。
static int *transformer_runner_d_pos_get(TransformerRunner *runner, int num_tokens, int pos_offset,
                                         cudaStream_t stream) {
    // 按 num_tokens 缓存 d_pos buffer；每步根据 pos_offset 刷新 pos[t]=offset+t
    int *d_pos = nullptr;
    const auto it = runner->d_pos_by_tokens.find(num_tokens);
    if (it != runner->d_pos_by_tokens.end()) {
        d_pos = it->second;
    } else {
        CUDA_CHECK(cudaMalloc(&d_pos, static_cast<size_t>(num_tokens) * sizeof(int)));
        runner->d_pos_by_tokens[num_tokens] = d_pos;
    }

    std::vector<int> pos_host(static_cast<size_t>(num_tokens));
    for (int t = 0; t < num_tokens; ++t) {
        pos_host[static_cast<size_t>(t)] = pos_offset + t;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_pos, pos_host.data(),
                               static_cast<size_t>(num_tokens) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    return d_pos;
}

// 生产入口：ctx 中 d_hidden_in/out、d_pos 已在 GPU
// TODO：暴露给 Python 端，构造 ctx 后可直接测 forward_device
extern "C" int transformer_runner_forward_device(TransformerRunner *runner,
                                                 const TransformerRunnerForwardCtx *ctx) {
    if (runner == nullptr || ctx == nullptr) {
        return -1;
    }
    if (ctx->d_pos == nullptr) {
        std::fprintf(stderr, "transformer_runner_forward_device: d_pos is null\n");
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

    transformer_layer_linears_forward_device(
        stream, runner->cublas_handle, buffers, runner->d_w_input_layernorm,
        runner->d_w_post_attention_layernorm, runner->d_w_q, runner->d_w_k, runner->d_w_v,
        runner->d_w_o, runner->d_w_gate, runner->d_w_up, runner->d_w_down, runner->rope_cache,
        ctx->d_pos, runner->head_dim, runner->num_q_heads, runner->num_kv_heads,
        kTransformerRmsNormEps);
    // 计算结束后 D2D
    transformer_runner_copy_output(stream, ctx->d_hidden_out, buffers, runner->hidden_size,
                                   ctx->num_tokens);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}

// Transformer test 入口：host hidden H2D -> layer forward (含 RoPE) -> D2H
extern "C" int transformer_runner_test(TransformerRunner *runner, const float *hidden_in_host,
                                       float *hidden_out_host, int num_tokens, int pos_offset) {
    if (runner == nullptr) {
        return -1;
    }

    // 按照 num_tokens 的大小缓存 buffer，避免每次都 cudaMalloc
    TransformerLayerLinearDeviceBuffers *buffers =
        transformer_runner_buffers_get(runner, num_tokens);
    cudaStream_t stream = runner->stream;
    int *d_pos = transformer_runner_d_pos_get(runner, num_tokens, pos_offset, stream);
    const size_t hidden_bytes = transformer_col_major_bytes(runner->hidden_size, num_tokens);
    // H2D
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_hidden, hidden_in_host, hidden_bytes,
                               cudaMemcpyHostToDevice, stream));
    // Linear + RoPE + Attention 占位 + FFN
    transformer_layer_linears_forward_device(
        stream, runner->cublas_handle, buffers, runner->d_w_input_layernorm,
        runner->d_w_post_attention_layernorm, runner->d_w_q, runner->d_w_k, runner->d_w_v,
        runner->d_w_o, runner->d_w_gate, runner->d_w_up, runner->d_w_down, runner->rope_cache,
        d_pos, runner->head_dim, runner->num_q_heads, runner->num_kv_heads, kTransformerRmsNormEps);
    // D2H
    CUDA_CHECK(cudaMemcpyAsync(hidden_out_host, buffers->d_hidden_out, hidden_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}
