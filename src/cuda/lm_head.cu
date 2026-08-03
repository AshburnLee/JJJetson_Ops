#include "lm_head.h"

#include <cuda_runtime.h>
#include <cstdio>

#include <cublas_utils.cuh>
#include <cuda_utils.cuh>

// lm_head：hidden → logits（词表投影）。实现为 cuBLAS SGEMM，无自定义 kernel。
//
// untied 权重 d_lm_head row-major [hidden_size, vocab_size]，存 lm_head[h, v]。
// 对每个 token t：
//   logits[v, t] = sum_h lm_head[h, v] * hidden[h, t]
// 矩阵形式（每列一个 token）：Logits[:,t] = lm_head^T @ hidden[:,t]
// 即 Logits [vocab, T] = lm_head^T [vocab, hidden] @ Hidden [hidden, T]
//
// tied 时 d_lm_head == d_embed，embed row-major [vocab, hidden]：
//   logits[v, t] = sum_h embed[v, h] * hidden[h, t]
// 即 Logits [vocab, T] = Embed [vocab, hidden] @ Hidden [hidden, T]

extern "C" void untied_lm_head_forward_device(void *stream, void *cublas_handle,
                                              const float *d_lm_head, const float *d_hidden,
                                              float *d_logits, int hidden_size, int vocab_size,
                                              int num_tokens) {
    if (hidden_size <= 0 || vocab_size <= 0 || num_tokens <= 0 || d_lm_head == nullptr ||
        d_hidden == nullptr || d_logits == nullptr) {
        std::fprintf(stderr,
                     "untied_lm_head_forward_device: invalid args hidden=%d vocab=%d tokens=%d\n",
                     hidden_size, vocab_size, num_tokens);
        return;
    }

    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle);
    CUBLAS_CHECK(cublasSetStream(handle, s));

    const float alpha = 1.f;
    const float beta = 0.f;
    // C[vocab, T] = A[vocab, hidden] @ B[hidden, T]
    // A 来自 d_lm_head row-major [hidden, vocab]：内存上 A(v,h)=lm_head[h,v] 位于 h*vocab+v，
    // 等价于 cuBLAS 列主序 [vocab x hidden]、lda=vocab → OP_N。
    // B 为 hidden col-major [hidden, T]，lda=hidden。
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, vocab_size, num_tokens, hidden_size,
                             &alpha, d_lm_head, vocab_size, d_hidden, hidden_size, &beta, d_logits,
                             vocab_size));
}

extern "C" void tied_lm_head_forward_device(void *stream, void *cublas_handle, const float *d_embed,
                                            const float *d_hidden, float *d_logits, int hidden_size,
                                            int vocab_size, int num_tokens) {
    if (hidden_size <= 0 || vocab_size <= 0 || num_tokens <= 0 || d_embed == nullptr ||
        d_hidden == nullptr || d_logits == nullptr) {
        std::fprintf(stderr,
                     "tied_lm_head_forward_device: invalid args hidden=%d vocab=%d tokens=%d\n",
                     hidden_size, vocab_size, num_tokens);
        return;
    }

    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle);
    CUBLAS_CHECK(cublasSetStream(handle, s));

    const float alpha = 1.f;
    const float beta = 0.f;
    // C[vocab, T] = Embed[vocab, hidden] @ Hidden[hidden, T]
    // d_embed row-major [vocab, hidden]：列主序视角为 [hidden x vocab]、lda=hidden，OP_T 得 [vocab
    // x hidden]。
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, vocab_size, num_tokens, hidden_size,
                             &alpha, d_embed, hidden_size, d_hidden, hidden_size, &beta, d_logits,
                             vocab_size));
}

extern "C" void tied_lm_head_forward_host(const float *embed_host, const float *hidden_host,
                                          float *logits_host, int hidden_size, int vocab_size,
                                          int num_tokens) {
    if (hidden_size <= 0 || vocab_size <= 0 || num_tokens <= 0) {
        return;
    }
    if (embed_host == nullptr || hidden_host == nullptr || logits_host == nullptr) {
        std::fprintf(stderr, "tied_lm_head_forward_host: null pointer argument\n");
        return;
    }

    const int64_t n_hidden = static_cast<int64_t>(hidden_size) * num_tokens;
    const int64_t n_logits = static_cast<int64_t>(vocab_size) * num_tokens;
    const int64_t n_embed = static_cast<int64_t>(vocab_size) * hidden_size;

    float *d_embed = nullptr;
    float *d_hidden = nullptr;
    float *d_logits = nullptr;
    cudaStream_t stream;
    cublasHandle_t handle;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaMallocAsync(&d_embed, static_cast<size_t>(n_embed) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_hidden, static_cast<size_t>(n_hidden) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_logits, static_cast<size_t>(n_logits) * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_embed, embed_host, static_cast<size_t>(n_embed) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_hidden, hidden_host, static_cast<size_t>(n_hidden) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    tied_lm_head_forward_device(stream, handle, d_embed, d_hidden, d_logits, hidden_size,
                                vocab_size, num_tokens);

    CUDA_CHECK(cudaMemcpyAsync(logits_host, d_logits, static_cast<size_t>(n_logits) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_embed, stream));
    CUDA_CHECK(cudaFreeAsync(d_hidden, stream));
    CUDA_CHECK(cudaFreeAsync(d_logits, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(handle));
}
