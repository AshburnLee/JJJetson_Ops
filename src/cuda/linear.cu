#include "linear.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cublas_utils.cuh>
#include <cuda_utils.cuh>

// 单 GEMM on device 接口
extern "C" void linear_forward_device(void *stream, void *cublas_handle, const float *input,
                                      const float *weight, float *output, int in_features,
                                      int out_features, int num_tokens) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle);

    CUBLAS_CHECK(cublasSetStream(handle, s));

    const float alpha = 1.f;
    const float beta = 0.f;
    // weight 是 row-major [out, in]，跟 PyTorch nn.Linear.weight 一样。
    // 例：in=2 out=2，W=[[1, 2], [3, 4]]，x=[1, 1]^T -> y=[3, 7]^T。
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, num_tokens,
                             in_features, &alpha, weight, in_features, input, in_features, &beta,
                             output, out_features));
}
// TODO: add more perfomred GEMM path

// ======================== 仅供 Python 测试 ================================
extern "C" void linear_forward_host(float *input, float *weight, float *output, int in_features,
                                    int num_tokens, int out_features) {
    if (in_features <= 0 || out_features <= 0 || num_tokens <= 0) {
        std::fprintf(stderr,
                     "linear_forward_host: invalid in_features=%d out_features=%d num_tokens=%d\n",
                     in_features, out_features, num_tokens);
        return;
    }
    if (input == nullptr || weight == nullptr || output == nullptr) {
        std::fprintf(stderr, "linear_forward_host: null pointer argument\n");
        return;
    }

    const int64_t n_in_elem = static_cast<int64_t>(in_features) * num_tokens;
    const int64_t n_out_elem = static_cast<int64_t>(out_features) * num_tokens;

    float *d_x = nullptr;
    float *d_w = nullptr;
    float *d_y = nullptr;
    cudaStream_t stream;
    cublasHandle_t handle;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaMallocAsync(&d_x, static_cast<size_t>(n_in_elem) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(
        &d_w, static_cast<size_t>(out_features) * in_features * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, static_cast<size_t>(n_out_elem) * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_x, input, static_cast<size_t>(n_in_elem) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_w, weight,
                               static_cast<size_t>(out_features) * in_features * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

#if defined(MY_OPS_DEBUG)
    std::printf("linear_forward_host launch: in=%d out=%d tokens=%d\n", in_features, out_features,
                num_tokens);
    std::fflush(stdout);
#endif

    linear_forward_device(stream, handle, d_x, d_w, d_y, in_features, out_features, num_tokens);

    CUDA_CHECK(cudaMemcpyAsync(output, d_y, static_cast<size_t>(n_out_elem) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, stream));
    CUDA_CHECK(cudaFreeAsync(d_w, stream));
    CUDA_CHECK(cudaFreeAsync(d_y, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(handle));
}
