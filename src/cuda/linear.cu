#include <cuda_runtime.h>
#include <cstdio>
#include <unordered_map>
#include <vector>

#include "cublas_utils.cuh"
#include "cuda_utils.cuh"
#include "linear.h"

// 单 GEMM on device 接口
extern "C" void linear_forward_device(void *stream, void *cublas_handle, const float *input,
                                      const float *weight, float *output, int in_features,
                                      int out_features, int num_tokens) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle);

    CUBLAS_CHECK(cublasSetStream(handle, s));

    const float alpha = 1.f;
    const float beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, num_tokens,
                             in_features, &alpha, weight, in_features, input, in_features, &beta,
                             output, out_features));
}
// TODO: add more perfomred GEMM path

// =============================
// for test single Linear only
extern "C" void linear(float *input, float *weight, float *output, std::vector<int> &input_dims,
                       int out_features) {
    const int64_t ne0_0 = input_dims[0];
    const int64_t ne0_1 = input_dims[1];
    const int64_t ne0_2 = input_dims[2];
    const int64_t ne0_3 = input_dims[3];

    const int in_features = static_cast<int>(ne0_0);
    const int num_tokens = static_cast<int>(ne0_1 * ne0_2 * ne0_3);
    const int64_t n_in_elem = ne0_0 * ne0_1 * ne0_2 * ne0_3;
    const int64_t n_out_elem = static_cast<int64_t>(out_features) * ne0_1 * ne0_2 * ne0_3;

    if (in_features <= 0 || out_features <= 0 || num_tokens <= 0) {
        std::fprintf(stderr, "linear: invalid in_features=%d out_features=%d num_tokens=%d\n",
                     in_features, out_features, num_tokens);
        return;
    }

    float *d_x = nullptr;
    float *d_w = nullptr;
    float *d_y = nullptr;
    cudaStream_t stream;
    cublasHandle_t handle;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaMallocAsync(&d_x, n_in_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(
        &d_w, static_cast<size_t>(out_features) * in_features * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, n_out_elem * sizeof(float), stream));

    CUDA_CHECK(
        cudaMemcpyAsync(d_x, input, n_in_elem * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_w, weight,
                               static_cast<size_t>(out_features) * in_features * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

#if defined(MY_OPS_DEBUG)
    std::printf("linear launch: in=%d out=%d tokens=%d\n", in_features, out_features, num_tokens);
    std::fflush(stdout);
#endif

    linear_forward_device(stream, handle, d_x, d_w, d_y, in_features, out_features, num_tokens);

    CUDA_CHECK(
        cudaMemcpyAsync(output, d_y, n_out_elem * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, stream));
    CUDA_CHECK(cudaFreeAsync(d_w, stream));
    CUDA_CHECK(cudaFreeAsync(d_y, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(handle));
}
