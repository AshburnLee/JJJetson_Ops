#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

#define CUDA_RMS_NORM_FUSED_ADD_BLOCK_SIZE 256

static __global__ void rms_norm_fused_add_kernel(float *__restrict__ input,
                                                 float *__restrict__ residual,
                                                 const float *__restrict__ weight,
                                                 const int hidden_size, const int num_tokens,
                                                 const float epsilon) {
    const int token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }

    const int64_t row_base = static_cast<int64_t>(token) * hidden_size;

    float sum_square = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        // z = input[i] + residual[i]
        const float z = input[row_base + i] + residual[row_base + i];
        residual[row_base + i] = z;
        sum_square += z * z;
    }

    sum_square = block_reduce_sum(sum_square);

    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        // inv_rms = 1 / sqrt(mean(z^2) + epsilon)
        inverse_rms = rsqrtf(sum_square / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float z = residual[row_base + i];
        // input[i] = z * inv_rms * weight[i]
        input[row_base + i] = z * inverse_rms * weight[i];
    }
}

// input/residual shape: [hidden_size, num_tokens, 1, 1]
extern "C" void rms_norm_fused_add(float *input, float *residual, float *weight,
                                   std::vector<int> &input_dims, float epsilon) {
    const int64_t ne0_0 = input_dims[0];
    const int64_t ne0_1 = input_dims[1];
    const int64_t ne0_2 = input_dims[2];
    const int64_t ne0_3 = input_dims[3];

    const int hidden_size = static_cast<int>(ne0_0);
    const int num_tokens = static_cast<int>(ne0_1 * ne0_2 * ne0_3);
    const int64_t n_elem = ne0_0 * ne0_1 * ne0_2 * ne0_3;

    if (hidden_size <= 0 || num_tokens <= 0) {
        std::fprintf(stderr, "rms_norm_fused_add: invalid hidden_size=%d num_tokens=%d\n",
                     hidden_size, num_tokens);
        return;
    }

    float *d_x = nullptr;
    float *d_r = nullptr;
    float *d_w = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_x, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_r, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_w, hidden_size * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_x, input, n_elem * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_r, residual, n_elem * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_w, weight, hidden_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    const int block_size = hidden_size < CUDA_RMS_NORM_FUSED_ADD_BLOCK_SIZE
                               ? hidden_size
                               : CUDA_RMS_NORM_FUSED_ADD_BLOCK_SIZE;
    const dim3 threads(block_size, 1, 1);
    const dim3 blocks(static_cast<unsigned>(num_tokens), 1, 1);

#if defined(MY_OPS_DEBUG)
    std::printf("rms_norm_fused_add launch: block=(%u,%u,%u), grid=(%u,%u,%u), hidden=%d tokens=%d "
                "eps=%g\n",
                threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, hidden_size,
                num_tokens, static_cast<double>(epsilon));
    std::fflush(stdout);
#endif

    rms_norm_fused_add_kernel<<<blocks, threads, 0, stream>>>(d_x, d_r, d_w, hidden_size,
                                                              num_tokens, epsilon);
    LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpyAsync(input, d_x, n_elem * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(residual, d_r, n_elem * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, stream));
    CUDA_CHECK(cudaFreeAsync(d_r, stream));
    CUDA_CHECK(cudaFreeAsync(d_w, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
