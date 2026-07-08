#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

#define CUDA_LAYER_NORM_BLOCK_SIZE 256

static __global__ void layer_norm_kernel(const float *__restrict__ input,
                                         const float *__restrict__ weight,
                                         const float *__restrict__ bias, float *__restrict__ output,
                                         const int hidden_size, const int num_tokens,
                                         const float epsilon) {
    // one block covers one TOKEN
    const int token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }

    const int64_t row_base = static_cast<int64_t>(token) * hidden_size;

    float sum_x = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum_x += input[row_base + i];
    }
    sum_x = block_reduce_sum(sum_x);

    __shared__ float mean;
    if (threadIdx.x == 0) {
        // mean = sum(x) / hidden_size
        mean = sum_x / static_cast<float>(hidden_size);
    }
    __syncthreads();

    float sum_square_diff = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float diff = input[row_base + i] - mean;
        sum_square_diff += diff * diff;
    }
    sum_square_diff = block_reduce_sum(sum_square_diff);

    __shared__ float inverse_std;
    if (threadIdx.x == 0) {
        // var = sum((x - mean)^2) / hidden_size
        // inverse_std = 1 / sqrt(var + epsilon)
        inverse_std = rsqrtf(sum_square_diff / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float x = input[row_base + i];
        // y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i]
        output[row_base + i] = (x - mean) * inverse_std * weight[i] + bias[i];
    }
}
// input shape: [hidden_size, num_tokens, 1, 1]
extern "C" void layer_norm(float *input, float *weight, float *bias, float *output,
                           std::vector<int> &input_dims, float epsilon) {
    const int64_t ne0_0 = input_dims[0];
    const int64_t ne0_1 = input_dims[1];
    const int64_t ne0_2 = input_dims[2];
    const int64_t ne0_3 = input_dims[3];

    const int hidden_size = static_cast<int>(ne0_0);
    const int num_tokens = static_cast<int>(ne0_1 * ne0_2 * ne0_3);
    const int64_t n_elem = ne0_0 * ne0_1 * ne0_2 * ne0_3;

    if (hidden_size <= 0 || num_tokens <= 0) {
        std::fprintf(stderr, "layer_norm: invalid hidden_size=%d num_tokens=%d\n", hidden_size,
                     num_tokens);
        return;
    }

    float *d_x = nullptr;
    float *d_w = nullptr;
    float *d_b = nullptr;
    float *d_y = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_x, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_w, hidden_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_b, hidden_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, n_elem * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_x, input, n_elem * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_w, weight, hidden_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_b, bias, hidden_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    const int block_size =
        hidden_size < CUDA_LAYER_NORM_BLOCK_SIZE ? hidden_size : CUDA_LAYER_NORM_BLOCK_SIZE;
    const dim3 threads(block_size, 1, 1);
    const dim3 blocks(static_cast<unsigned>(num_tokens), 1, 1);

#if defined(MY_OPS_DEBUG)
    std::printf(
        "layer_norm launch: block=(%u,%u,%u), grid=(%u,%u,%u), hidden=%d tokens=%d eps=%g\n",
        threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, hidden_size, num_tokens,
        static_cast<double>(epsilon));
    std::fflush(stdout);
#endif

    layer_norm_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w, d_b, d_y, hidden_size, num_tokens,
                                                      epsilon);
    LAUNCH_CHECK();

    CUDA_CHECK(
        cudaMemcpyAsync(output, d_y, n_elem * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, stream));
    CUDA_CHECK(cudaFreeAsync(d_w, stream));
    CUDA_CHECK(cudaFreeAsync(d_b, stream));
    CUDA_CHECK(cudaFreeAsync(d_y, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
