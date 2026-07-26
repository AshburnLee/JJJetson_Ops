// RMSNorm（非 fused）：y[i] = x[i] / RMS(x) * weight[i]
// 布局 col-major [hidden_size, num_tokens, 1, 1]，在 hidden_size 维归一化
#include "rms_norm.h"

#include <cuda_runtime.h>
#include <cstdio>

#include "cuda_utils.cuh"

#define CUDA_RMS_NORM_BLOCK_SIZE 256

static __global__ void rms_norm_kernel(const float *__restrict__ input,
                                       const float *__restrict__ weight, float *__restrict__ output,
                                       const int hidden_size, const int num_tokens,
                                       const float epsilon) {
    const int token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }

    const int64_t row_base = static_cast<int64_t>(token) * hidden_size;

    float sum_square = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float x = input[row_base + i];
        sum_square += x * x;
    }

    sum_square = block_reduce_sum(sum_square);

    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(sum_square / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float x = input[row_base + i];
        output[row_base + i] = x * inverse_rms * weight[i];
    }
}

static int rms_norm_check_shape(int hidden_size, int num_tokens) {
    if (hidden_size <= 0 || num_tokens <= 0) {
        std::fprintf(stderr, "rms_norm_forward_device: invalid hidden_size=%d num_tokens=%d\n",
                     hidden_size, num_tokens);
        return -1;
    }
    return 0;
}

static void rms_norm_launch_device(cudaStream_t stream, const float *d_input, const float *d_weight,
                                   float *d_output, int hidden_size, int num_tokens,
                                   float epsilon) {
    const int block_size =
        hidden_size < CUDA_RMS_NORM_BLOCK_SIZE ? hidden_size : CUDA_RMS_NORM_BLOCK_SIZE;
    const dim3 threads(block_size, 1, 1);
    const dim3 blocks(static_cast<unsigned>(num_tokens), 1, 1);

#if defined(MY_OPS_DEBUG)
    std::printf("rms_norm_forward_device launch: block=(%u,%u,%u), grid=(%u,%u,%u), hidden=%d "
                "tokens=%d eps=%g\n",
                threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z, hidden_size,
                num_tokens, static_cast<double>(epsilon));
    std::fflush(stdout);
#endif

    rms_norm_kernel<<<blocks, threads, 0, stream>>>(d_input, d_weight, d_output, hidden_size,
                                                    num_tokens, epsilon);
    LAUNCH_CHECK();
}

// -========================-- 生产（device）--========================-
extern "C" int rms_norm_forward_device(void *stream, const float *d_input, const float *d_weight,
                                       float *d_output, int hidden_size, int num_tokens,
                                       float epsilon) {
    if (rms_norm_check_shape(hidden_size, num_tokens) != 0) {
        return -1;
    }
    if (d_input == nullptr || d_weight == nullptr || d_output == nullptr) {
        std::fprintf(stderr, "rms_norm_forward_device: d_input/d_weight/d_output is null\n");
        return -1;
    }

    cudaStream_t s = stream != nullptr ? static_cast<cudaStream_t>(stream) : nullptr;
    if (s == nullptr) {
        std::fprintf(stderr, "rms_norm_forward_device: stream is null\n");
        return -1;
    }

    rms_norm_launch_device(s, d_input, d_weight, d_output, hidden_size, num_tokens, epsilon);
    return 0;
}

// ======================== 仅供 Python 测试 ================================
// 非生产：host H2D → rms_norm_forward_device → D2H，仅供 Python binding 测试
extern "C" void rms_norm_forward_host(const float *input, const float *weight, float *output,
                                      int hidden_size, int num_tokens, float epsilon) {
    if (rms_norm_check_shape(hidden_size, num_tokens) != 0) {
        return;
    }
    if (input == nullptr || weight == nullptr || output == nullptr) {
        std::fprintf(stderr, "rms_norm_forward_host: null pointer argument\n");
        return;
    }

    const int64_t n_elem = static_cast<int64_t>(hidden_size) * num_tokens;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_x = nullptr;
    float *d_w = nullptr;
    float *d_y = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_x, static_cast<size_t>(n_elem) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_w, static_cast<size_t>(hidden_size) * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, static_cast<size_t>(n_elem) * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_x, input, static_cast<size_t>(n_elem) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_w, weight, static_cast<size_t>(hidden_size) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    if (rms_norm_forward_device(stream, d_x, d_w, d_y, hidden_size, num_tokens, epsilon) != 0) {
        CUDA_CHECK(cudaFreeAsync(d_x, stream));
        CUDA_CHECK(cudaFreeAsync(d_w, stream));
        CUDA_CHECK(cudaFreeAsync(d_y, stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(output, d_y, static_cast<size_t>(n_elem) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, stream));
    CUDA_CHECK(cudaFreeAsync(d_w, stream));
    CUDA_CHECK(cudaFreeAsync(d_y, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
