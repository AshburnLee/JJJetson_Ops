#include <cuda_runtime.h>
#include <cstring>       // memcpy
#include <stdio.h>
#include <vector>
#include "cuda_utils.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256
static __forceinline__ __device__ int64_t idx_after_roll(const int64_t idx, const int64_t ne) {
    if (idx < 0) {
        return idx + ne;
    }
    if (idx >= ne) {
        return idx - ne;
    }
    return idx;
}

static __global__ void roll_kernel(const float * __restrict__ src,
                                 float * __restrict__ dst,
                                 const int64_t ne0_0,
                                 const int64_t ne0_1,
                                 const int64_t ne0_2,
                                 const int64_t ne0_3,
                                 const int     shift0,
                                 const int     shift1,
                                 const int     shift2,
                                 const int     shift3) {
    // grid 和 block 都是只有x方向有值，故索引计算只有x
    const int64_t idx       = threadIdx.x + int64_t(blockDim.x) * blockIdx.x; 
    const int64_t stride0   = 1;
    const int64_t stride1   = ne0_0;
    const int64_t stride2   = ne0_0 * ne0_1;
    const int64_t stride3   = ne0_0 * ne0_1 * ne0_2;
    const int64_t n_elements = ne0_0 * ne0_1 * ne0_2 * ne0_3;

    if (idx >= n_elements) {
        return;
    }

    // 核心公式3。表达4个维度中元素索引
    const int64_t i0 = (idx / stride0) % ne0_0;
    const int64_t i1 = (idx / stride1) % ne0_1;
    const int64_t i2 = (idx / stride2) % ne0_2;
    const int64_t i3 = (idx / stride3) % ne0_3;

    // roll 之后的 4个维度中元素索引，从 ix 转换为 dx
    const int64_t d0 = idx_after_roll(i0 - shift0, ne0_0);
    const int64_t d1 = idx_after_roll(i1 - shift1, ne0_1);
    const int64_t d2 = idx_after_roll(i2 - shift2, ne0_2);
    const int64_t d3 = idx_after_roll(i3 - shift3, ne0_3);

    // src的读取变化了
    int64_t src_id =  1 * d0 + stride1 * d1 + stride2 * d2 + stride3 * d3;
    int64_t sdt_id =  1 * i0 + stride1 * i1 + stride2 * i2 + stride3 * i3;

    dst[sdt_id] = src[src_id];
}

extern "C" void roll(
                float* input, 
                float* output, 
                std::vector<int>& input_dims, 
                std::vector<int>& shifts) {
    // 直接按 col-major 约定解释 dims：
    // dims = [ne0_0, ne0_1, ne0_2, ne0_3]，其中 ne0_0 是变化最快的维度
    const int64_t ne0_0 = input_dims[0];
    const int64_t ne0_1 = input_dims[1];
    const int64_t ne0_2 = input_dims[2];
    const int64_t ne0_3 = input_dims[3];

    // shift0 对应变化最快的维度，shifts 顺序与 dims 一一对应
    const int64_t shift0 = shifts[0];
    const int64_t shift1 = shifts[1];
    const int64_t shift2 = shifts[2];
    const int64_t shift3 = shifts[3];

    const int64_t n_elem  = ne0_0 * ne0_1 * ne0_2 * ne0_3;

    float *d_x = nullptr;
    float *d_y = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_x, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, input, n_elem * sizeof(float), cudaMemcpyHostToDevice, stream));

    const dim3 threads(CUDA_ROPE_BLOCK_SIZE, 1, 1);
    const int blocks_x = (n_elem + CUDA_ROPE_BLOCK_SIZE - 1) / (CUDA_ROPE_BLOCK_SIZE);
    const dim3 blocks(blocks_x, 1, 1);
    roll_kernel<<<blocks, threads, 0, stream>>>(d_x, d_y, ne0_0, ne0_1, ne0_2, ne0_3, shift0, shift1, shift2, shift3);
    LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpyAsync(output, d_y, n_elem * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_y, nullptr));
}
