#include "swiglu.h"

#include "cuda_utils.cuh"

static __device__ __forceinline__ float swiglu_silu_f(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

static __global__ void swiglu_silu_mul_kernel(const float *gate, const float *up, float *mid,
                                              int n_elem) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elem) {
        mid[i] = swiglu_silu_f(gate[i]) * up[i];
    }
}

// 生产用 Device API
extern "C" void swiglu_silu_mul_launch_device(void *stream, const float *gate, const float *up,
                                              float *mid, int n_elem) {
    if (n_elem <= 0) {
        return;
    }

    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (n_elem + threads - 1) / threads;
    swiglu_silu_mul_kernel<<<blocks, threads, 0, s>>>(gate, up, mid, n_elem);
    LAUNCH_CHECK();
}

// TODO: 添加 test 入口
