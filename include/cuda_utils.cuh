#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t error = call;                                         \
        if (error != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(error));       \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define LAUNCH_CHECK()                                                    \
    do {                                                                  \
        cudaError_t error = cudaGetLastError();                           \
        if (error != cudaSuccess) {                                       \
            fprintf(stderr, "Kernel launch failed at %s:%d - %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(error));       \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// 使用 shuffle 指令在单个 warp 内做 reduce max / sum。
// 所有 32 个线程都参与（mask 0xffffffff）。
static __device__ __forceinline__ float warp_reduce_down_max(float val) {
    unsigned mask = 0xffffffffu;
    // 典型的树形归约：16,8,4,2,1
    val = fmaxf(val, __shfl_down_sync(mask, val, 16));
    val = fmaxf(val, __shfl_down_sync(mask, val, 8));
    val = fmaxf(val, __shfl_down_sync(mask, val, 4));
    val = fmaxf(val, __shfl_down_sync(mask, val, 2));
    val = fmaxf(val, __shfl_down_sync(mask, val, 1));
    return val;
}

static __device__ __forceinline__ float warp_reduce_down_sum(float val) {
    unsigned mask = 0xffffffffu;
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val;
}

static __device__ __forceinline__ float warp_reduce_xor_max(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, 32));
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_xor_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

#endif  // CUDA_UTILS_H_
