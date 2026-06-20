#ifndef CUDA_UTILS_CUH_
#define CUDA_UTILS_CUH_

#include "cuda_utils.h"
// 该 Header 只有 Device TU 翻译单元

// 使用 shuffle 指令在单个 warp 内做 reduce max / sum
// 所有 32 个线程都参与（mask 0xffffffff）
static __device__ __forceinline__ float warp_reduce_down_max(float val) {
    unsigned mask = 0xffffffffu;
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

#endif // CUDA_UTILS_CUH_
