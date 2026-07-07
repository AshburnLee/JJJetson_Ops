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

// block level sum reduce，两端reduce
// warp 内 reduce 写入 shared 最后在第一个 warp 再 reduce
// 返回值仅 threadIdx.x == 0 有效
static __device__ __forceinline__ float block_reduce_sum(float val) {
    val = warp_reduce_down_sum(val);

    __shared__ float warp_partial[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        warp_partial[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x + 31) >> 5) ? warp_partial[lane] : 0.f;
    if (warp_id == 0) {
        val = warp_reduce_down_sum(val);
    }
    return val;
}

#endif // CUDA_UTILS_CUH_
