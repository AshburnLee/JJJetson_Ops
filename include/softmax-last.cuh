#ifndef SOFTMAX_LAST_CUH_
#define SOFTMAX_LAST_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>

#pragma once

using namespace std;

int GetNearGreaterPowerOfTwo(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) {
      ++log2_value;
    }
    return 1 << log2_value;
}

template <typename T>
void LaunchSoftmax(T* input_data, T* output_data, std::vector<int>& dims, int axis);

// 显式实例化：生成 float 版本的符号（解决 undefined）
template void LaunchSoftmax<float>(float* input_data, float* output_data, std::vector<int>& dims, int axis);


#ifdef __CUDACC__
/*
CudaShuffleXorSync 函数是 CUDA 7.0 及更高版本中引入的，用于 warp 内线程间的同步通信。
它依赖于 warp 内线程的物理布局，因此只能在同一个 warp 内的线程间使用
*/
template <typename T, int KernelWarpSize>
__device__ __forceinline__ T WarpReduceSum(T value) {
#pragma unroll
  for (int offset = KernelWarpSize / 2; offset > 0; offset /= 2) {
    T sum_val = __shfl_xor_sync(0xFFFFFFFF, value, offset);
    value = value + sum_val;
  }
  return value;
}

template <typename T, int KernelWarpSize>
__device__ __forceinline__ T WarpReduceMax(T value) {
#pragma unroll
  for (int offset = KernelWarpSize / 2; offset > 0; offset /= 2) {
    T max_val = __shfl_xor_sync(0xFFFFFFFF, value, offset);
    value = max(value, max_val);
  }
  return value;
}

template <typename T>
__device__ T get_negative_infinity();

template <>
__device__ float get_negative_infinity<float>() {
    return -__int_as_float(0x7f800000);
}

template <>
__device__ double get_negative_infinity<double>() {
    return -__longlong_as_double(0x7ff0000000000000LL);
}


/// 在 warp 中计算 log softmax
/// 在 warp 内部的规约 (reduction) 通常不需要使用共享内存。 这是因为 warp 内部的线程可以通过 warp shuffle 指令直接进行高效的数据交换。 
/// warp shuffle 指令允许线程在 warp 内快速地访问其他线程的数据，而无需经过共享内存。
/// warp shuffle 指令比访问共享内存更快，因为共享内存访问需要同步操作 (__syncthreads)，而 warp shuffle 是同步的。
/// warp shuffle 的延迟比共享内存访问低
/// 对于 warp 内部的规约，warp shuffle 是更优的选择。 只有在需要跨 warp 进行规约时，才需要使用共享内存。
template <typename T, typename AccT, int NearGreaterPowerOfTwo>
__global__ void ComputeLogSoftmaxForwardInWarp(T *dst, const T *src,
                                               int batch_size,
                                               int element_count) {
  constexpr int near_greater_power_of_two = NearGreaterPowerOfTwo;
  constexpr int kernel_warp_size = (near_greater_power_of_two < 32) ? near_greater_power_of_two : 32;
  constexpr int warp_iter = near_greater_power_of_two / kernel_warp_size;
  int batch_id = blockDim.y * blockIdx.x + threadIdx.y;

  // set effective_warp_id as 1 when warps do effective work,
  // when warps do ineffective work, effective_warp_id remains unchanged.
  int effective_warp_id = batch_size - batch_id;
  if (effective_warp_id > 1) effective_warp_id = 1;

  int thread_in_warp_idx = threadIdx.x;

  // 1.read data from global memory to registers
  AccT elements[warp_iter];
  // set effective_element_count as the num of elements when warps do effective
  // work
  // set effective_element_count as 0, when warps do ineffective work
  int effective_element_count = (effective_warp_id <= 0) ? 0 : element_count;
  for (int it = 0; it < warp_iter; ++it) {
    int element_index = thread_in_warp_idx + it * kernel_warp_size;
    if (element_index < effective_element_count) {
      elements[it] =
          static_cast<AccT>(src[batch_id * element_count + element_index]);
    } else {
      // elements[it] = -std::numeric_limits<AccT>::infinity();
      elements[it] = get_negative_infinity<AccT>();
    }
  }

  // 2.compute max_value. For each thread, loop all registers to find max
  AccT max_value = elements[0];
#pragma unroll
  for (int it = 1; it < warp_iter; ++it) {
    max_value = (max_value > elements[it]) ? max_value : elements[it];
  }
  max_value = WarpReduceMax<AccT, kernel_warp_size>(max_value);

  // 3.For each warp, accumulate all thread registers
  AccT sum = 0.0f;
#pragma unroll
  for (int it = 0; it < warp_iter; ++it) {
    sum += std::exp(elements[it] - max_value);
  }
  sum = WarpReduceSum<AccT, kernel_warp_size>(sum);

  // 4.store result.
  sum = std::log(sum);
#pragma unroll
  for (int it = 0; it < warp_iter; ++it) {
    int element_index = thread_in_warp_idx + it * kernel_warp_size;
    if (element_index < element_count) {
      dst[batch_id * element_count + element_index] = static_cast<T>(elements[it] - max_value - sum);
    } else {
      break;
    }
  }
}

#define LAUNCH_WARP_FORWAR_COMPUTE(near_greater_power_of_two)                \
  case near_greater_power_of_two:                                            \
    ComputeLogSoftmaxForwardInWarp<                                          \
        T, AccT, near_greater_power_of_two><<<blocks, threads, 0>>>(         \
        dst, src, outer_size, dim_size);                                     \
    break;

template <typename T, typename AccT>
void LaunchSoftmaxForwardForLastAxis(T *dst, const T *src, int dim_size,
                                     int outer_size) {
  // BLOCK 配置
  int threads_per_block = 128;
  int near_greater_power_of_two = GetNearGreaterPowerOfTwo(dim_size);
  int kernel_warp_size = (near_greater_power_of_two < 32) ? near_greater_power_of_two : 32;
  int warps_per_block = (threads_per_block / kernel_warp_size);
  int blocks = (outer_size + warps_per_block - 1) / warps_per_block;
  dim3 threads(kernel_warp_size, warps_per_block, 1);

  /*
  cout << "============\nblock config: \n";
  cout << "threads_per_block: " << threads_per_block << "\n";
  cout << "dim_size: " << dim_size << "\n";
  cout << "near_greater_power_of_two: " << near_greater_power_of_two << "\n";

  cout << "kernel_warp_size: " << kernel_warp_size << "\n";
  cout << "warps_per_block: " << warps_per_block << "\n";
  cout << "blocks: " << blocks << "\n";
  cout << "============\n";
  */

  // 参数化配置参数，用户根据这个参数值，调整 block 的具体配置
  switch (near_greater_power_of_two) {
    LAUNCH_WARP_FORWAR_COMPUTE(1);
    LAUNCH_WARP_FORWAR_COMPUTE(2);
    LAUNCH_WARP_FORWAR_COMPUTE(4);     // dim_size: 3~4
    LAUNCH_WARP_FORWAR_COMPUTE(8);     // dim_size: 5~8
    LAUNCH_WARP_FORWAR_COMPUTE(16);    // dim_size: 9~16
    LAUNCH_WARP_FORWAR_COMPUTE(32);    // dim_size: 17~32
    LAUNCH_WARP_FORWAR_COMPUTE(64);    // dim_size: 33~64
    LAUNCH_WARP_FORWAR_COMPUTE(128);   // dim_size 65~128
    LAUNCH_WARP_FORWAR_COMPUTE(256);   // dim_size 129~256
    LAUNCH_WARP_FORWAR_COMPUTE(512);   // dim_size 257~512
    LAUNCH_WARP_FORWAR_COMPUTE(1024);  // dim_size 513~1024

    default:
      break;
  }
}


// template <typename T>
// void LaunchSoftmax(T* input_data, T* output_data, std::vector<int>& dims, int axis);

template <typename T>
void LaunchSoftmax(T* input_data, T* output_data, std::vector<int>& dims, int axis = 3) {

  int dim_size = dims[axis];
  int inner_size = 1;
  for (int i = axis + 1; i < 4; ++i) {
    inner_size *= dims[i];
  }

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= dims[i];
  }

  // warmup run & for ncu
  if (inner_size == 1 && dim_size <= 1024 && dim_size * sizeof(T) <= 4096) {
    LaunchSoftmaxForwardForLastAxis<T, T>(output_data, input_data, dim_size, outer_size);
  } else {
    cout << "axis is not the last\n";
  }

  for (int i = 0; i < 10; ++i) {
    if (inner_size == 1 && dim_size <= 1024 && dim_size * sizeof(T) <= 4096) {
      LaunchSoftmaxForwardForLastAxis<T, T>(output_data, input_data, dim_size, outer_size);
    } else {
      cout << "axis is not the last\n";
    }
    cudaDeviceSynchronize();
  }
}

#endif  // __CUDACC__


#endif  // SOFTMAX_LAST_CUH_
