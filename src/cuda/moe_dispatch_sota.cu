// 迁移自 vllm
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "cuda_utils.cuh"

namespace {

constexpr int pad_to_multiple_of_16(int input) {
    constexpr int ALIGNMENT = 16;
    return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

// vLLM CubKeyValueSorter::expertsToBits
__host__ inline int experts_to_sort_key_bits(int num_experts) {
    const int v = 2 * num_experts - 1;
    return static_cast<int>(std::floor(std::log2(static_cast<double>(v)))) + 1;
}

__host__ size_t radix_sort_workspace_bytes_host(int num_key_value_pairs, int num_experts) {
    const int begin_bit = 0;
    const int end_bit = experts_to_sort_key_bits(num_experts);
    size_t storage = 0;
    int *null_int = nullptr;
    cub::DeviceRadixSort::SortPairs(nullptr, storage, null_int, null_int, null_int, null_int,
                                    static_cast<int>(num_key_value_pairs), begin_bit, end_bit);
    if (storage == 0) {
        storage = 1;
    }
    return storage;
}

__global__ void moe_dispatch_sota_fill_arange_kernel(int *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}

template <typename T>
__device__ inline int find_total_elts_less_than_target(T const *sorted_indices, int arr_length,
                                                       T target) {
    int low = 0;
    int high = arr_length - 1;
    int target_location = -1;
    while (low <= high) {
        const int mid = (low + high) / 2;
        if (sorted_indices[mid] >= target) {
            high = mid - 1;
        } else {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

__global__ void moe_dispatch_sota_expert_offsets_kernel(const int *sorted_expert_ids,
                                                        int sorted_len, int num_experts,
                                                        int *expert_offsets) {
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert > num_experts) {
        return;
    }
    expert_offsets[expert] =
        find_total_elts_less_than_target(sorted_expert_ids, sorted_len, expert);
}

__global__ void moe_dispatch_sota_expand_rows_kernel(const float *d_x, float *d_permuted_x,
                                                     const int *d_expanded_src_for_dest,
                                                     int *d_source_token, int *d_source_k,
                                                     int top_k, int hidden_size, int num_routes) {
    const int dest_row = blockIdx.x;
    if (dest_row >= num_routes) {
        return;
    }
    const int expanded_src = d_expanded_src_for_dest[dest_row];
    const int src_token = expanded_src / top_k;
    const int src_k = expanded_src - src_token * top_k;
    if (threadIdx.x == 0) {
        d_source_token[dest_row] = src_token;
        d_source_k[dest_row] = src_k;
    }
    __syncthreads();

    const float *src_row = d_x + static_cast<size_t>(src_token) * hidden_size;
    float *dst_row = d_permuted_x + static_cast<size_t>(dest_row) * hidden_size;
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        dst_row[h] = src_row[h];
    }
}

} // namespace

extern "C" size_t moe_dispatch_sota_sort_workspace_bytes(int num_tokens, int top_k,
                                                         int num_experts) {
    const int num_routes = num_tokens * top_k;
    if (num_routes <= 0 || num_experts <= 0) {
        return 1;
    }
    const size_t raw = radix_sort_workspace_bytes_host(num_routes, num_experts);
    return static_cast<size_t>(pad_to_multiple_of_16(static_cast<int>(raw)));
}

extern "C" void moe_dispatch_sota_launch_cuda(
    cudaStream_t stream, const float *d_x, const int *d_expert_ids, int num_tokens, int top_k,
    int hidden_size, int num_experts, void *d_sort_workspace, size_t sort_workspace_bytes,
    int *d_sorted_expert_ids, int *d_expanded_src_for_dest_row, float *d_permuted_x,
    int *d_source_token, int *d_source_k, int *d_expert_offsets, int *d_arange_buf) {
    const int num_routes = num_tokens * top_k;
    CUDA_CHECK(cudaMemsetAsync(d_expert_offsets, 0,
                               static_cast<size_t>(num_experts + 1) * sizeof(int), stream));

    const int threads_fill = 256;
    const int blocks_fill = (num_routes + threads_fill - 1) / threads_fill;
    moe_dispatch_sota_fill_arange_kernel<<<blocks_fill, threads_fill, 0, stream>>>(d_arange_buf,
                                                                                   num_routes);
    LAUNCH_CHECK();

    const int begin_bit = 0;
    const int end_bit = experts_to_sort_key_bits(num_experts);
    size_t temp_storage = sort_workspace_bytes;
    const cudaError_t sort_err = cub::DeviceRadixSort::SortPairs(
        d_sort_workspace, temp_storage, d_expert_ids, d_sorted_expert_ids, d_arange_buf,
        d_expanded_src_for_dest_row, num_routes, begin_bit, end_bit, stream);
    if (sort_err != cudaSuccess) {
        fprintf(stderr, "cub::DeviceRadixSort::SortPairs failed: %s\n",
                cudaGetErrorString(sort_err));
        exit(EXIT_FAILURE);
    }

    const int off_threads = 256;
    const int off_blocks = (num_experts + 1 + off_threads - 1) / off_threads;
    moe_dispatch_sota_expert_offsets_kernel<<<off_blocks, off_threads, 0, stream>>>(
        d_sorted_expert_ids, num_routes, num_experts, d_expert_offsets);
    LAUNCH_CHECK();

    const int expand_threads = 256;
    moe_dispatch_sota_expand_rows_kernel<<<num_routes, expand_threads, 0, stream>>>(
        d_x, d_permuted_x, d_expanded_src_for_dest_row, d_source_token, d_source_k, top_k,
        hidden_size, num_routes);
    LAUNCH_CHECK();
}

extern "C" void moe_dispatch_sota(const float *x_host, const int *expert_ids_host, int num_tokens,
                                  int top_k, int hidden_size, int num_experts,
                                  float *permuted_x_host, int *source_token_host,
                                  int *source_k_host, int *expert_offsets_host) {
    const int num_routes = num_tokens * top_k;
    const size_t x_bytes = static_cast<size_t>(num_tokens) * hidden_size * sizeof(float);
    const size_t perm_bytes = static_cast<size_t>(num_routes) * hidden_size * sizeof(float);
    const size_t route_int_bytes = static_cast<size_t>(num_routes) * sizeof(int);
    const size_t off_bytes = static_cast<size_t>(num_experts + 1) * sizeof(int);

    const size_t ws_bytes = moe_dispatch_sota_sort_workspace_bytes(num_tokens, top_k, num_experts);

    float *d_x = nullptr;
    float *d_perm = nullptr;
    int *d_ids = nullptr;
    int *d_sorted_experts = nullptr;
    int *d_expanded_src = nullptr;
    int *d_arange = nullptr;
    int *d_src_t = nullptr;
    int *d_src_k = nullptr;
    int *d_off = nullptr;
    void *d_ws = nullptr;

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_perm, perm_bytes));
    CUDA_CHECK(cudaMalloc(&d_ids, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&d_sorted_experts, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&d_expanded_src, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&d_arange, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&d_src_t, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&d_src_k, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&d_off, off_bytes));
    CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));

    CUDA_CHECK(cudaMemcpyAsync(d_x, x_host, x_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_ids, expert_ids_host, route_int_bytes, cudaMemcpyHostToDevice, stream));

    moe_dispatch_sota_launch_cuda(stream, d_x, d_ids, num_tokens, top_k, hidden_size, num_experts,
                                  d_ws, ws_bytes, d_sorted_experts, d_expanded_src, d_perm, d_src_t,
                                  d_src_k, d_off, d_arange);

    CUDA_CHECK(
        cudaMemcpyAsync(permuted_x_host, d_perm, perm_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(source_token_host, d_src_t, route_int_bytes, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(
        cudaMemcpyAsync(source_k_host, d_src_k, route_int_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(expert_offsets_host, d_off, off_bytes, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_perm));
    CUDA_CHECK(cudaFree(d_ids));
    CUDA_CHECK(cudaFree(d_sorted_experts));
    CUDA_CHECK(cudaFree(d_expanded_src));
    CUDA_CHECK(cudaFree(d_arange));
    CUDA_CHECK(cudaFree(d_src_t));
    CUDA_CHECK(cudaFree(d_src_k));
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_ws));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
