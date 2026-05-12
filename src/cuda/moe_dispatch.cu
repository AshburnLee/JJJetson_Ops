#include <cuda_runtime.h>
#include <cstring>

#include "cuda_utils.cuh"

static __global__ void moe_dispatch_hist_kernel(const int *expert_ids, int num_routes,
                                                int num_experts, int *counts) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < num_routes) {
        const int e = expert_ids[r];
        if (e >= 0 && e < num_experts) {
            atomicAdd(&counts[e], 1);
        }
    }
}

static __global__ void moe_dispatch_exclusive_prefix_kernel(const int *counts, int num_experts,
                                                            int *offsets) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int sum = 0;
    for (int e = 0; e < num_experts; ++e) {
        offsets[e] = sum;
        sum += counts[e];
    }
    offsets[num_experts] = sum;
}

static __global__ void moe_dispatch_init_next_kernel(const int *offsets, int num_experts,
                                                     int *next_slot) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_experts) {
        next_slot[e] = offsets[e];
    }
}

static __global__ void moe_dispatch_scatter_kernel(const float *x,        /* in */
                                                   const int *expert_ids, /* in */
                                                   int num_tokens, int top_k, int hidden_size,
                                                   int num_experts, int *next_slot,
                                                   float *permuted_x, /* out */
                                                   int *source_token, /* out */
                                                   int *source_k) {   /* out */
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_routes = num_tokens * top_k;
    if (r >= num_routes) {
        return;
    }
    const int t = r / top_k;
    const int k = r - t * top_k;
    const int e = expert_ids[r];
    if (e < 0 || e >= num_experts) {
        return;
    }
    const int pos = atomicAdd(&next_slot[e], 1);
    const float *src = x + static_cast<size_t>(t) * hidden_size;
    float *dst = permuted_x + static_cast<size_t>(pos) * hidden_size;
    for (int h = 0; h < hidden_size; ++h) {
        dst[h] = src[h];
    }
    source_token[pos] = t;
    source_k[pos] = k;
}

extern "C" void moe_dispatch_launch_cuda(cudaStream_t stream, const float *d_x,
                                         const int *d_expert_ids, int num_tokens, int top_k,
                                         int hidden_size, int num_experts, float *d_permuted_x,
                                         int *d_source_token, int *d_source_k,
                                         int *d_expert_offsets, int *d_counts, int *d_next_slot) {
    const int num_routes = num_tokens * top_k;
    CUDA_CHECK(
        cudaMemsetAsync(d_counts, 0, static_cast<size_t>(num_experts) * sizeof(int), stream));

    const int threads = 256;
    const int blocks_hist = (num_routes + threads - 1) / threads;
    moe_dispatch_hist_kernel<<<blocks_hist, threads, 0, stream>>>(d_expert_ids, num_routes,
                                                                  num_experts, d_counts);
    LAUNCH_CHECK();

    // 1 个thread？
    moe_dispatch_exclusive_prefix_kernel<<<1, 1, 0, stream>>>(d_counts, num_experts,
                                                              d_expert_offsets);
    LAUNCH_CHECK();

    const int init_threads = 256;
    const int init_blocks = (num_experts + init_threads - 1) / init_threads;
    moe_dispatch_init_next_kernel<<<init_blocks, init_threads, 0, stream>>>(
        d_expert_offsets, num_experts, d_next_slot);
    LAUNCH_CHECK();

    moe_dispatch_scatter_kernel<<<blocks_hist, threads, 0, stream>>>(
        d_x, d_expert_ids, num_tokens, top_k, hidden_size, num_experts, d_next_slot, d_permuted_x,
        d_source_token, d_source_k);
    LAUNCH_CHECK();
}

extern "C" void moe_dispatch(const float *x_host, const int *expert_ids_host, int num_tokens,
                             int top_k, int hidden_size, int num_experts, float *permuted_x_host,
                             int *source_token_host, int *source_k_host, int *expert_offsets_host) {
    const int num_routes = num_tokens * top_k;
    const size_t x_size = static_cast<size_t>(num_tokens) * hidden_size;
    const size_t perm_size = static_cast<size_t>(num_routes) * hidden_size;
    const size_t id_size = static_cast<size_t>(num_routes) * sizeof(int);

    float *d_x = nullptr, *d_perm = nullptr;
    int *d_ids = nullptr, *d_src_t = nullptr, *d_src_k = nullptr, *d_off = nullptr,
        *d_counts = nullptr, *d_next = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMalloc(&d_x, x_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_perm, perm_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ids, id_size));
    CUDA_CHECK(cudaMalloc(&d_src_t, id_size));
    CUDA_CHECK(cudaMalloc(&d_src_k, id_size));
    CUDA_CHECK(cudaMalloc(&d_off, static_cast<size_t>(num_experts + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, static_cast<size_t>(num_experts) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next, static_cast<size_t>(num_experts) * sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_x, x_host, x_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ids, expert_ids_host, id_size, cudaMemcpyHostToDevice, stream));

    moe_dispatch_launch_cuda(stream, d_x, d_ids, num_tokens, top_k, hidden_size, num_experts,
                             d_perm, d_src_t, d_src_k, d_off, d_counts, d_next);

    CUDA_CHECK(cudaMemcpyAsync(permuted_x_host, d_perm, perm_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(source_token_host, d_src_t, id_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(source_k_host, d_src_k, id_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(expert_offsets_host, d_off,
                               static_cast<size_t>(num_experts + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_perm));
    CUDA_CHECK(cudaFree(d_ids));
    CUDA_CHECK(cudaFree(d_src_t));
    CUDA_CHECK(cudaFree(d_src_k));
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
