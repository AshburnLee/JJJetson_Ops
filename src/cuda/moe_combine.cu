#include <cuda_runtime.h>
#include "cuda_utils.cuh"

static __global__ void moe_combine_kernel(const float* expert_out,
                                          const int* source_token,
                                          const int* source_k,
                                          const float* route_weights,
                                          float* y,
                                          int num_routes, /*num_token * top-k*/
                                          int hidden_size,
                                          int top_k) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_routes) {
        return;
    }
    // d_expert_out 与 dispatch 输出的 permuted 行对齐, 第 pos 行对应该 slot 的专家输出
    const int t = source_token[pos];
    const int k = source_k[pos];
    const float w = route_weights[static_cast<size_t>(t) * top_k + k];
    const float* src = expert_out + static_cast<size_t>(pos) * hidden_size;
    float* dst_base = y + static_cast<size_t>(t) * hidden_size;
    // 每一个 hidden_size 位置，累加不同 pos，使用 atomicAdd 防止不同thread对同一个位置的访存冲突 
    for (int h = 0; h < hidden_size; ++h) {
        atomicAdd(dst_base + h, w * src[h]);
    }
}

extern "C" void moe_combine_launch_cuda(cudaStream_t stream,
                                        const float* d_expert_out,
                                        const int* d_source_token,
                                        const int* d_source_k,
                                        const float* d_route_weights,
                                        float* d_y,
                                        int num_routes,
                                        int hidden_size,
                                        int top_k) {
    const int threads = 256;
    const int blocks = (num_routes + threads - 1) / threads;
    moe_combine_kernel<<<blocks, threads, 0, stream>>>(d_expert_out,
                                                       d_source_token,
                                                       d_source_k,
                                                       d_route_weights,
                                                       d_y,
                                                       num_routes,
                                                       hidden_size,
                                                       top_k);
    LAUNCH_CHECK();
}

extern "C" void moe_combine(const float* expert_out_host,
                            const int* source_token_host,
                            const int* source_k_host,
                            const float* route_weights_host,
                            float* y_host,
                            int num_routes,
                            int hidden_size,
                            int num_tokens,
                            int top_k) {
    const size_t out_size = static_cast<size_t>(num_routes) * hidden_size;
    const size_t y_size = static_cast<size_t>(num_tokens) * hidden_size;
    const size_t meta_size = static_cast<size_t>(num_routes) * sizeof(int);
    const size_t rw_size = static_cast<size_t>(num_tokens) * top_k * sizeof(float);

    float *d_out = nullptr, *d_y = nullptr, *d_rw = nullptr;
    int *d_src_t = nullptr, *d_src_k = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, y_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rw, rw_size));
    CUDA_CHECK(cudaMalloc(&d_src_t, meta_size));
    CUDA_CHECK(cudaMalloc(&d_src_k, meta_size));

    CUDA_CHECK(cudaMemcpyAsync(d_out, expert_out_host, out_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y, y_host, y_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_rw, route_weights_host, rw_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_src_t, source_token_host, meta_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_src_k, source_k_host, meta_size, cudaMemcpyHostToDevice, stream));

    moe_combine_launch_cuda(stream, d_out, d_src_t, d_src_k, d_rw, d_y, num_routes, hidden_size, top_k);

    CUDA_CHECK(cudaMemcpyAsync(y_host, d_y, y_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_rw));
    CUDA_CHECK(cudaFree(d_src_t));
    CUDA_CHECK(cudaFree(d_src_k));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
