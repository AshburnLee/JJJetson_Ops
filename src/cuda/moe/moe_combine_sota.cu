#include <cuda_runtime.h>
#include "cuda_utils.cuh"

namespace {

__global__ void moe_combine_sota_build_inv_permuted_idx_kernel(const int *source_token,
                                                               const int *source_k,
                                                               int *inv_permuted_idx,
                                                               int num_routes, int top_k) {
    const int permuted_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (permuted_row >= num_routes) {
        return;
    }
    const int token_t = source_token[permuted_row];
    const int slot_k = source_k[permuted_row];
    const int route_r = token_t * top_k + slot_k;
    inv_permuted_idx[route_r] = permuted_row;
}

__global__ void moe_combine_sota_unpermute_kernel(const float *expert_out,
                                                  const int *inv_permuted_idx,
                                                  const float *route_weights, float *y,
                                                  int num_tokens, int hidden_size, int top_k) {
    // 每一个 TOKEN 一个 CUDA block
    const int token_t = blockIdx.x;
    if (token_t >= num_tokens) {
        return;
    }

    float *y_row = y + static_cast<size_t>(token_t) * hidden_size;
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float acc = 0.f;
        for (int slot_k = 0; slot_k < top_k; ++slot_k) {
            const int route_r = token_t * top_k + slot_k;
            const int permuted_row = inv_permuted_idx[route_r];
            const float w = route_weights[route_r];
            const float *expert_row = expert_out + static_cast<size_t>(permuted_row) * hidden_size;
            acc += w * expert_row[h];
        }
        y_row[h] = acc;
    }
}

} // namespace

extern "C" void moe_combine_sota_build_inv_permuted_idx_launch_cuda(cudaStream_t stream,
                                                                    const int *d_source_token,
                                                                    const int *d_source_k,
                                                                    int *d_inv_permuted_idx,
                                                                    int num_routes, int top_k) {
    const int threads = 256;
    const int blocks = (num_routes + threads - 1) / threads;
    moe_combine_sota_build_inv_permuted_idx_kernel<<<blocks, threads, 0, stream>>>(
        d_source_token, d_source_k, d_inv_permuted_idx, num_routes, top_k);
    LAUNCH_CHECK();
}

extern "C" void moe_combine_sota_launch_cuda(cudaStream_t stream, const float *d_expert_out,
                                             const int *d_inv_permuted_idx,
                                             const float *d_route_weights, float *d_y,
                                             int num_tokens, int hidden_size, int top_k) {
    const int unpermute_threads = 256;
    moe_combine_sota_unpermute_kernel<<<num_tokens, unpermute_threads, 0, stream>>>(
        d_expert_out, d_inv_permuted_idx, d_route_weights, d_y, num_tokens, hidden_size, top_k);
    LAUNCH_CHECK();
}

// 推理场景下禁止使用
extern "C" void moe_combine_sota(const float *expert_out_host, const int *source_token_host,
                                 const int *source_k_host, const float *route_weights_host,
                                 float *y_host, int num_routes, int hidden_size, int num_tokens,
                                 int top_k) {
    const size_t expert_out_bytes = static_cast<size_t>(num_routes) * hidden_size * sizeof(float);
    const size_t y_bytes = static_cast<size_t>(num_tokens) * hidden_size * sizeof(float);
    const size_t route_meta_bytes = static_cast<size_t>(num_routes) * sizeof(int);
    const size_t route_weight_bytes = static_cast<size_t>(num_tokens) * top_k * sizeof(float);

    float *d_expert_out = nullptr;
    float *d_y = nullptr;
    float *d_route_weights = nullptr;
    int *d_source_token = nullptr;
    int *d_source_k = nullptr;
    int *d_inv_permuted_idx = nullptr;

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMalloc(&d_expert_out, expert_out_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, y_bytes));
    CUDA_CHECK(cudaMalloc(&d_route_weights, route_weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_source_token, route_meta_bytes));
    CUDA_CHECK(cudaMalloc(&d_source_k, route_meta_bytes));
    CUDA_CHECK(cudaMalloc(&d_inv_permuted_idx, route_meta_bytes));

    CUDA_CHECK(cudaMemcpyAsync(d_expert_out, expert_out_host, expert_out_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_route_weights, route_weights_host, route_weight_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_source_token, source_token_host, route_meta_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_source_k, source_k_host, route_meta_bytes, cudaMemcpyHostToDevice,
                               stream));

    moe_combine_sota_build_inv_permuted_idx_launch_cuda(stream, d_source_token, d_source_k,
                                                        d_inv_permuted_idx, num_routes, top_k);
    moe_combine_sota_launch_cuda(stream, d_expert_out, d_inv_permuted_idx, d_route_weights, d_y,
                                 num_tokens, hidden_size, top_k);

    CUDA_CHECK(cudaMemcpyAsync(y_host, d_y, y_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_expert_out));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_route_weights));
    CUDA_CHECK(cudaFree(d_source_token));
    CUDA_CHECK(cudaFree(d_source_k));
    CUDA_CHECK(cudaFree(d_inv_permuted_idx));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
