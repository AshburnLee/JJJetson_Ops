// Full MoE pipeline (sota): top-k -> dispatch_sota -> grouped GEMM x3 -> SwiGLU -> combine_sota
#include "moe_pipeline_sota.h"
#include "cuda_utils.cuh"

extern "C" void moe_top_k_launch_cuda(cudaStream_t stream, const float *d_logits, float *d_weights,
                                      int *d_expert_ids, int num_tokens, int num_experts,
                                      int top_k);

extern "C" size_t moe_dispatch_sota_sort_workspace_bytes(int num_tokens, int top_k,
                                                         int num_experts);

extern "C" void moe_dispatch_sota_launch_cuda(
    cudaStream_t stream, const float *d_x, const int *d_expert_ids, int num_tokens, int top_k,
    int hidden_size, int num_experts, void *d_sort_workspace, size_t sort_workspace_bytes,
    int *d_sorted_expert_ids, int *d_expanded_src_for_dest_row, float *d_permuted_x,
    int *d_source_token, int *d_source_k, int *d_expert_offsets, int *d_arange_buf);

extern "C" void moe_combine_sota_build_inv_permuted_idx_launch_cuda(cudaStream_t stream,
                                                                    const int *d_source_token,
                                                                    const int *d_source_k,
                                                                    int *d_inv_permuted_idx,
                                                                    int num_routes, int top_k);

extern "C" void moe_combine_sota_launch_cuda(cudaStream_t stream, const float *d_expert_out,
                                             const int *d_inv_permuted_idx,
                                             const float *d_route_weights, float *d_y,
                                             int num_tokens, int hidden_size, int top_k);

namespace {

static __device__ __forceinline__ float silu_f(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

__device__ int expert_id_for_permuted_row(const int *expert_offsets, int num_experts, int row) {
    int lo = 0;
    int hi = num_experts - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (expert_offsets[mid] <= row) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// Grouped GEMM (朴素 dot-product)
__global__ void moe_grouped_gemm_nt_kernel(const float *permuted_x, const float *weight, float *out,
                                           const int *expert_offsets, int num_experts, int num_rows,
                                           int in_dim, int out_dim) {
    const int row = blockIdx.x;
    if (row >= num_rows) {
        return;
    }
    const int expert_e = expert_id_for_permuted_row(expert_offsets, num_experts, row);
    const float *x_row = permuted_x + static_cast<size_t>(row) * in_dim;
    const float *w_base = weight + static_cast<size_t>(expert_e) * out_dim * in_dim;

    for (int n = threadIdx.x; n < out_dim; n += blockDim.x) {
        float acc = 0.f;
        const float *w_row = w_base + static_cast<size_t>(n) * in_dim;
        for (int k = 0; k < in_dim; ++k) {
            acc += x_row[k] * w_row[k];
        }
        out[static_cast<size_t>(row) * out_dim + n] = acc;
    }
}

static void moe_grouped_gemm_nt_launch_cuda(cudaStream_t stream, const float *d_permuted_x,
                                            const float *d_weight, float *d_out,
                                            const int *d_expert_offsets, int num_experts,
                                            int num_rows, int in_dim, int out_dim) {
    const int threads = 256;
    moe_grouped_gemm_nt_kernel<<<num_rows, threads, 0, stream>>>(
        d_permuted_x, d_weight, d_out, d_expert_offsets, num_experts, num_rows, in_dim, out_dim);
    LAUNCH_CHECK();
}

__global__ void moe_swiglu_silu_mul_kernel(const float *gate, const float *up, float *mid,
                                           int n_elem) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elem) {
        mid[i] = silu_f(gate[i]) * up[i];
    }
}

} // namespace

extern "C" MoePipelineSotaBuffers *moe_pipeline_sota_buffers_create(int num_tokens, int hidden_size,
                                                                    int intermediate_size,
                                                                    int num_experts, int top_k) {
    const int H = hidden_size;
    const int I = intermediate_size;
    const int R = num_tokens * top_k;

    auto *buffers = new MoePipelineSotaBuffers{};
    buffers->num_tokens = num_tokens;
    buffers->hidden_size = H;
    buffers->intermediate_size = I;
    buffers->num_experts = num_experts;
    buffers->top_k = top_k;

    const size_t x_bytes = static_cast<size_t>(num_tokens) * H * sizeof(float);
    const size_t logits_bytes = static_cast<size_t>(num_tokens) * num_experts * sizeof(float);
    const size_t route_float_bytes = static_cast<size_t>(num_tokens) * top_k * sizeof(float);
    const size_t route_int_bytes = static_cast<size_t>(R) * sizeof(int);
    const size_t w_gu_bytes = static_cast<size_t>(num_experts) * I * H * sizeof(float);
    const size_t w_d_bytes = static_cast<size_t>(num_experts) * H * I * sizeof(float);
    const size_t ri_bytes = static_cast<size_t>(R) * I * sizeof(float);
    const size_t expert_out_bytes = static_cast<size_t>(R) * H * sizeof(float);
    const size_t off_bytes = static_cast<size_t>(num_experts + 1) * sizeof(int);

    buffers->sort_workspace_bytes =
        moe_dispatch_sota_sort_workspace_bytes(num_tokens, top_k, num_experts);

    CUDA_CHECK(cudaMalloc(&buffers->d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_logits, logits_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_y, x_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_w_gate, w_gu_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_w_up, w_gu_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_w_down, w_d_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_permuted_x, expert_out_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_gate, ri_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_up, ri_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_mid, ri_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_expert_out, expert_out_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_route_weights, route_float_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_expert_ids, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_sorted_expert_ids, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_expanded_src_for_dest_row, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_source_token, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_source_k, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_expert_offsets, off_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_arange_buf, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_inv_permuted_idx, route_int_bytes));
    CUDA_CHECK(cudaMalloc(&buffers->d_sort_workspace, buffers->sort_workspace_bytes));

    return buffers;
}

extern "C" void moe_pipeline_sota_buffers_destroy(MoePipelineSotaBuffers *buffers) {
    if (buffers == nullptr) {
        return;
    }

    cudaFree(buffers->d_x);
    cudaFree(buffers->d_logits);
    cudaFree(buffers->d_y);
    cudaFree(buffers->d_w_gate);
    cudaFree(buffers->d_w_up);
    cudaFree(buffers->d_w_down);
    cudaFree(buffers->d_permuted_x);
    cudaFree(buffers->d_gate);
    cudaFree(buffers->d_up);
    cudaFree(buffers->d_mid);
    cudaFree(buffers->d_expert_out);
    cudaFree(buffers->d_route_weights);
    cudaFree(buffers->d_expert_ids);
    cudaFree(buffers->d_sorted_expert_ids);
    cudaFree(buffers->d_expanded_src_for_dest_row);
    cudaFree(buffers->d_source_token);
    cudaFree(buffers->d_source_k);
    cudaFree(buffers->d_expert_offsets);
    cudaFree(buffers->d_arange_buf);
    cudaFree(buffers->d_inv_permuted_idx);
    cudaFree(buffers->d_sort_workspace);

    delete buffers;
}

// 7 步骤的完整 moe，用于eager模式、Graph capture、Graph Replay
extern "C" void moe_pipeline_sota_forward_device(void *stream, MoePipelineSotaBuffers *buffers) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int num_tokens = buffers->num_tokens;
    const int H = buffers->hidden_size;
    const int I = buffers->intermediate_size;
    const int num_experts = buffers->num_experts;
    const int top_k = buffers->top_k;
    const int R = num_tokens * top_k;

    moe_top_k_launch_cuda(s, buffers->d_logits, buffers->d_route_weights, buffers->d_expert_ids,
                          num_tokens, num_experts, top_k);

    moe_dispatch_sota_launch_cuda(
        s, buffers->d_x, buffers->d_expert_ids, num_tokens, top_k, H, num_experts,
        buffers->d_sort_workspace, buffers->sort_workspace_bytes, buffers->d_sorted_expert_ids,
        buffers->d_expanded_src_for_dest_row, buffers->d_permuted_x, buffers->d_source_token,
        buffers->d_source_k, buffers->d_expert_offsets, buffers->d_arange_buf);

    moe_grouped_gemm_nt_launch_cuda(s, buffers->d_permuted_x, buffers->d_w_gate, buffers->d_gate,
                                    buffers->d_expert_offsets, num_experts, R, H, I);
    moe_grouped_gemm_nt_launch_cuda(s, buffers->d_permuted_x, buffers->d_w_up, buffers->d_up,
                                    buffers->d_expert_offsets, num_experts, R, H, I);

    const int silu_threads = 256;
    const int silu_blocks = (R * I + silu_threads - 1) / silu_threads;
    moe_swiglu_silu_mul_kernel<<<silu_blocks, silu_threads, 0, s>>>(buffers->d_gate, buffers->d_up,
                                                                    buffers->d_mid, R * I);
    LAUNCH_CHECK();

    moe_grouped_gemm_nt_launch_cuda(s, buffers->d_mid, buffers->d_w_down, buffers->d_expert_out,
                                    buffers->d_expert_offsets, num_experts, R, I, H);

    moe_combine_sota_build_inv_permuted_idx_launch_cuda(
        s, buffers->d_source_token, buffers->d_source_k, buffers->d_inv_permuted_idx, R, top_k);
    moe_combine_sota_launch_cuda(s, buffers->d_expert_out, buffers->d_inv_permuted_idx,
                                 buffers->d_route_weights, buffers->d_y, num_tokens, H, top_k);
}

// eager 模式入口
extern "C" void moe_pipeline_sota_forward(const float *x_host, const float *logits_host,
                                          const float *w_gate_host, const float *w_up_host,
                                          const float *w_down_host, float *y_host, int num_tokens,
                                          int hidden_size, int intermediate_size, int num_experts,
                                          int top_k, void *stream_in) {
    const bool own_stream = (stream_in == nullptr);
    cudaStream_t s = own_stream ? nullptr : static_cast<cudaStream_t>(stream_in);
    if (own_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }

    MoePipelineSotaBuffers *buffers = moe_pipeline_sota_buffers_create(
        num_tokens, hidden_size, intermediate_size, num_experts, top_k);

    const size_t x_bytes = static_cast<size_t>(num_tokens) * hidden_size * sizeof(float);
    const size_t logits_bytes = static_cast<size_t>(num_tokens) * num_experts * sizeof(float);
    const size_t w_gu_bytes =
        static_cast<size_t>(num_experts) * intermediate_size * hidden_size * sizeof(float);
    const size_t w_d_bytes =
        static_cast<size_t>(num_experts) * hidden_size * intermediate_size * sizeof(float);

    CUDA_CHECK(cudaMemcpyAsync(buffers->d_x, x_host, x_bytes, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(
        cudaMemcpyAsync(buffers->d_logits, logits_host, logits_bytes, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(
        cudaMemcpyAsync(buffers->d_w_gate, w_gate_host, w_gu_bytes, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(buffers->d_w_up, w_up_host, w_gu_bytes, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(
        cudaMemcpyAsync(buffers->d_w_down, w_down_host, w_d_bytes, cudaMemcpyHostToDevice, s));

    moe_pipeline_sota_forward_device(static_cast<void *>(s), buffers);

    CUDA_CHECK(cudaMemcpyAsync(y_host, buffers->d_y, x_bytes, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    moe_pipeline_sota_buffers_destroy(buffers);

    if (own_stream) {
        CUDA_CHECK(cudaStreamDestroy(s));
    }
}
