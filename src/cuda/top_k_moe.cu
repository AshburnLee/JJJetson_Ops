#include <cuda_runtime.h>
#include <cstring>       // memcpy
#include <stdio.h>
#include <vector>
#include <stdexcept>   // std::runtime_error
#include "cuda_utils.cuh"  // CUDA_CHECK

#define CUDA_ROPE_BLOCK_SIZE 256
#define WARP_SIZE 32

template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    // 计算每个thread vals 中3个值的max
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_xor_max(max_val);

    float sum = 0.f;

#pragma unroll
    // 计算每个 thread vals 中3个值的的 expf(vals-max_val)
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }
    // 将所有 thread 各自的sum 集合起来，求和得全局sum, 并将这个 Global sum 广播给每一个thread
    sum = warp_reduce_xor_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int n_experts, bool with_norm, bool delayed_softmax = false>
__launch_bounds__ (WARP_SIZE * 4, 1)
__global__ void top_k_moe_kernel(const float* logits,
                                 float* weights,
                                 int* ids,
                                 const int n_rows,
                                 const int n_expert_used,
                                 const float clamp_val
) {
    const int row_id = threadIdx.y + blockDim.y * blockIdx.x;
    if (row_id >= n_rows) {
        return;
    }

    // 定位 block 中这一行 thread 在输入输出的位置
    logits  += row_id * n_experts;
    weights += row_id * n_expert_used;
    ids     += row_id * n_expert_used;

    // 每个 thread 负责的 expert 数量。96>32, 则这个值是3
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    float wt[experts_per_thread];  // 开辟 3 个临时空间，分散存储，避免race condition

    // 每个 thread 读取自己负责的 几个 logits
#pragma unroll 
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert_id = threadIdx.x + i;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert_id < n_experts) ? logits[expert_id] : -INFINITY;
    }

    float wt_sum = 0.f;
    float out_wt[experts_per_thread]; // 分散存储，避免race condition

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        out_wt[i] = 0.f; // 初始化为0，避免未定义的数值对计算的影响
    }
    // top-k, 故循环 k 次
    for (int k = 0; k < n_expert_used; ++k) {
        // 找当前 WARP 内局部最大值
        float max_val       = wt[0];
        int   max_expert_id = threadIdx.x;

#pragma unroll
        // 先循环找到每个 thread 负责专家组中的最大值
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert_id = threadIdx.x + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert_id < n_experts) && wt[i] > max_val) {
                max_val       = wt[i];
                max_expert_id = expert_id;
            }
        }
#pragma unroll
        // warp level reduce 得到 warp 内32个最大值的最大值
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val     = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int expert_id = __shfl_xor_sync(0xFFFFFFFF, max_expert_id, mask, WARP_SIZE);
            if (val > max_val || (val == max_val && expert_id < max_expert_id)) {
                max_val       = val;
                max_expert_id = expert_id;
            }
        }

        // 至此，得到了 96 个专家中的最大值，max_val 是分数，max_expert_id 是对应的索引。
        // 找到一个最大值后，将其写入新的寄存器（不是wt，wt中是每个thread 负责的原始expert 值）
        if ((k & (WARP_SIZE - 1)/*= (k % WARP_SIZE)*/) == threadIdx.x) {
            out_wt[k / WARP_SIZE] = max_val;  // 每次循环k, max_val 写入的位置都不一样
        }

        // 找到一个 top-1 后，把它分数设为 -inf，进入下一次循环找 top-2
        if ((max_expert_id & (WARP_SIZE - 1)) == threadIdx.x) {
            wt[max_expert_id / WARP_SIZE] = -INFINITY;
            ids[k] = max_expert_id;

            if constexpr (with_norm) {
                wt_sum += max_val;
            }
        }
        // printf("max id: %d, max val: %f \n", max_expert_id, max_val);
    }
    // 归一化选中的 k 个权重
    if constexpr (with_norm) {
        wt_sum              = warp_reduce_xor_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;
        for (int i = 0; i < experts_per_thread; i++) {
            out_wt[i] *= inv_sum;
        }
    }

    if constexpr (delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(out_wt, n_expert_used, threadIdx.x);
    }
#pragma unroll
    // 写回最终的 k 个 gate weight（已归一化或 softmax）
    for (int i = 0; i < experts_per_thread; i++) {
        // 注意，这里的Idx与i有关
        const int idx = threadIdx.x + i * WARP_SIZE;
        if (idx < n_expert_used) { // 只写回top-k个位置
            weights[idx] = out_wt[i];
        }
    }

    if (!with_norm) {
        (void)clamp_val;
    }
}

extern "C" void top_k_moe(const float* logits, 
                          const int topk, 
                          float* weights, 
                          int* ids, 
                          const std::vector<int>& input_dims) {
    /*
    logits: 
    input_dims:  n_experts * n_tokens
    topk:  int val < n_experts

    output:
    weights:     topk * n_tokens
    ids:         topk * n_tokens
    */
    // python 传入的是 row-major 的
    const float clamp_val = 1e-8f;
    const int64_t n_tokens = input_dims[0];
    const int64_t n_experts = input_dims[1];

    const int64_t input_size = n_tokens * n_experts;
    const int64_t out_size   = n_tokens * topk;

    float *d_digits = nullptr;
    float *d_weight = nullptr;
    int *d_ids      = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_digits, input_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_weight, out_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ids, out_size * sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_digits, logits, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));

    const int rows_per_block = 4;
    dim3 blocks((n_tokens + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3 threads(WARP_SIZE, rows_per_block, 1);
#if defined(MY_OPS_DEBUG)
    std::printf(
        "Kernel launch config: block=(%u,%u,%u), grid=(%u,%u,%u)\n",
        threads.x, threads.y, threads.z,
        blocks.x, blocks.y, blocks.z);
    std::fflush(stdout);
#endif
    switch (n_experts) {
        case 2: 
            top_k_moe_kernel<2,false,true><<<blocks, threads, 0, stream>>>(d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
            LAUNCH_CHECK();
            break;
        case 4: 
            top_k_moe_kernel<4,false,true><<<blocks, threads, 0, stream>>>(d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
            LAUNCH_CHECK();
            break;
        case 8: 
            top_k_moe_kernel<8,false,true><<<blocks, threads, 0, stream>>>(d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
            LAUNCH_CHECK();
            break;
        case 16: 
            top_k_moe_kernel<16,false,true><<<blocks, threads, 0, stream>>>(d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
            LAUNCH_CHECK();
            break;
        case 32: 
            top_k_moe_kernel<32,false,true><<<blocks, threads, 0, stream>>>(d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
            LAUNCH_CHECK();
            break;
        case 64: 
            top_k_moe_kernel<64,false,true><<<blocks, threads, 0, stream>>>(d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
            LAUNCH_CHECK();
            break;
        case 128: 
            top_k_moe_kernel<128,false,true><<<blocks, threads, 0, stream>>>(d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
            LAUNCH_CHECK();
            break;
        case 256: 
            top_k_moe_kernel<256,false,true><<<blocks, threads, 0, stream>>>(d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
            LAUNCH_CHECK();
            break;
        case 512: 
            top_k_moe_kernel<512,false,true><<<blocks, threads, 0, stream>>>(d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
            LAUNCH_CHECK();
            break;
        case 1024: 
            top_k_moe_kernel<1024,false,true><<<blocks, threads, 0, stream>>>(d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
            LAUNCH_CHECK();
            break;
        default: throw std::runtime_error("unsupported n_experts");
    }

    CUDA_CHECK(cudaMemcpyAsync(weights, d_weight, out_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(ids, d_ids, out_size * sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeAsync(d_digits, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_weight, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_ids, nullptr));
}
