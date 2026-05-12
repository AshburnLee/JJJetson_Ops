#include <cuda_runtime.h>
#include <cstring> // memcpy
#include <stdio.h>
#include <vector>
#include <stdexcept>      // std::runtime_error
#include "cuda_utils.cuh" // CUDA_CHECK

#define CUDA_ROPE_BLOCK_SIZE 256
#define WARP_SIZE 32

template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit,
                                     const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    // 计算每个thread vals 中3个值的max
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = lane + i * WARP_SIZE;
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
        const int idx = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i] = val;
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
        const int idx = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

/*
输入shape是：logits 1024 个tokens(=n_rows)，128 个expert，topk=4=n_expert_used，
block (32,4,1), grid(256,1,1), kernel 做的事情：

1. 1024 行并行：grid 上 256 个 block × 每 block 4 行，block 的 一个warp（一行 thread） 负责一个token
的128个expert，
2. 每行一个 warp（32 线程） 把 128 个 logits 按 4 个一组摊到 32 个 lane，
3. 经 4 轮 warp 内比较 + shuffle 选出 top-4，再写 ids/weights，
4. 最后对结果进行归一化
*/
template <int n_experts, bool with_norm, bool delayed_softmax = false>
__launch_bounds__(WARP_SIZE * 4, 1) __global__
    void moe_top_k_kernel(const float *logits, float *weights, int *ids, const int n_rows,
                          const int top_k, const float clamp_val) {
    const int row_id = threadIdx.y + blockDim.y * blockIdx.x;
    if (row_id >= n_rows) {
        return;
    }

    // 定位 block 中这一行 thread 在输入输出的位置
    logits += row_id * n_experts;
    weights += row_id * top_k;
    ids += row_id * top_k;

    // 每个 thread 负责的 expert 数量。一个token的一行（128个expert）由block一行thread（32个）负责，
    // 所以每一个thread 负责128/32=4 个logits
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    float thread_hold_logits[experts_per_thread]; // 开辟 4 个临时空间，分散存储，避免race condition

    // 每个 thread 读取自己负责的 4 个 logits
    // lane0:
    // thread_hold_logits[0] <- logits[0 + 0]
    // thread_hold_logits[1] <- logits[0 + 32]
    // thread_hold_logits[2] <- logits[0 + 64]
    // thread_hold_logits[3] <- logits[0 + 96]
    // lane1:
    // thread_hold_logits[0] <- logits[1 + 0]
    // thread_hold_logits[1] <- logits[1 + 32]
    // thread_hold_logits[2] <- logits[1 + 64]
    // thread_hold_logits[3] <- logits[1 + 96]
    // ...
    // lane31:
    // thread_hold_logits[0] <- logits[31 + 0]
    // thread_hold_logits[1] <- logits[31 + 32]
    // thread_hold_logits[2] <- logits[31 + 64]
    // thread_hold_logits[3] <- logits[31 + 96]
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert_id = threadIdx.x + i;
        thread_hold_logits[i / WARP_SIZE] =
            (n_experts % WARP_SIZE == 0 || expert_id < n_experts) ? logits[expert_id] : -INFINITY;
    }

    float wt_sum = 0.f;
    // 缓存起来的结果，分散存储，避免race condition
    // 这里每个 thread out_wt 大小是4，正好覆盖 32*4=128 个expert，当top-k=128 时，依然可以存下
    // 这128个值！
    float out_wt[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        out_wt[i] = 0.f; // 初始化为0，避免未定义的数值对计算的影响
    }
    // top-k, 故循环 k 次
    for (int k = 0; k < top_k; ++k) {
        // 找当前 WARP 内局部最大值
        float max_val = thread_hold_logits[0];
        int max_expert_id = threadIdx.x;

#pragma unroll
        // 先循环找到每个 thread 负责4个expert中的最大值，（结束后32个thread 各自HOLD部分最大值）
        for (int i = 1; i < experts_per_thread; i++) {
            // i = 1, expert_id = threadIdx.x + 32
            // i = 2, expert_id = threadIdx.x + 64
            // i = 3, expert_id = threadIdx.x + 96
            const int expert_id = threadIdx.x + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert_id < n_experts) &&
                thread_hold_logits[i] > max_val) {
                max_val = thread_hold_logits[i];
                max_expert_id = expert_id;
            }
        }
#pragma unroll
        // 然后在找到32个thread中的最大值。
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            const float val = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int expert_id = __shfl_xor_sync(0xFFFFFFFF, max_expert_id, mask, WARP_SIZE);
            // 如果两个max相同，则选id小的
            if (val > max_val || (val == max_val && expert_id < max_expert_id)) {
                max_val = val;
                max_expert_id = expert_id;
            }
        }

        // 至此，得到了 128 个专家中的最大值，max_val 是分数，max_expert_id 是对应的索引。
        // 找到一个最大值后，将其写入对应的寄存器 out_wt
        // 这里的if 条件的设计是：top-i 的那个值写入 lane id=i 的out_w中：
        // k=0 (找 top-1)  =>   lane 0  (threadIdx.x==0)  执行写入
        //         写自己的 out_wt[0] , 本轮全局 max（第 1 大的 logit）
        // k=1 (找 top-2)  =>   lane 1  (threadIdx.x==1)  执行写入
        //         写自己的 out_wt[0] , 本轮全局 max（第 2 大）
        // k=2 (找 top-3)  =>   lane 2  (threadIdx.x==2)
        //         写自己的 out_wt[0] , 第 3 大
        // k=3 (找 top-4)  =>   lane 3  (threadIdx.x==3)
        //         写自己的 out_wt[0] , 第 4 大
        if ((k & (WARP_SIZE - 1) /*= (k % WARP_SIZE)*/) == threadIdx.x) {
            out_wt[k / WARP_SIZE] = max_val; // 每次循环k, max_val 写入的位置都不一样
        }

        // 只有拥有这个 expert 的那条 lane 才能把 thread_hold_logits[] 置成 -inf，其他不能变
        // 根据expert的划分，编号为 e 的 expert 一定落在：
        // lane：e % 32（即是 e & 31）
        // 该 lane 上的槽位：wt[e / 32]
        if ((max_expert_id & (WARP_SIZE - 1)) == threadIdx.x) {
            thread_hold_logits[max_expert_id / WARP_SIZE] = -INFINITY;
            ids[k] = max_expert_id;

            if constexpr (with_norm) {
                wt_sum += max_val;
            }
        }
        // printf("max id: %d, max val: %f \n", max_expert_id, max_val);
    }
    // 归一化选中的 k 个权重
    if constexpr (with_norm) {
        wt_sum = warp_reduce_xor_sum(wt_sum);
        wt_sum = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;
        for (int i = 0; i < experts_per_thread; i++) {
            out_wt[i] *= inv_sum;
        }
    }
    // 至此，out_wt 中存放了topk个logits只是，分别存在于不同lane的 out_wt 中，如：
    // 第 k 大的logit 写在 lane k 的 out_wt[0], 即 4 个数值分布在 4 个不同线程的寄存器里

    if constexpr (delayed_softmax) {
        // 这里的softmax 不是在“连续的存储上做的”，warp-level reduce是warp 的 32 条 lane
        // 各自寄存器里的 vals[] 上，拼成一条固定长度（这里长度是n_expert_used）的向量，然后用__shfl
        // 指令 在 lane之间交换信息
        softmax_warp_inplace<experts_per_thread, true>(out_wt, top_k, threadIdx.x);
    }
#pragma unroll
    // 写回最终的 k 个 gate weight（已归一化或 softmax）
    for (int i = 0; i < experts_per_thread; i++) {
        // 注意，这里的Idx与i有关
        const int idx = threadIdx.x + i * WARP_SIZE;
        if (idx < top_k) { // 只写回top-k个位置
            weights[idx] = out_wt[i];
        }
    }

    if (!with_norm) {
        (void)clamp_val;
    }
}

extern "C" void moe_top_k(const float *logits, const int topk, float *weights, int *ids,
                          const std::vector<int> &input_dims) {
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
    const int64_t out_size = n_tokens * topk;

    float *d_digits = nullptr;
    float *d_weight = nullptr;
    int *d_ids = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_digits, input_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_weight, out_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ids, out_size * sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_digits, logits, input_size * sizeof(float), cudaMemcpyHostToDevice,
                               stream));

    const int rows_per_block = 4;
    dim3 blocks((n_tokens + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3 threads(WARP_SIZE, rows_per_block, 1);
#if defined(MY_OPS_DEBUG)
    std::printf("Kernel launch config: block=(%u,%u,%u), grid=(%u,%u,%u)\n", threads.x, threads.y,
                threads.z, blocks.x, blocks.y, blocks.z);
    std::fflush(stdout);
#endif
    switch (n_experts) {
    case 2:
        moe_top_k_kernel<2, false, true><<<blocks, threads, 0, stream>>>(
            d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
        LAUNCH_CHECK();
        break;
    case 4:
        moe_top_k_kernel<4, false, true><<<blocks, threads, 0, stream>>>(
            d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
        LAUNCH_CHECK();
        break;
    case 8:
        moe_top_k_kernel<8, false, true><<<blocks, threads, 0, stream>>>(
            d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
        LAUNCH_CHECK();
        break;
    case 16:
        moe_top_k_kernel<16, false, true><<<blocks, threads, 0, stream>>>(
            d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
        LAUNCH_CHECK();
        break;
    case 32:
        moe_top_k_kernel<32, false, true><<<blocks, threads, 0, stream>>>(
            d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
        LAUNCH_CHECK();
        break;
    case 64:
        moe_top_k_kernel<64, false, true><<<blocks, threads, 0, stream>>>(
            d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
        LAUNCH_CHECK();
        break;
    case 128:
        moe_top_k_kernel<128, false, true><<<blocks, threads, 0, stream>>>(
            d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
        LAUNCH_CHECK();
        break;
    case 256:
        moe_top_k_kernel<256, false, true><<<blocks, threads, 0, stream>>>(
            d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
        LAUNCH_CHECK();
        break;
    case 512:
        moe_top_k_kernel<512, false, true><<<blocks, threads, 0, stream>>>(
            d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
        LAUNCH_CHECK();
        break;
    case 1024:
        moe_top_k_kernel<1024, false, true><<<blocks, threads, 0, stream>>>(
            d_digits, d_weight, d_ids, (int)n_tokens, topk, clamp_val);
        LAUNCH_CHECK();
        break;
    default:
        throw std::runtime_error("unsupported n_experts");
    }

    CUDA_CHECK(cudaMemcpyAsync(weights, d_weight, out_size * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(ids, d_ids, out_size * sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeAsync(d_digits, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_weight, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_ids, nullptr));
}
