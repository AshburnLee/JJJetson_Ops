#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include "cuda_fp16.h"
#include "cuda_utils.cuh"

// 与 fa_kernel_one_pass_parallel 对齐：grid (16,1,1)，block (32,4,1)，HEAD_DIM=128，Q 行数 13，KV tile 32，GQA blockIdx.x/2。
// 
// QK 子问题太小，喂不饱 Tensor Core
// TC 路径里 只有 2 个 warp 做 WMMA，另外 2 个 warp 在 GEMM 阶段基本在等
// MMA 条数少、启动与同步开销占比大，峰值算力很难拉起来。
// 
// TC 不是免费的，仅仅改用tc 指令，会变慢，要 围绕 TC 重做数据流、并行划分和流水线
__global__ void fa_kernel_one_pass_parallel_tc(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ dst,
    const float scale) {
    using namespace nvcuda::wmma;

    constexpr int HEAD_DIM = 128;
    constexpr int TOKENS_PER_Q = 13;
    constexpr int QHEAD_PER_BLOCK = 1;
    constexpr int ROWS = TOKENS_PER_Q * QHEAD_PER_BLOCK;
    constexpr int KV_TOKEN_TILE = 32;
    constexpr int LOOP_KV = 256 / KV_TOKEN_TILE;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    __shared__ half q_shared[ROWS][HEAD_DIM];
    int tid_blockwise = threadIdx.x + blockDim.x * threadIdx.y;
    int warp_id = tid_blockwise / 32;
    int lane_id = tid_blockwise % 32;

    const half* qhead_elem_start = Q + blockIdx.x * TOKENS_PER_Q * HEAD_DIM;
    int half_head = warp_id % 2;
    int row_start = half_head * 64;
    int s_col0_id = lane_id * 2 + row_start;

#pragma unroll
    for (int token_id = 0; token_id < TOKENS_PER_Q; token_id++) {
        if (warp_id < 2) {
            const half2* row_h2 =
                reinterpret_cast<const half2*>(qhead_elem_start + token_id * HEAD_DIM);
            half2 data = row_h2[s_col0_id / 2];
            *reinterpret_cast<half2*>(&q_shared[token_id][s_col0_id]) = data;
        }
    }
    __syncthreads();

    __shared__ half k_shared[KV_TOKEN_TILE][HEAD_DIM + 2];
    __shared__ half v_shared[KV_TOKEN_TILE][HEAD_DIM + 2];
    __shared__ float s_shared[ROWS][KV_TOKEN_TILE];
    // WMMA 输出 16×16，临时落盘再写回 13×32，避免 s_shared[13][32] 越界。
    __shared__ float s_tc_st[WMMA_M][KV_TOKEN_TILE];
    __shared__ float dst_shared[ROWS][HEAD_DIM];
    __shared__ float stream_num_scale[ROWS];
    __shared__ float stream_l_new[ROWS];

    // WMMA 辅助缓冲：Q 行数 13 需垫到 16；K^T 为 128×32 供 matrix_b col_major 连续列访问。
    __shared__ alignas(16) half q_pad[WMMA_M][HEAD_DIM];
    __shared__ alignas(16) half kt[HEAD_DIM][KV_TOKEN_TILE];

    const int kv_head_block_offset = blockIdx.x / 2;
    const int warp_row_id = warp_id / 2;

    __shared__ float m[ROWS];
    __shared__ float l[ROWS];

    if (threadIdx.x < ROWS) {
        m[threadIdx.x] = -INFINITY;
        l[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    int WARPS = blockDim.y;
    int ROWS_PER_WARP = (ROWS + WARPS - 1) / WARPS;
    int block_tid = threadIdx.x + blockDim.x * threadIdx.y;

    for (int i = block_tid; i < ROWS * HEAD_DIM; i += HEAD_DIM) {
        int row = i / 128;
        int col = i % 128;
        dst_shared[row][col] = 0.0f;
    }
    __syncthreads();

    for (int tile_id = 0; tile_id < LOOP_KV; tile_id++) {
        for (int token_id = 0; token_id < KV_TOKEN_TILE; token_id += 2) {
            int col_warp_id = warp_row_id + token_id;
            const half* row_base = K + col_warp_id * HEAD_DIM + tile_id * KV_TOKEN_TILE * HEAD_DIM +
                                   kv_head_block_offset * 256 * HEAD_DIM;
            const half2* row_h2 = reinterpret_cast<const half2*>(row_base);
            half2 data = row_h2[s_col0_id / 2];
            *reinterpret_cast<half2*>(&k_shared[col_warp_id][s_col0_id]) = data;
        }
        __syncthreads();

        // --- 构造 WMMA 输入 ---
        for (int t = block_tid; t < WMMA_M * HEAD_DIM; t += blockDim.x * blockDim.y) {
            int pr = t / HEAD_DIM;
            int pc = t % HEAD_DIM;
            q_pad[pr][pc] = (pr < ROWS) ? q_shared[pr][pc] : half(0);
        }
        for (int t = block_tid; t < HEAD_DIM * KV_TOKEN_TILE; t += blockDim.x * blockDim.y) {
            int h = t / KV_TOKEN_TILE;
            int j = t % KV_TOKEN_TILE;
            kt[h][j] = k_shared[j][h];
        }
        __syncthreads();

        /*
         * Tensor Core（WMMA m16n16k16）用于 S = Q_pad @ K^T 的 K=128 维缩并。
         * 瓶颈：CUDA Core 路径对每个 (q 行, k 列) 要做 128 次半精度乘加，指令数与延迟高；
         *       TC 在单条 MMA 内完成 16×16×16 的乘加阵列，显著提高算力上限、减轻指令与发射压力。
         * 划分：warp0 负责 S 的列 0..15，warp1 负责列 16..31；warp2/3 不参与 GEMM，后续参与 softmax。
         */
        if (warp_id == 0 || warp_id == 1) {
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            fill_fragment(c_frag, 0.0f);
            const int n_col0 = (warp_id == 0) ? 0 : WMMA_N;

            for (int k_step = 0; k_step < HEAD_DIM / WMMA_K; ++k_step) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                // kt[k][n] 为 C 行主序，K 维沿行连续，故 matrix_b 用 row_major，ldb=32。
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
                load_matrix_sync(a_frag, &q_pad[0][k_step * WMMA_K], HEAD_DIM);
                load_matrix_sync(b_frag, &kt[k_step * WMMA_K][n_col0], KV_TOKEN_TILE);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            store_matrix_sync(&s_tc_st[0][n_col0], c_frag, KV_TOKEN_TILE, mem_row_major);
        }
        __syncthreads();
        for (int t = block_tid; t < ROWS * KV_TOKEN_TILE; t += blockDim.x * blockDim.y) {
            int sr = t / KV_TOKEN_TILE;
            int sc = t % KV_TOKEN_TILE;
            s_shared[sr][sc] = s_tc_st[sr][sc];
        }
        __syncthreads();

        int w_id = threadIdx.y;
        int l_id = threadIdx.x;

        int row_start_w = w_id * ROWS_PER_WARP;
        int row_end_w = min(row_start_w + ROWS_PER_WARP, ROWS);

        for (int r = row_start_w; r < row_end_w; ++r) {
            const float s_val = s_shared[r][l_id];
            float row_max = warp_reduce_xor_max(s_val);
            float exp_val = expf(s_val - row_max);
            float row_sum = warp_reduce_xor_sum(exp_val);
            float scale_new = 0.0f;
            if (l_id == 0) {
                const float m_old = m[r];
                const float l_old = l[r];
                const float m_new = fmaxf(m_old, row_max);
                const float scale_old = expf(m_old - m_new);
                scale_new = expf(row_max - m_new);
                const float l_new = l_old * scale_old + row_sum * scale_new;
                stream_num_scale[r] = l_old * scale_old;
                stream_l_new[r] = l_new;
                m[r] = m_new;
                l[r] = l_new;
            }
            scale_new = __shfl_sync(0xFFFFFFFFu, scale_new, 0);
            s_shared[r][l_id] = exp_val * scale_new;
        }
        __syncthreads();

        for (int token_id = 0; token_id < KV_TOKEN_TILE; token_id += 2) {
            int col_warp_id = warp_row_id + token_id;
            const half* row_base = V + col_warp_id * HEAD_DIM + tile_id * KV_TOKEN_TILE * HEAD_DIM +
                                   kv_head_block_offset * 256 * HEAD_DIM;
            const half2* row_h2 = reinterpret_cast<const half2*>(row_base);
            half2 data = row_h2[s_col0_id / 2];
            *reinterpret_cast<half2*>(&v_shared[col_warp_id][s_col0_id]) = data;
        }
        __syncthreads();

        /*
         * P@V：内维仅 32，且权重经 streaming softmax 为 FP32；此处沿用 FMA 与参考 kernel 一致，
         * 避免再量化到 FP16 做 WMMA 带来的与标量路径的数值差异。
         */
        const int col = block_tid;
        float v_reg[KV_TOKEN_TILE];
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            v_reg[k] = __half2float(v_shared[k][col]);
        }
        for (int dst_row = 0; dst_row < ROWS; ++dst_row) {
            float sum_pv = 0.0f;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                sum_pv += s_shared[dst_row][k] * v_reg[k];
            }
            const float l_new_r = stream_l_new[dst_row];
            dst_shared[dst_row][col] =
                (stream_num_scale[dst_row] * dst_shared[dst_row][col] + sum_pv) / l_new_r;
        }
        __syncthreads();
    }

    if (block_tid < HEAD_DIM) {
        for (int r = 0; r < ROWS; ++r) {
            const int token_id = r;
            const int qhead_global = blockIdx.x;
            if (qhead_global < 16) {
                const int hd = block_tid;
                const int dst_id = hd + HEAD_DIM * token_id + HEAD_DIM * TOKENS_PER_Q * qhead_global;
                dst[dst_id] = dst_shared[r][hd];
            }
        }
    }
    __syncthreads();
    (void)scale;
}

extern "C" void fa_one_pass_parallel_tc(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* dst_host,
    float scale) {

    using half_t = half;
    constexpr int HEAD_DIM = 128;
    constexpr int TOKENS_PER_Q = 13;
    constexpr int Q_HEADS = 16;
    constexpr int KV_TOKENS = 256;
    constexpr int KV_HEADS = 8;

    const size_t q_elems = static_cast<size_t>(HEAD_DIM) * TOKENS_PER_Q * Q_HEADS;
    const size_t kv_elems = static_cast<size_t>(HEAD_DIM) * KV_TOKENS * KV_HEADS;
    const size_t dst_elems = q_elems;

    half_t* d_q = nullptr;
    half_t* d_k = nullptr;
    half_t* d_v = nullptr;
    float* d_dst = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_q, q_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_k, kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_v, kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, dst_elems * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_q, q_host, q_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_k, k_host, kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v, v_host, kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));

    dim3 threads(32, 4, 1);
    dim3 blocks(16, 1, 1);

    fa_kernel_one_pass_parallel_tc<<<blocks, threads, 0, stream>>>(d_q, d_k, d_v, d_dst, scale);
    LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, dst_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeAsync(d_q, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_k, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_v, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_dst, nullptr));
}
