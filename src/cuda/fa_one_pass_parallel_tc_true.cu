#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include "cuda_fp16.h"
#include "cuda_utils.cuh"

// 使用 Tensor core计算两段 mma

namespace {

constexpr int HEAD_DIM = 128;
constexpr int TOKENS_PER_Q = 13;
constexpr int Q_HEADS = 16;
constexpr int KV_TOKENS = 256;
constexpr int KV_HEADS = 8;
constexpr int KV_TOKEN_TILE = 32;
constexpr int LOOP_KV = KV_TOKENS / KV_TOKEN_TILE; // 8

constexpr int ROWS_TWO_HEADS = TOKENS_PER_Q * 2;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_ROWS = 32;
constexpr int N_TILES = HEAD_DIM / WMMA_N; // 8
// padding shared 步长，减轻 Wmma load/store 的列方向 stride 对 TMMA/对齐的要求；另见下 s_scores_f。
// leading dim（half 个数）须为 8 的倍数以满足 WMMA 16B 行基地址对齐
constexpr int Q_STRIDE = HEAD_DIM + 8;
constexpr int KV_STRIDE = KV_TOKEN_TILE + 8;
constexpr int V_STRIDE = HEAD_DIM + 8;

}  // namespace

__global__ void __launch_bounds__(256, 4) fa_kernel_one_pass_parallel_tc_true(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ dst,
    const float scale) {
    using namespace nvcuda::wmma;

    const int block_threads = blockDim.x * blockDim.y;
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int warp_id = tid / 32;
    const int num_warps = block_threads / 32; // 8
    
    // q tile （26+6）x（128+8）
    __shared__ alignas(16) half q_shared[WMMA_ROWS][Q_STRIDE];
    // k tile 和 v tile 共享一块shared 32x(128+8)
    __shared__ alignas(16) half kv_shared[KV_TOKEN_TILE][V_STRIDE];
    // QK 输出 S, softmax 后变为 P, 32x(32+8)
    __shared__ alignas(16) half s_scores[WMMA_ROWS][KV_STRIDE];
    // 输出累加, 最后写 global, 26x128
    __shared__ alignas(16) float dst_acc[ROWS_TWO_HEADS][HEAD_DIM];
    __shared__ float stream_num_scale[ROWS_TWO_HEADS];
    __shared__ float m[ROWS_TWO_HEADS];
    __shared__ float l[ROWS_TWO_HEADS];

    // softmax@V 的结果
    __shared__ alignas(16) float pv_acc[WMMA_ROWS][HEAD_DIM];

    const int kv_h = blockIdx.x;
    const int q0 = kv_h * 2;
    const int q1 = q0 + 1;

    const half* q0_base = Q + static_cast<size_t>(q0) * TOKENS_PER_Q * HEAD_DIM;
    const half* q1_base = Q + static_cast<size_t>(q1) * TOKENS_PER_Q * HEAD_DIM;
    const size_t kv_plane_elems = static_cast<size_t>(KV_TOKENS) * HEAD_DIM;

    constexpr int Q_HALF2_ONE = TOKENS_PER_Q * (HEAD_DIM / 2);
    // load Q 到 Q Tile 中
    // 把Q head 1 load到shared中
    for (int i = tid; i < Q_HALF2_ONE; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        const int j2 = i % (HEAD_DIM / 2);
        const half2* src = reinterpret_cast<const half2*>(q0_base + row * HEAD_DIM);
        reinterpret_cast<half2*>(&q_shared[row][0])[j2] = src[j2];
    }
    // 把Q head2 load到shared中
    for (int i = tid; i < Q_HALF2_ONE; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        const int j2 = i % (HEAD_DIM / 2);
        const int dr = row + TOKENS_PER_Q;
        const half2* src = reinterpret_cast<const half2*>(q1_base + row * HEAD_DIM);
        reinterpret_cast<half2*>(&q_shared[dr][0])[j2] = src[j2];
    }
    // padding 的部分填数值 0
    for (int t = tid; t < (WMMA_ROWS - ROWS_TWO_HEADS) * HEAD_DIM; t += block_threads) {
        const int r = ROWS_TWO_HEADS + t / HEAD_DIM;
        const int c = t % HEAD_DIM;
        q_shared[r][c] = half(0);
    }
    __syncthreads();

    for (int i = tid; i < ROWS_TWO_HEADS; i += block_threads) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
    __syncthreads();

    for (int t = tid; t < ROWS_TWO_HEADS * HEAD_DIM; t += block_threads) {
        const int r = t / HEAD_DIM;
        const int c = t % HEAD_DIM;
        dst_acc[r][c] = 0.0f;
    }
    __syncthreads();

    const int rows_per_warp = (ROWS_TWO_HEADS + num_warps - 1) / num_warps;
    const half* k_head = K + static_cast<size_t>(kv_h) * kv_plane_elems;
    const half* v_head = V + static_cast<size_t>(kv_h) * kv_plane_elems;

    constexpr int K_VEC = KV_TOKEN_TILE * (HEAD_DIM / 2);

    for (int tile_id = 0; tile_id < LOOP_KV; ++tile_id) {
        const half* k_tile_base = k_head + static_cast<size_t>(tile_id * KV_TOKEN_TILE) * HEAD_DIM;
        // load K 到 K Tile 中
        for (int i = tid; i < K_VEC; i += block_threads) {
            const int row = i / (HEAD_DIM / 2);
            const int j2 = i % (HEAD_DIM / 2);
            const half2* src_h2 =
                reinterpret_cast<const half2*>(k_tile_base + row * HEAD_DIM);
            reinterpret_cast<half2*>(&kv_shared[row][0])[j2] = src_h2[j2];
        }
        __syncthreads();

        // WMMA 第一段：S = Q * K^T。K 在 kv_shared 为row-major[seq][d]；B 用 col_major 从同一缓冲加载，
        // 避免了转置。输出 32×32 half，warp 0..3 负责 2×2 个 16×16 象限
        /*  s_scores:
        *             列 0..15     列 16..31
        *           +-------------+-------------+
        *   行 0..15|   warp 0    |   warp 1    |
        *           +-------------+-------------+
        *  行 16..31|   warp 2    |   warp 3    |
        *           +-------------+-------------+
        */
        if (warp_id < 4) {
            // 本 warp 负责 S 上哪个 16×16 角：行块 warp_m、列块 warp_n
            const int warp_m = warp_id / 2;  // 0,1,0,1
            const int warp_n = warp_id % 2;  // 0,0,1,1
            const int row0 = warp_m * WMMA_M;  // Q 行起点 0 或 16:   0,16,0,16
            const int col0 = warp_n * WMMA_N;  // K^T 列起点 0 或 16: 0,0,16,16

            // c_frag：half 累加器，存当前象限的 S 子块 16×16
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
            fill_fragment(c_frag, half(0));
            // 沿 HEAD_DIM 分 8 段，每段 K=16，做一次 m16n16k16，
            // 顺序执行8次，后8次结果acc
            for (int k_step = 0; k_step < HEAD_DIM / WMMA_K/*128/16=8*/; ++k_step) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
                load_matrix_sync(a_frag, &q_shared[row0][k_step * WMMA_K], Q_STRIDE);
                load_matrix_sync(b_frag, &kv_shared[col0][k_step * WMMA_K], V_STRIDE);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            // c_frag 既是输入累加器，也是输出，每次计算结果都在上一次结果上继续累加
            store_matrix_sync(&s_scores[row0][col0], c_frag, KV_STRIDE, mem_row_major);
        }
        __syncthreads();

        // 计算softmax
        const int w_soft = threadIdx.y;
        const int l_id = threadIdx.x;
        const int row_start_w = w_soft * rows_per_warp;
        const int row_end_w = min(row_start_w + rows_per_warp, ROWS_TWO_HEADS);

        for (int r = row_start_w; r < row_end_w; ++r) {
            const float s_val = __half2float(s_scores[r][l_id]);
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
                m[r] = m_new;
                l[r] = l_new;
            }
            scale_new = __shfl_sync(0xFFFFFFFFu, scale_new, 0);
            const float upd = exp_val * scale_new;
            // softmax 结果 写入 s_scores (复用)
            s_scores[r][l_id] = __float2half_rn(upd);
        }
        __syncthreads();

        // load V 到 V tile 中（K 和 V 共享同一个 tile）
        const half* v_tile_base = v_head + static_cast<size_t>(tile_id * KV_TOKEN_TILE) * HEAD_DIM;
        for (int i = tid; i < K_VEC; i += block_threads) {
            const int row = i / (HEAD_DIM / 2);
            const int j2 = i % (HEAD_DIM / 2);
            const half2* src_h2 =
                reinterpret_cast<const half2*>(v_tile_base + row * HEAD_DIM);
            reinterpret_cast<half2*>(&kv_shared[row][0])[j2] = src_h2[j2];
        }
        __syncthreads();

        // WMMA #2：O_tile = PV，累加到 pv_acc（逻辑 32×128 FP32）。P=s_scores（32×32），V=kv_shared（32×128）
        /*
         *            0..15    16..31   32..47  ...  112..127
         *         +--------+--------+--------+ ... +--------+
         * 行 0..15| warp0  | warp1  | warp2  | ... | warp7  |  : m_tile=0，P 的上半
         *         +--------+--------+--------+ ... +--------+
         *行 16..31| warp0  | warp1  | warp2  | ... | warp7  |  : m_tile=1，P 的下半
         *         +--------+--------+--------+ ... +--------+
         *
         *  内层 k_step=0,1：沿 KV token 维 32 = 2×16 顺序累加（每 (m_tile,n_tile) 共 2 次 mma，非并行）
         */
        for (int n_tile = warp_id; n_tile < N_TILES/*=8*/; n_tile += num_warps/*=8*/) {
            // 本 warp：n_tile 等于 warp_id
            for (int m_tile = 0; m_tile < WMMA_ROWS / WMMA_M/*=32/16=2*/; ++m_tile) {
                const int row0 = m_tile * WMMA_M;  // s_scores 行起点: 0,16

                // acc：FP32 累加器，当前 (m_tile,n_tile) 对应输出子块 16×16
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
                fill_fragment(acc, 0.0f);
                // 沿 KV token 32 分 2 段，每段 K=16，顺序 2 次 mma_sync，结果累加进 acc
                for (int k_step = 0; k_step < KV_TOKEN_TILE / WMMA_K/*32/16=2*/; ++k_step) {
                    // a_frag：P 子块，来自 s_scores；b_frag：V 子块，来自 kv_shared
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
                    load_matrix_sync(a_frag, &s_scores[row0][k_step * WMMA_K], KV_STRIDE);
                    load_matrix_sync(b_frag, &kv_shared[k_step * WMMA_K][n_tile * WMMA_N], V_STRIDE);
                    mma_sync(acc, a_frag, b_frag, acc);
                }
                // 该 16×16 子块写入 pv_acc
                store_matrix_sync(&pv_acc[row0][n_tile * WMMA_N], acc, HEAD_DIM, mem_row_major);
            }
        }
        __syncthreads();

        constexpr int OUT_VEC = ROWS_TWO_HEADS * HEAD_DIM;
        for (int t = tid; t < OUT_VEC; t += block_threads) {
            const int r = t / HEAD_DIM;
            const int c = t % HEAD_DIM;
            const float l_new_r = l[r];
            dst_acc[r][c] = (stream_num_scale[r] * dst_acc[r][c] + pv_acc[r][c]) / l_new_r;
        }
        __syncthreads();
    }

    for (int t = tid; t < TOKENS_PER_Q * HEAD_DIM; t += block_threads) {
        const int r = t / HEAD_DIM;
        const int c = t % HEAD_DIM;
        if (q0 < Q_HEADS) {
            const int dst_id = c + HEAD_DIM * r + HEAD_DIM * TOKENS_PER_Q * q0;
            dst[dst_id] = dst_acc[r][c];
        }
    }
    for (int t = tid; t < TOKENS_PER_Q * HEAD_DIM; t += block_threads) {
        const int r = t / HEAD_DIM;
        const int c = t % HEAD_DIM;
        const int dr = r + TOKENS_PER_Q;
        if (q1 < Q_HEADS) {
            const int dst_id = c + HEAD_DIM * r + HEAD_DIM * TOKENS_PER_Q * q1;
            dst[dst_id] = dst_acc[dr][c];
        }
    }
    __syncthreads();
    (void)scale;
}

extern "C" void fa_one_pass_parallel_tc_true(
                            const uint16_t* q_host,
                            const uint16_t* k_host,
                            const uint16_t* v_host,
                            float* dst_host,
                            float scale) {
    // q: (head_dim, n_token, n_qhead)    = (128,13,16)
    // kv: (head_dim, n_token, n_kv_head) = (128,256,8)

    using half_t = half;

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

    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void*)fa_kernel_one_pass_parallel_tc_true));
    const size_t static_shmem = static_cast<size_t>(attr.sharedSizeBytes);
    constexpr int kDefaultShmemPerBlock = 48 * 1024;

    int max_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    if (static_shmem > static_cast<size_t>(kDefaultShmemPerBlock)) {
        const int excess = static_cast<int>(static_shmem - static_cast<size_t>(kDefaultShmemPerBlock));
        if (excess > 0) {
            if (max_optin < static_cast<int>(static_shmem)) {
                std::fprintf(stderr,
                             "[fa_one_pass_parallel_tc_true] static shared %zu B exceeds "
                             "cudaDevAttrMaxSharedMemoryPerBlockOptin=%d; launch may fail.\n",
                             static_shmem, max_optin);
            }
            CUDA_CHECK(cudaFuncSetAttribute(
                (void*)fa_kernel_one_pass_parallel_tc_true,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                excess));
        }
    }

    dim3 threads(32, 8, 1);
    dim3 blocks(KV_HEADS, 1, 1);  // (8,1,1) KV head只被读一次

    fa_kernel_one_pass_parallel_tc_true<<<blocks, threads, 0, stream>>>(d_q, d_k, d_v, d_dst, scale);
    LAUNCH_CHECK();

    CUDA_CHECK(
        cudaMemcpyAsync(dst_host, d_dst, dst_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeAsync(d_q, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_k, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_v, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_dst, nullptr));
}
