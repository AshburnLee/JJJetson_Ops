// 使用双缓冲：
// 当前 tile 的 V cp.async 与 QK WMMA 异步，隐藏 load V
// 下一 tile 的 K prefetch 与 softmax 异步，隐藏 load K
// 两次 wait 后做 WMMA PV

#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include "cuda_fp16.h"
#include "cuda_utils.cuh"

namespace fa_db {

constexpr int HEAD_DIM = 128;
constexpr int TOKENS_PER_Q = 13;
constexpr int Q_HEADS = 16;
constexpr int KV_TOKENS = 256;
constexpr int KV_HEADS = 8;
constexpr int KV_TOKEN_TILE = 32;
constexpr int LOOP_KV = KV_TOKENS / KV_TOKEN_TILE;

constexpr int ROWS_TWO_HEADS = TOKENS_PER_Q * 2;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_ROWS = 32;
constexpr int N_TILES = HEAD_DIM / WMMA_N;
constexpr int Q_STRIDE = HEAD_DIM + 8;
constexpr int KV_STRIDE = KV_TOKEN_TILE + 8;
constexpr int V_STRIDE = HEAD_DIM + 8;
constexpr int ACC_STRIDE = HEAD_DIM + 8;

constexpr int KV_TILE_NUM_HALF = KV_TOKEN_TILE * HEAD_DIM;

// 将 K/V tile从 gmem 搬运到 smem，非异步地
__device__ void db_sync_copy_kv_tile(half (*dst_rowmajor)[V_STRIDE], const half *g_tile_base,
                                     int tid, int block_threads) {
    constexpr int K_VEC = KV_TOKEN_TILE * (HEAD_DIM / 2);
    for (int i = tid; i < K_VEC; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        const int j2 = i % (HEAD_DIM / 2);
        const half2 *src_h2 = reinterpret_cast<const half2 *>(g_tile_base + row * HEAD_DIM);
        reinterpret_cast<half2 *>(&dst_rowmajor[row][0])[j2] = src_h2[j2];
    }
}

// cp.asyn 是Ampere 才有的 features
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// 在 device函数中插入 一条 PTX，
// 把之前 issue 的 cp.async 划成一组，之后用 cp.async.wait_group 对这一组一起做完成等待
__device__ __forceinline__ void db_cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// 与上配合使用
template <int N> __device__ __forceinline__ void db_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// 发起异步拷贝动作，以 16B 为粒度将整个 tile 从 gmem 搬运到 smem
__device__ void db_issue_async_kv_tile(half (*dst_rowmajor)[V_STRIDE], const half *g_tile_base,
                                       int tid, int block_threads) {
#pragma unroll 1
    for (int i = tid; i < KV_TILE_NUM_HALF / 8; i += block_threads) {
        const int flat8 = i * 8;
        const int r = flat8 / HEAD_DIM;
        const int c = flat8 % HEAD_DIM;
        half *dst = &dst_rowmajor[r][c];
        const half *src = g_tile_base + static_cast<size_t>(r) * HEAD_DIM + c;
        uint32_t sm_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sm_addr), "l"(src));
    }
}

#endif
// 先 cp.async 再 commit 然后 wait 0
__device__ void db_prologue_load_k_tile(half (*k_buf)[KV_TOKEN_TILE][V_STRIDE],
                                        const half *k_tile_base, int tid, int block_threads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    db_issue_async_kv_tile(k_buf[0], k_tile_base, tid, block_threads);
    db_cp_async_commit_group();
    db_cp_async_wait<0>();
#else
    db_sync_copy_kv_tile(k_buf[0], k_tile_base, tid, block_threads);
#endif
}

// 真实的异步 load
__device__ void db_begin_copy_kv_tile(half (*dst_rowmajor)[V_STRIDE], const half *g_tile_base,
                                      int tid, int block_threads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    db_issue_async_kv_tile(dst_rowmajor, g_tile_base, tid, block_threads);
    db_cp_async_commit_group();
#else
    db_sync_copy_kv_tile(dst_rowmajor, g_tile_base, tid, block_threads);
#endif
}

} // namespace fa_db

__global__ void __launch_bounds__(256, 4)
    fa_kernel_one_pass_parallel_double_buffer(const half *__restrict__ Q,
                                              const half *__restrict__ K,
                                              const half *__restrict__ V, float *__restrict__ dst,
                                              const float scale) {
    using namespace fa_db;
    using namespace nvcuda::wmma;

    const int block_threads = blockDim.x * blockDim.y;
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int warp_id = tid / 32;
    const int num_warps = block_threads / 32;

    __shared__ alignas(16) half q_shared[WMMA_ROWS][Q_STRIDE];
    // K/V 双 Double buffer：与优化前单块 kv_shared 不同，便于 K_{t+1} 与 V_t 并行存在
    __shared__ alignas(16) half k_double_buf[2][KV_TOKEN_TILE][V_STRIDE];
    __shared__ alignas(16) half v_double_buf[2][KV_TOKEN_TILE][V_STRIDE];
    __shared__ alignas(16) half s_scores[WMMA_ROWS][KV_STRIDE];
    __shared__ alignas(16) float dst_acc[ROWS_TWO_HEADS][ACC_STRIDE];
    __shared__ float stream_num_scale[ROWS_TWO_HEADS];
    __shared__ float m[ROWS_TWO_HEADS];
    __shared__ float l[ROWS_TWO_HEADS];
    __shared__ alignas(16) float pv_acc[WMMA_ROWS][ACC_STRIDE];

    const int kv_h = blockIdx.x;
    const int q0 = kv_h * 2;
    const int q1 = q0 + 1;

    const half *q0_base = Q + static_cast<size_t>(q0) * TOKENS_PER_Q * HEAD_DIM;
    const half *q1_base = Q + static_cast<size_t>(q1) * TOKENS_PER_Q * HEAD_DIM;
    const size_t kv_plane_elems = static_cast<size_t>(KV_TOKENS) * HEAD_DIM;

    constexpr int Q_HALF2_ONE = TOKENS_PER_Q * (HEAD_DIM / 2);

    for (int i = tid; i < Q_HALF2_ONE; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        const int j2 = i % (HEAD_DIM / 2);
        const half2 *src = reinterpret_cast<const half2 *>(q0_base + row * HEAD_DIM);
        reinterpret_cast<half2 *>(&q_shared[row][0])[j2] = src[j2];
    }
    for (int i = tid; i < Q_HALF2_ONE; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        const int j2 = i % (HEAD_DIM / 2);
        const int dr = row + TOKENS_PER_Q;
        const half2 *src = reinterpret_cast<const half2 *>(q1_base + row * HEAD_DIM);
        reinterpret_cast<half2 *>(&q_shared[dr][0])[j2] = src[j2];
    }
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
    const half *k_head = K + static_cast<size_t>(kv_h) * kv_plane_elems;
    const half *v_head = V + static_cast<size_t>(kv_h) * kv_plane_elems;

    const half *k0_ptr = k_head;
    // 异步 load 后马上同步，因wait就在其中，结束后smem中已load完成，形式上是async，实际上是sync
    db_prologue_load_k_tile(k_double_buf, k0_ptr, tid, block_threads);
    __syncthreads();

    for (int tile_id = 0; tile_id < LOOP_KV; ++tile_id) {
        const int cb = tile_id & 1; // 0 or 1, pingpong
        half(*k_active)[V_STRIDE] = k_double_buf[cb];

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if (tile_id > 0) {
            db_cp_async_wait<0>();
        }
#endif
        __syncthreads();

        const half *v_tile_base = v_head + static_cast<size_t>(tile_id * KV_TOKEN_TILE) * HEAD_DIM;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // load V，与 QK WMMA 异步
        db_begin_copy_kv_tile(v_double_buf[cb], v_tile_base, tid, block_threads);
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
        db_sync_copy_kv_tile(v_double_buf[cb], v_tile_base, tid, block_threads);
#endif

        // WMMA QK
        if (warp_id < 4) {
            const int warp_m = warp_id / 2;
            const int warp_n = warp_id % 2;
            const int row0 = warp_m * WMMA_M;
            const int col0 = warp_n * WMMA_N;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
            fill_fragment(c_frag, half(0));
            for (int k_step = 0; k_step < HEAD_DIM / WMMA_K; ++k_step) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
                load_matrix_sync(a_frag, &q_shared[row0][k_step * WMMA_K], Q_STRIDE);
                load_matrix_sync(b_frag, &k_active[col0][k_step * WMMA_K], V_STRIDE);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            store_matrix_sync(&s_scores[row0][col0], c_frag, KV_STRIDE, mem_row_major);
        }
        __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // 下一 tile 的 K 预取，写入 k_double_buf[cb^1]，与 softmax 重叠
        if (tile_id + 1 < LOOP_KV) {
            const int nb = cb ^ 1;
            const half *k_next =
                k_head + static_cast<size_t>((tile_id + 1) * KV_TOKEN_TILE) * HEAD_DIM;
            db_begin_copy_kv_tile(k_double_buf[nb], k_next, tid, block_threads);
        }
#endif

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
            s_scores[r][l_id] = __float2half_rn(upd);
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // load V 和 WMMA QK；load K(t+1) 和 softmax，两次 commit 后一次 wait
        db_cp_async_wait<0>();
#endif
        __syncthreads();

        half(*v_active)[V_STRIDE] = v_double_buf[cb];

        // WMMA PV
        for (int n_tile = warp_id; n_tile < N_TILES; n_tile += num_warps) {
            for (int m_tile = 0; m_tile < WMMA_ROWS / WMMA_M; ++m_tile) {
                const int row0 = m_tile * WMMA_M;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
                fill_fragment(acc, 0.0f);
                for (int k_step = 0; k_step < KV_TOKEN_TILE / WMMA_K; ++k_step) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
                    load_matrix_sync(a_frag, &s_scores[row0][k_step * WMMA_K], KV_STRIDE);
                    load_matrix_sync(b_frag, &v_active[k_step * WMMA_K][n_tile * WMMA_N], V_STRIDE);
                    mma_sync(acc, a_frag, b_frag, acc);
                }
                store_matrix_sync(&pv_acc[row0][n_tile * WMMA_N], acc, ACC_STRIDE, mem_row_major);
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

extern "C" void fa_one_pass_parallel_double_buffer(const uint16_t *q_host, const uint16_t *k_host,
                                                   const uint16_t *v_host, float *dst_host,
                                                   float scale) {
    using namespace fa_db;
    using half_t = half;

    const size_t q_elems = static_cast<size_t>(HEAD_DIM) * TOKENS_PER_Q * Q_HEADS;
    const size_t kv_elems = static_cast<size_t>(HEAD_DIM) * KV_TOKENS * KV_HEADS;
    const size_t dst_elems = q_elems;

    half_t *d_q = nullptr;
    half_t *d_k = nullptr;
    half_t *d_v = nullptr;
    float *d_dst = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_q, q_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_k, kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_v, kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, dst_elems * sizeof(float), stream));

    CUDA_CHECK(
        cudaMemcpyAsync(d_q, q_host, q_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_k, k_host, kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_v, v_host, kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));

    cudaFuncAttributes attr{};
    CUDA_CHECK(
        cudaFuncGetAttributes(&attr, (const void *)fa_kernel_one_pass_parallel_double_buffer));
    const size_t static_shmem = static_cast<size_t>(attr.sharedSizeBytes);
    constexpr int kDefaultShmemPerBlock = 48 * 1024;

    int max_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    if (static_shmem > static_cast<size_t>(kDefaultShmemPerBlock)) {
        const int excess =
            static_cast<int>(static_shmem - static_cast<size_t>(kDefaultShmemPerBlock));
        if (excess > 0) {
            if (max_optin < static_cast<int>(static_shmem)) {
                std::fprintf(stderr,
                             "[fa_one_pass_parallel_double_buffer] static shared %zu B exceeds "
                             "cudaDevAttrMaxSharedMemoryPerBlockOptin=%d; launch may fail.\n",
                             static_shmem, max_optin);
            }
            CUDA_CHECK(cudaFuncSetAttribute((void *)fa_kernel_one_pass_parallel_double_buffer,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, excess));
        }
    }

    dim3 threads(32, 8, 1);
    dim3 blocks(KV_HEADS, 1, 1);

    fa_kernel_one_pass_parallel_double_buffer<<<blocks, threads, 0, stream>>>(d_q, d_k, d_v, d_dst,
                                                                              scale);
    LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, dst_elems * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeAsync(d_q, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_k, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_v, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_dst, nullptr));
}
