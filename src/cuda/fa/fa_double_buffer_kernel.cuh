#pragma once

#include <cstdio>

#include <cuda_fp16.h>
#include <mma.h>

#include "cuda_utils.cuh"
#include "fa.h"

namespace fa_db {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWmmaRows = 32;
constexpr int kKvTokenTile = 32;
constexpr int kMaxQTokensPerBlock = kWmmaRows / 2; // 每 block 2 个 Q head

template <int HEAD_DIM> struct HeadConsts {
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kQStride = HEAD_DIM + 8;
    static constexpr int kKvStride = kKvTokenTile + 8;
    static constexpr int kVStride = HEAD_DIM + 8;
    static constexpr int kAccStride = HEAD_DIM + 8;
    static constexpr int kNTiles = HEAD_DIM / kWmmaN;
    static constexpr int kKvTileNumHalf = kKvTokenTile * HEAD_DIM;
};

struct FaDoubleBufferKernelParams {
    int num_q_tokens;
    int num_kv_tokens;
    int num_q_heads;
    int num_kv_heads;
    float scale;
    int causal;
    int q_pos_offset;
};

// 本 tile 有几行是真 KV。例：num_kv_tokens=4、tile=32、tile_id=0 -> 4；
// num_kv_tokens=37、tile_id=1 -> 5。多出来的行只在 smem 里填 0，不要去 gmem 读。
__device__ __forceinline__ int db_kv_tile_valid_rows(int tile_id, int num_kv_tokens) {
    const int t0 = tile_id * kKvTokenTile;
    const int remain = num_kv_tokens - t0;
    if (remain <= 0) {
        return 0;
    }
    return remain < kKvTokenTile ? remain : kKvTokenTile;
}

// 末 tile 不满 32 时，把 smem 里多出来的行写成 0。
// 例：valid_rows=4，行 4..31 清零。softmax 仍会把这些列打成 -inf；
// 清零是为了 WMMA 不要把 OOB/NaN 的 K 混进合法列的 score。
template <int HEAD_DIM>
__device__ void db_zero_kv_tile_pad(half (*dst_rowmajor)[HeadConsts<HEAD_DIM>::kVStride],
                                    int valid_rows, int tid, int block_threads) {
    constexpr int kVec = kKvTokenTile * (HEAD_DIM / 2);
    for (int i = tid; i < kVec; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        if (row >= valid_rows) {
            const int j2 = i % (HEAD_DIM / 2);
            reinterpret_cast<half2 *>(&dst_rowmajor[row][0])[j2] = __half2half2(half(0));
        }
    }
}

// 将 K/V tile 从 gmem 搬运到 smem，非异步。只读 valid_rows 行，其余填 0。
template <int HEAD_DIM>
__device__ void db_sync_copy_kv_tile(half (*dst_rowmajor)[HeadConsts<HEAD_DIM>::kVStride],
                                     const half *g_tile_base, int valid_rows, int tid,
                                     int block_threads) {
    constexpr int kVec = kKvTokenTile * (HEAD_DIM / 2);
    for (int i = tid; i < kVec; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        const int j2 = i % (HEAD_DIM / 2);
        if (row < valid_rows) {
            const half2 *src_h2 = reinterpret_cast<const half2 *>(g_tile_base + row * HEAD_DIM);
            reinterpret_cast<half2 *>(&dst_rowmajor[row][0])[j2] = src_h2[j2];
        } else {
            reinterpret_cast<half2 *>(&dst_rowmajor[row][0])[j2] = __half2half2(half(0));
        }
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template <int N> __device__ __forceinline__ void db_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void db_cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int HEAD_DIM>
__device__ void db_issue_async_kv_tile(half (*dst_rowmajor)[HeadConsts<HEAD_DIM>::kVStride],
                                       const half *g_tile_base, int valid_rows, int tid,
                                       int block_threads) {
    constexpr int kFlatChunks = HeadConsts<HEAD_DIM>::kKvTileNumHalf / 8;
#pragma unroll 1
    for (int i = tid; i < kFlatChunks; i += block_threads) {
        const int flat8 = i * 8;
        const int r = flat8 / HEAD_DIM;
        if (r >= valid_rows) {
            continue;
        }
        const int c = flat8 % HEAD_DIM;
        half *dst = &dst_rowmajor[r][c];
        const half *src = g_tile_base + static_cast<size_t>(r) * HEAD_DIM + c;
        const uint32_t sm_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sm_addr), "l"(src));
    }
}

#endif

template <int HEAD_DIM>
__device__ void db_prologue_load_k_tile(half (*k_buf)[kKvTokenTile][HeadConsts<HEAD_DIM>::kVStride],
                                        const half *k_tile_base, int valid_rows, int tid,
                                        int block_threads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    db_issue_async_kv_tile<HEAD_DIM>(k_buf[0], k_tile_base, valid_rows, tid, block_threads);
    db_cp_async_commit_group();
    db_cp_async_wait<0>();
    db_zero_kv_tile_pad<HEAD_DIM>(k_buf[0], valid_rows, tid, block_threads);
#else
    db_sync_copy_kv_tile<HEAD_DIM>(k_buf[0], k_tile_base, valid_rows, tid, block_threads);
#endif
}

template <int HEAD_DIM>
__device__ void db_begin_copy_kv_tile(half (*dst_rowmajor)[HeadConsts<HEAD_DIM>::kVStride],
                                      const half *g_tile_base, int valid_rows, int tid,
                                      int block_threads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    db_issue_async_kv_tile<HEAD_DIM>(dst_rowmajor, g_tile_base, valid_rows, tid, block_threads);
    db_cp_async_commit_group();
#else
    db_sync_copy_kv_tile<HEAD_DIM>(dst_rowmajor, g_tile_base, valid_rows, tid, block_threads);
#endif
}

template <int HEAD_DIM>
__global__ void __launch_bounds__(256, 4)
    fa_kernel_double_buffer(const half *__restrict__ Q, const half *__restrict__ K,
                            const half *__restrict__ V, float *__restrict__ dst,
                            FaDoubleBufferKernelParams params) {
    using namespace nvcuda::wmma;
    using C = HeadConsts<HEAD_DIM>;

    const int num_q_tokens = params.num_q_tokens;
    const int num_kv_tokens = params.num_kv_tokens;
    const int num_q_heads = params.num_q_heads;
    const int num_kv_heads = params.num_kv_heads;
    // WMMA、softmax、双缓冲这些 完全不知道 g。它们只看到：shared 里两行 Q、一份 K/V tile
    // 所以该 kernel 结构几乎没变

    // 每 block 仍算 2 个 Q。g=q/kv 为偶数。
    // 例 g=2、blockIdx.x=3 -> pairs=1, kv_h=3, pair=0, q0=6, q1=7（与旧 kv_h*2 相同）
    // 例 g=8、blockIdx.x=5 -> pairs=4, kv_h=1, pair=1, q0=10, q1=11（KV 头 1 的第二对 Q）
    const int g = num_q_heads / num_kv_heads;
    const int pairs = g / 2;
    const int kv_h = static_cast<int>(blockIdx.x) / pairs;
    const int pair = static_cast<int>(blockIdx.x) % pairs;
    const int q0 = kv_h * g + pair * 2;
    const int q1 = q0 + 1;

    const int rows_two_heads = num_q_tokens * 2;
    const int loop_kv = (num_kv_tokens + kKvTokenTile - 1) / kKvTokenTile;

    const int block_threads = blockDim.x * blockDim.y;
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int warp_id = tid / 32;
    const int num_warps = block_threads / 32;

    __shared__ alignas(16) half q_shared[kWmmaRows][C::kQStride];
    __shared__ alignas(16) half k_double_buf[2][kKvTokenTile][C::kVStride];
    __shared__ alignas(16) half v_double_buf[2][kKvTokenTile][C::kVStride];
    __shared__ alignas(16) half s_scores[kWmmaRows][C::kKvStride];
    __shared__ alignas(16) float dst_acc[kWmmaRows][C::kAccStride];
    __shared__ float stream_num_scale[kWmmaRows];
    __shared__ float m[kWmmaRows];
    __shared__ float l[kWmmaRows];
    __shared__ alignas(16) float pv_acc[kWmmaRows][C::kAccStride];

    const half *q0_base = Q + static_cast<size_t>(q0) * num_q_tokens * HEAD_DIM;
    const half *q1_base = Q + static_cast<size_t>(q1) * num_q_tokens * HEAD_DIM;
    const size_t kv_plane_elems = static_cast<size_t>(num_kv_tokens) * HEAD_DIM;

    const int q_half2_one = num_q_tokens * (HEAD_DIM / 2);

    for (int i = tid; i < q_half2_one; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        const int j2 = i % (HEAD_DIM / 2);
        if (q0 < num_q_heads) {
            const half2 *src = reinterpret_cast<const half2 *>(q0_base + row * HEAD_DIM);
            reinterpret_cast<half2 *>(&q_shared[row][0])[j2] = src[j2];
        }
    }
    for (int i = tid; i < q_half2_one; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        const int j2 = i % (HEAD_DIM / 2);
        const int dr = row + num_q_tokens;
        if (q1 < num_q_heads) {
            const half2 *src = reinterpret_cast<const half2 *>(q1_base + row * HEAD_DIM);
            reinterpret_cast<half2 *>(&q_shared[dr][0])[j2] = src[j2];
        }
    }
    for (int t = tid; t < (kWmmaRows - rows_two_heads) * HEAD_DIM; t += block_threads) {
        const int r = rows_two_heads + t / HEAD_DIM;
        const int c = t % HEAD_DIM;
        q_shared[r][c] = half(0);
    }
    __syncthreads();

    for (int i = tid; i < rows_two_heads; i += block_threads) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
    __syncthreads();

    for (int t = tid; t < rows_two_heads * HEAD_DIM; t += block_threads) {
        const int r = t / HEAD_DIM;
        const int c = t % HEAD_DIM;
        dst_acc[r][c] = 0.0f;
    }
    __syncthreads();

    const int rows_per_warp = (rows_two_heads + num_warps - 1) / num_warps;
    const half *k_head = K + static_cast<size_t>(kv_h) * kv_plane_elems;
    const half *v_head = V + static_cast<size_t>(kv_h) * kv_plane_elems;

    const int valid0 = db_kv_tile_valid_rows(0, num_kv_tokens);
    db_prologue_load_k_tile<HEAD_DIM>(k_double_buf, k_head, valid0, tid, block_threads);
    __syncthreads();

    for (int tile_id = 0; tile_id < loop_kv; ++tile_id) {
        const int cb = tile_id & 1;
        half(*k_active)[C::kVStride] = k_double_buf[cb];
        const int valid_cur = db_kv_tile_valid_rows(tile_id, num_kv_tokens);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if (tile_id > 0) {
            db_cp_async_wait<0>();
            db_zero_kv_tile_pad<HEAD_DIM>(k_active, valid_cur, tid, block_threads);
        }
#endif
        __syncthreads();

        const half *v_tile_base = v_head + static_cast<size_t>(tile_id * kKvTokenTile) * HEAD_DIM;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        db_begin_copy_kv_tile<HEAD_DIM>(v_double_buf[cb], v_tile_base, valid_cur, tid,
                                        block_threads);
#else
        db_sync_copy_kv_tile<HEAD_DIM>(v_double_buf[cb], v_tile_base, valid_cur, tid,
                                       block_threads);
#endif

        if (warp_id < 4) {
            const int warp_m = warp_id / 2;
            const int warp_n = warp_id % 2;
            const int row0 = warp_m * kWmmaM;
            const int col0 = warp_n * kWmmaN;

            fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, half> c_frag;
            fill_fragment(c_frag, half(0));
            for (int k_step = 0; k_step < HEAD_DIM / kWmmaK; ++k_step) {
                fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, half, row_major> a_frag;
                fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, half, col_major> b_frag;
                load_matrix_sync(a_frag, &q_shared[row0][k_step * kWmmaK], C::kQStride);
                load_matrix_sync(b_frag, &k_active[col0][k_step * kWmmaK], C::kVStride);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            store_matrix_sync(&s_scores[row0][col0], c_frag, C::kKvStride, mem_row_major);
        }
        __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if (tile_id + 1 < loop_kv) {
            const int nb = cb ^ 1;
            const int valid_next = db_kv_tile_valid_rows(tile_id + 1, num_kv_tokens);
            const half *k_next =
                k_head + static_cast<size_t>((tile_id + 1) * kKvTokenTile) * HEAD_DIM;
            db_begin_copy_kv_tile<HEAD_DIM>(k_double_buf[nb], k_next, valid_next, tid,
                                            block_threads);
        }
#endif

        const int w_soft = threadIdx.y;
        const int l_id = threadIdx.x;
        const int row_start_w = w_soft * rows_per_warp;
        const int row_end_w = min(row_start_w + rows_per_warp, rows_two_heads);
        const int global_kv_col = tile_id * kKvTokenTile + l_id;

        for (int r = row_start_w; r < row_end_w; ++r) {
            const int q_row = (r < num_q_tokens) ? r : (r - num_q_tokens);
            const int q_abs = params.q_pos_offset + q_row;
            // pad：列已经出了本头的 token 数。causal：看见未来 token 就丢掉。
            // 例：q_pos_offset=0、q_row=0、num_kv_tokens=4 -> 只留 kv_col=0
            int keep = (global_kv_col < num_kv_tokens) ? 1 : 0;
            if (params.causal != 0 && global_kv_col > q_abs) {
                keep = 0;
            }
            float s_val = keep ? __half2float(s_scores[r][l_id]) * params.scale : -INFINITY;
            const float row_max = warp_reduce_xor_max(s_val);
            const float exp_val = expf(s_val - row_max);
            const float row_sum = warp_reduce_xor_sum(exp_val);
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
        db_cp_async_wait<0>();
        db_zero_kv_tile_pad<HEAD_DIM>(v_double_buf[cb], valid_cur, tid, block_threads);
#endif
        __syncthreads();

        half(*v_active)[C::kVStride] = v_double_buf[cb];

        for (int n_tile = warp_id; n_tile < C::kNTiles; n_tile += num_warps) {
            for (int m_tile = 0; m_tile < kWmmaRows / kWmmaM; ++m_tile) {
                const int row0 = m_tile * kWmmaM;
                fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc;
                fill_fragment(acc, 0.0f);
                for (int k_step = 0; k_step < kKvTokenTile / kWmmaK; ++k_step) {
                    fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, half, row_major> a_frag;
                    fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, half, row_major> b_frag;
                    load_matrix_sync(a_frag, &s_scores[row0][k_step * kWmmaK], C::kKvStride);
                    load_matrix_sync(b_frag, &v_active[k_step * kWmmaK][n_tile * kWmmaN],
                                     C::kVStride);
                    mma_sync(acc, a_frag, b_frag, acc);
                }
                store_matrix_sync(&pv_acc[row0][n_tile * kWmmaN], acc, C::kAccStride,
                                  mem_row_major);
            }
        }
        __syncthreads();

        for (int t = tid; t < rows_two_heads * HEAD_DIM; t += block_threads) {
            const int r = t / HEAD_DIM;
            const int c = t % HEAD_DIM;
            const float l_new_r = l[r];
            dst_acc[r][c] = (stream_num_scale[r] * dst_acc[r][c] + pv_acc[r][c]) / l_new_r;
        }
        __syncthreads();
    }

    for (int t = tid; t < num_q_tokens * HEAD_DIM; t += block_threads) {
        const int r = t / HEAD_DIM;
        const int c = t % HEAD_DIM;
        if (q0 < num_q_heads) {
            const int dst_id = c + HEAD_DIM * r + HEAD_DIM * num_q_tokens * q0;
            dst[dst_id] = dst_acc[r][c];
        }
    }
    for (int t = tid; t < num_q_tokens * HEAD_DIM; t += block_threads) {
        const int r = t / HEAD_DIM;
        const int c = t % HEAD_DIM;
        const int dr = r + num_q_tokens;
        if (q1 < num_q_heads) {
            const int dst_id = c + HEAD_DIM * r + HEAD_DIM * num_q_tokens * q1;
            dst[dst_id] = dst_acc[dr][c];
        }
    }
    __syncthreads();
}

template <int HEAD_DIM>
static void fa_double_buffer_launch_templated(cudaStream_t stream, const half *d_q, const half *d_k,
                                              const half *d_v, float *d_dst,
                                              const FaDoubleBufferKernelParams &kparams) {
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void *)fa_kernel_double_buffer<HEAD_DIM>));
    const size_t static_shmem = static_cast<size_t>(attr.sharedSizeBytes);
    constexpr int kDefaultShmemPerBlock = 48 * 1024;

    int max_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    if (static_shmem > static_cast<size_t>(kDefaultShmemPerBlock)) {
        const int excess =
            static_cast<int>(static_shmem - static_cast<size_t>(kDefaultShmemPerBlock));
        if (excess > 0) {
            if (max_optin < static_cast<int>(static_shmem)) {
                std::fprintf(
                    stderr, "[fa_double_buffer] head_dim=%d static shared %zu B exceeds optin=%d\n",
                    HEAD_DIM, static_shmem, max_optin);
            }
            CUDA_CHECK(cudaFuncSetAttribute((void *)fa_kernel_double_buffer<HEAD_DIM>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, excess));
        }
    }

    dim3 threads(32, 8, 1);
    // 每 block 2 个 Q；g=2 时 n_blocks==num_kv_heads，与旧 grid 相同。
    const int n_blocks = kparams.num_q_heads / 2;
    dim3 blocks(n_blocks, 1, 1);
    fa_kernel_double_buffer<HEAD_DIM>
        <<<blocks, threads, 0, stream>>>(d_q, d_k, d_v, d_dst, kparams);
    LAUNCH_CHECK();
}

} // namespace fa_db
