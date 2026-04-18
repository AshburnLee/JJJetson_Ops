#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include "cuda_fp16.h"
#include "cuda_utils.cuh"

// fa_one_pass_parallel_tc_true — 双 Q-head 合并 + QK half 累加器 + half S（streaming softmax 后复用为 P）
//
// - gridDim.x = 8：每 KV head 一个 CTA，一次处理 GQA 一对 Q head（26 行 Q pad→32），QK 用 4 warp 铺满 32×32 WMMA。
// - QK：WMMA 累加器为 half；S 存 half，softmax 在 FP32 上算后再写回 half。
// - P·V：WMMA（P 为 half，V 为 half，累加 FP32）；与 fa_ref（全 FP32 路径）相比可出现约 0.06 量级 max diff。
// - V_LD = HEAD_DIM，保证 WMMA B 加载 16B 对齐。
//
// 若需与参考实现 strict 对齐，请改用 fa_one_pass_parallel_tc.cu 的 FP32 QK + CUDA Core PV 路径。

namespace {

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
constexpr int V_LD = HEAD_DIM;
constexpr int N_TILES = HEAD_DIM / WMMA_N;

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
    const int num_warps = block_threads / 32;

    __shared__ alignas(16) half Qsh[WMMA_ROWS][HEAD_DIM];
    __shared__ half kv_shared[KV_TOKEN_TILE][V_LD];
    __shared__ alignas(16) half s_scores[WMMA_ROWS][KV_TOKEN_TILE];
    __shared__ alignas(16) float dst_acc[ROWS_TWO_HEADS][HEAD_DIM];
    __shared__ float stream_num_scale[ROWS_TWO_HEADS];
    __shared__ float m[ROWS_TWO_HEADS];
    __shared__ float l[ROWS_TWO_HEADS];

    __shared__ union {
        alignas(16) half kt[HEAD_DIM][KV_TOKEN_TILE];
        alignas(16) float pv_acc[WMMA_ROWS][HEAD_DIM];
    } u_kt_pv;

    const int kv_h = blockIdx.x;
    const int q0 = kv_h * 2;
    const int q1 = q0 + 1;

    const half* q0_base = Q + static_cast<size_t>(q0) * TOKENS_PER_Q * HEAD_DIM;
    const half* q1_base = Q + static_cast<size_t>(q1) * TOKENS_PER_Q * HEAD_DIM;
    const size_t kv_plane_elems = static_cast<size_t>(KV_TOKENS) * HEAD_DIM;

    constexpr int Q_HALF2_ONE = TOKENS_PER_Q * (HEAD_DIM / 2);

    for (int i = tid; i < Q_HALF2_ONE; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        const int j2 = i % (HEAD_DIM / 2);
        const half2* src = reinterpret_cast<const half2*>(q0_base + row * HEAD_DIM);
        reinterpret_cast<half2*>(&Qsh[row][0])[j2] = src[j2];
    }
    for (int i = tid; i < Q_HALF2_ONE; i += block_threads) {
        const int row = i / (HEAD_DIM / 2);
        const int j2 = i % (HEAD_DIM / 2);
        const int dr = row + TOKENS_PER_Q;
        const half2* src = reinterpret_cast<const half2*>(q1_base + row * HEAD_DIM);
        reinterpret_cast<half2*>(&Qsh[dr][0])[j2] = src[j2];
    }
    for (int t = tid; t < (WMMA_ROWS - ROWS_TWO_HEADS) * HEAD_DIM; t += block_threads) {
        const int r = ROWS_TWO_HEADS + t / HEAD_DIM;
        const int c = t % HEAD_DIM;
        Qsh[r][c] = half(0);
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

        for (int i = tid; i < K_VEC; i += block_threads) {
            const int row = i / (HEAD_DIM / 2);
            const int j2 = i % (HEAD_DIM / 2);
            const half2* src_h2 =
                reinterpret_cast<const half2*>(k_tile_base + row * HEAD_DIM);
            reinterpret_cast<half2*>(&kv_shared[row][0])[j2] = src_h2[j2];
        }
        __syncthreads();

        for (int t = tid; t < HEAD_DIM * KV_TOKEN_TILE; t += block_threads) {
            const int h = t / KV_TOKEN_TILE;
            const int j = t % KV_TOKEN_TILE;
            u_kt_pv.kt[h][j] = kv_shared[j][h];
        }
        __syncthreads();

        if (warp_id < 4) {
            const int warp_m = warp_id / 2;
            const int warp_n = warp_id % 2;
            const int row0 = warp_m * WMMA_M;
            const int col0 = warp_n * WMMA_N;

            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
            fill_fragment(c_frag, half(0));
            for (int k_step = 0; k_step < HEAD_DIM / WMMA_K; ++k_step) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
                load_matrix_sync(a_frag, &Qsh[row0][k_step * WMMA_K], HEAD_DIM);
                load_matrix_sync(b_frag, &u_kt_pv.kt[k_step * WMMA_K][col0], KV_TOKEN_TILE);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            store_matrix_sync(&s_scores[row0][col0], c_frag, KV_TOKEN_TILE, mem_row_major);
        }
        __syncthreads();

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
        __syncthreads();

        const half* v_tile_base = v_head + static_cast<size_t>(tile_id * KV_TOKEN_TILE) * HEAD_DIM;
        for (int i = tid; i < K_VEC; i += block_threads) {
            const int row = i / (HEAD_DIM / 2);
            const int j2 = i % (HEAD_DIM / 2);
            const half2* src_h2 =
                reinterpret_cast<const half2*>(v_tile_base + row * HEAD_DIM);
            reinterpret_cast<half2*>(&kv_shared[row][0])[j2] = src_h2[j2];
        }
        __syncthreads();

        for (int n_tile = warp_id; n_tile < N_TILES; n_tile += num_warps) {
            for (int m_tile = 0; m_tile < WMMA_ROWS / WMMA_M; ++m_tile) {
                const int row0 = m_tile * WMMA_M;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
                fill_fragment(acc, 0.0f);
                for (int k_step = 0; k_step < KV_TOKEN_TILE / WMMA_K; ++k_step) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
                    load_matrix_sync(a_frag, &s_scores[row0][k_step * WMMA_K], KV_TOKEN_TILE);
                    load_matrix_sync(
                        b_frag, &kv_shared[k_step * WMMA_K][n_tile * WMMA_N], V_LD);
                    mma_sync(acc, a_frag, b_frag, acc);
                }
                store_matrix_sync(
                    &u_kt_pv.pv_acc[row0][n_tile * WMMA_N], acc, HEAD_DIM, mem_row_major);
            }
        }
        __syncthreads();

        constexpr int OUT_VEC = ROWS_TWO_HEADS * HEAD_DIM;
        for (int t = tid; t < OUT_VEC; t += block_threads) {
            const int r = t / HEAD_DIM;
            const int c = t % HEAD_DIM;
            const float l_new_r = l[r];
            dst_acc[r][c] =
                (stream_num_scale[r] * dst_acc[r][c] + u_kt_pv.pv_acc[r][c]) / l_new_r;
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
