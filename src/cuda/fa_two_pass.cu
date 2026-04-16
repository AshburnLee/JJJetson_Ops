#include <cuda_runtime.h>
#include <cstdio>
#include "cuda_fp16.h"
#include "cuda_utils.cuh"

// Tensor 列主序。两遍 KV：先扫 K 更新全局 m/l，再扫 K+V 写输出。
__global__ void fa_kernel_two_pass(
                const half* __restrict__ Q,
                const half* __restrict__ K,
                const half* __restrict__ V,
                float* __restrict__ dst,
                const float scale
#if defined(MY_OPS_DEBUG)
                , float* __restrict__ m_out,
                float* __restrict__ l_out,
                float* __restrict__ s_out,
                float* __restrict__ row_sum_out,
                float* __restrict__ scale_old_out,
                float* __restrict__ scale_new_out,
                float* __restrict__ exp_val_out
#endif
                ) {
    constexpr int HEAD_DIM = 128;
    constexpr int TOKENS_PER_Q = 13;
    constexpr int QHEAD_PER_BLOCK = 2;
    constexpr int KV_TOKEN_TILE = 32;
    constexpr int LOOP_KV = 256 / KV_TOKEN_TILE;

    __shared__ half q_shared[TOKENS_PER_Q * QHEAD_PER_BLOCK][HEAD_DIM];
    int tid_blockwise = threadIdx.x + blockDim.x * threadIdx.y;
    int warp_id = tid_blockwise / 32;
    int lane_id = tid_blockwise % 32;
    int qhead_block_offset = blockIdx.x * QHEAD_PER_BLOCK;
    int qheadid_in_block = warp_id / 2;
    int half_head        = warp_id % 2;
    int row_start        = half_head * 64;
    int qhead_id_global = qhead_block_offset + qheadid_in_block;
    const half* qhead_elem_start = Q + qhead_id_global * 13 * 128;
    int s_col0_id = lane_id * 2 + row_start;

#pragma unroll
    for (int token_id = 0; token_id < TOKENS_PER_Q; token_id++) {
        const half2* row_h2 =
            reinterpret_cast<const half2*>(qhead_elem_start + token_id * HEAD_DIM);
        half2 data = row_h2[s_col0_id / 2];
        int s_row_id = qheadid_in_block * TOKENS_PER_Q + token_id;
        *reinterpret_cast<half2*>(&q_shared[s_row_id][s_col0_id]) = data;
    }
    __syncthreads();

    __shared__ half k_shared[KV_TOKEN_TILE][HEAD_DIM+2];
    __shared__ half v_shared[KV_TOKEN_TILE][HEAD_DIM+2];
    __shared__ float s_shared[TOKENS_PER_Q * QHEAD_PER_BLOCK][KV_TOKEN_TILE];
    __shared__ float dst_shared[TOKENS_PER_Q * QHEAD_PER_BLOCK][HEAD_DIM];
    int kv_head_block_offset = blockIdx.x;
    int warp_row_id = qheadid_in_block;

    __shared__ float m[26];
    __shared__ float l[26];

    if (threadIdx.x < 26) {
        m[threadIdx.x] = -INFINITY;
        l[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    int ROWS = 26;
    int WARPS = blockDim.y;
    int ROWS_PER_WARP = (ROWS + WARPS - 1) / WARPS;

    int block_tid = threadIdx.x + blockDim.x * threadIdx.y;
    for (int i = block_tid; i < TOKENS_PER_Q * QHEAD_PER_BLOCK * HEAD_DIM; i += HEAD_DIM) {
        int row = i / 128;
        int col = i % 128;
        dst_shared[row][col] = 0.0f;
    }
    __syncthreads();

    for(int tile_id = 0; tile_id < LOOP_KV; tile_id++) {
        for(int token_id = 0; token_id < KV_TOKEN_TILE; token_id += 2) {
            int col_warp_id = warp_row_id + token_id;
            const half* row_base = K + col_warp_id * HEAD_DIM + tile_id * KV_TOKEN_TILE * HEAD_DIM +
                                   kv_head_block_offset * 256 * HEAD_DIM;
            const half2* row_h2 = reinterpret_cast<const half2*>(row_base);
            half2 data = row_h2[s_col0_id / 2];
            *reinterpret_cast<half2*>(&k_shared[col_warp_id][s_col0_id]) = data;
        }
        __syncthreads();

        const int tid2 = threadIdx.y * blockDim.x + threadIdx.x;
        for (int idx = tid2; idx < 13*2*32; idx += 128) {
            int i = idx / 32;
            int j = idx % 32;
            float sum = 0.0f;
#pragma unroll
            for (int hd = 0; hd < 128; hd += 2) {
                const half2* q = reinterpret_cast<const half2*>(&q_shared[i][hd]);
                const half2* k = reinterpret_cast<const half2*>(&k_shared[j][hd]);
                sum += __half2float(q->x) * __half2float(k->x) + __half2float(q->y) * __half2float(k->y);
            }
            s_shared[i][j] = sum;
#if defined(MY_OPS_DEBUG)
            if (s_out != nullptr) {
                int block = blockIdx.x;
                int tile  = tile_id;
                int idx4  = (((block * LOOP_KV) + tile) * (TOKENS_PER_Q * QHEAD_PER_BLOCK) + i) * KV_TOKEN_TILE + j;
                s_out[idx4] = sum;
            }
#endif
        }
        __syncthreads();

        warp_id = threadIdx.y;
        lane_id = threadIdx.x;
        int row_start_r = warp_id * ROWS_PER_WARP;
        int row_end = min(row_start_r + ROWS_PER_WARP, ROWS);

        for (int r = row_start_r; r < row_end; ++r) {
            float s = s_shared[r][lane_id];
            float row_max = warp_reduce_xor_max(s);
            float exp_val = expf(s - row_max);
#if defined(MY_OPS_DEBUG)
            if (exp_val_out != nullptr) {
                int idx4 = (((blockIdx.x * LOOP_KV) + tile_id) * 26 + r) * 32 + lane_id;
                exp_val_out[idx4] = exp_val;
            }
#endif
            float row_sum = warp_reduce_xor_sum(exp_val);

            if (lane_id == 0) {
                float m_new = fmaxf(m[r], row_max);
                float scale_old = expf(m[r] - m_new);
                float scale_new = expf(row_max - m_new);
                l[r] = l[r] * scale_old + row_sum * scale_new;
                m[r] = m_new;

#if defined(MY_OPS_DEBUG)
                if (row_sum_out != nullptr && scale_old_out != nullptr && scale_new_out != nullptr) {
                    int idx3 = ((blockIdx.x * LOOP_KV) + tile_id) * 26 + r;
                    row_sum_out[idx3] = row_sum;
                    scale_old_out[idx3] = scale_old;
                    scale_new_out[idx3] = scale_new;
                }
#endif
            }
        }
        __syncthreads();
    }

#if defined(MY_OPS_DEBUG)
    if (m_out != nullptr && l_out != nullptr) {
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < 26; i += blockDim.x * blockDim.y) {
            m_out[blockIdx.x * 26 + i] = m[i];
            l_out[blockIdx.x * 26 + i] = l[i];
        }
        __syncthreads();
    }
#endif

    for(int tile_id = 0; tile_id < LOOP_KV; tile_id++) {
        for(int token_id = 0; token_id < KV_TOKEN_TILE; token_id += 2) {
            int col_warp_id = warp_row_id + token_id;
            const half* row_base = K + col_warp_id * HEAD_DIM + tile_id * KV_TOKEN_TILE * HEAD_DIM +
                                   kv_head_block_offset * 256 * HEAD_DIM;
            const half2* row_h2 = reinterpret_cast<const half2*>(row_base);
            half2 data = row_h2[s_col0_id / 2];
            *reinterpret_cast<half2*>(&k_shared[col_warp_id][s_col0_id]) = data;
        }
        __syncthreads();

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int idx = tid; idx < 13*2*32; idx += 128) {
            int i = idx / 32;
            int j = idx % 32;
            float sum = 0.0f;
#pragma unroll
            for (int hd = 0; hd < 128; hd += 2) {
                const half2* q = reinterpret_cast<const half2*>(&q_shared[i][hd]);
                const half2* k = reinterpret_cast<const half2*>(&k_shared[j][hd]);
                sum += __half2float(q->x) * __half2float(k->x) + __half2float(q->y) * __half2float(k->y);
            }
            s_shared[i][j] = sum;
        }
        __syncthreads();

        warp_id = threadIdx.y;
        lane_id = threadIdx.x;
        int row_start_r = warp_id * ROWS_PER_WARP;
        int row_end = min(row_start_r + ROWS_PER_WARP, ROWS);
        for (int r = row_start_r; r < row_end; ++r) {
            s_shared[r][lane_id] = expf(s_shared[r][lane_id] - m[r]) / l[r];
        }
        __syncthreads();

        for(int token_id = 0; token_id < KV_TOKEN_TILE; token_id += 2) {
            int col_warp_id = warp_row_id + token_id;
            const half* row_base = V + col_warp_id * HEAD_DIM + tile_id * KV_TOKEN_TILE * HEAD_DIM +
                                   kv_head_block_offset * 256 * HEAD_DIM;
            const half2* row_h2 = reinterpret_cast<const half2*>(row_base);
            half2 data = row_h2[s_col0_id / 2];
            *reinterpret_cast<half2*>(&v_shared[col_warp_id][s_col0_id]) = data;
        }
        __syncthreads();

        // warp stall
        for (int dst_row = 0; dst_row < TOKENS_PER_Q * QHEAD_PER_BLOCK; ++dst_row ) {
            int col = block_tid;
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                float v = __half2float(v_shared[k][col]);
                sum += s_shared[dst_row][k] * v;
            }
            dst_shared[dst_row][col] += sum;
        }
        __syncthreads();
    }

    if (block_tid < HEAD_DIM) {
        for (int r = 0; r < TOKENS_PER_Q * QHEAD_PER_BLOCK; ++r) {
            const int qhead_local  = r / TOKENS_PER_Q;
            const int token_id     = r % TOKENS_PER_Q;
            const int qhead_global = blockIdx.x * QHEAD_PER_BLOCK + qhead_local;
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

extern "C" void fa_two_pass(
                const uint16_t* q_host,
                const uint16_t* k_host,
                const uint16_t* v_host,
                float* dst_host,
                float scale) {

    using half_t = half;
    constexpr int HEAD_DIM       = 128;
    constexpr int TOKENS_PER_Q   = 13;
    constexpr int Q_HEADS        = 16;
    constexpr int KV_TOKENS      = 256;
    constexpr int KV_HEADS       = 8;

    const size_t q_elems   = static_cast<size_t>(HEAD_DIM) * TOKENS_PER_Q * Q_HEADS;
    const size_t kv_elems  = static_cast<size_t>(HEAD_DIM) * KV_TOKENS * KV_HEADS;
    const size_t dst_elems = q_elems;

    half_t* d_q   = nullptr;
    half_t* d_k   = nullptr;
    half_t* d_v   = nullptr;
    float*  d_dst = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_q,   q_elems  * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_k,   kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_v,   kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, dst_elems * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_q,   q_host,  q_elems  * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_k,   k_host,  kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v,   v_host,  kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));

    dim3 threads(32, 4, 1);
    dim3 blocks(8, 1, 1);

#if defined(MY_OPS_DEBUG)
    fa_kernel_two_pass<<<blocks, threads, 0, stream>>>(
        d_q, d_k, d_v, d_dst, scale,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
#else
    fa_kernel_two_pass<<<blocks, threads, 0, stream>>>(d_q, d_k, d_v, d_dst, scale);
#endif
    LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, dst_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeAsync(d_q,   nullptr));
    CUDA_CHECK(cudaFreeAsync(d_k,   nullptr));
    CUDA_CHECK(cudaFreeAsync(d_v,   nullptr));
    CUDA_CHECK(cudaFreeAsync(d_dst, nullptr));
}
