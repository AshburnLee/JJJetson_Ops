#include <cuda_runtime.h>
#include <cstdio>
#include "cuda_fp16.h"
#include "cuda_utils.cuh"

// 单遍 streaming：每 block 2 个 Q head，对应 1 个 KV head（GQA 16Q/8KV）。
__global__ void
fa_kernel_one_pass(const half *__restrict__ Q, const half *__restrict__ K,
                   const half *__restrict__ V, float *__restrict__ dst, const float scale
#if defined(MY_OPS_DEBUG)
                   ,
                   float *__restrict__ m_out, float *__restrict__ l_out, float *__restrict__ s_out,
                   float *__restrict__ row_sum_out, float *__restrict__ scale_old_out,
                   float *__restrict__ scale_new_out, float *__restrict__ exp_val_out
#endif
) {
    constexpr int HEAD_DIM = 128;
    constexpr int TOKENS_PER_Q = 13;
    constexpr int QHEAD_PER_BLOCK = 2;
    constexpr int ROWS = TOKENS_PER_Q * QHEAD_PER_BLOCK;
    constexpr int KV_TOKEN_TILE = 32;
    constexpr int LOOP_KV = 256 / KV_TOKEN_TILE;

    __shared__ half q_shared[ROWS][HEAD_DIM];
    int tid_blockwise = threadIdx.x + blockDim.x * threadIdx.y;
    int warp_id = tid_blockwise / 32;
    int lane_id = tid_blockwise % 32;

    const int qhead_block_offset = blockIdx.x * QHEAD_PER_BLOCK;
    const int qheadid_in_block = warp_id / 2;
    int half_head = warp_id % 2;
    int row_start = half_head * 64;
    int s_col0_id = lane_id * 2 + row_start;

    const int qhead_id_global = qhead_block_offset + qheadid_in_block;
    const half *qhead_elem_start = Q + qhead_id_global * TOKENS_PER_Q * HEAD_DIM;

#pragma unroll
    for (int token_id = 0; token_id < TOKENS_PER_Q; token_id++) {
        const half2 *row_h2 =
            reinterpret_cast<const half2 *>(qhead_elem_start + token_id * HEAD_DIM);
        half2 data = row_h2[s_col0_id / 2];
        const int s_row_id = qheadid_in_block * TOKENS_PER_Q + token_id;
        *reinterpret_cast<half2 *>(&q_shared[s_row_id][s_col0_id]) = data;
    }
    __syncthreads();

    __shared__ half k_shared[KV_TOKEN_TILE][HEAD_DIM + 2];
    __shared__ half v_shared[KV_TOKEN_TILE][HEAD_DIM + 2];
    __shared__ float s_shared[ROWS][KV_TOKEN_TILE];
    __shared__ float dst_shared[ROWS][HEAD_DIM];
    __shared__ float stream_num_scale[ROWS];
    __shared__ float stream_l_new[ROWS];

    const int kv_head_block_offset = blockIdx.x;
    const int warp_row_id = qheadid_in_block;

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
            const half *row_base = K + col_warp_id * HEAD_DIM + tile_id * KV_TOKEN_TILE * HEAD_DIM +
                                   kv_head_block_offset * 256 * HEAD_DIM;
            const half2 *row_h2 = reinterpret_cast<const half2 *>(row_base);
            half2 data = row_h2[s_col0_id / 2];
            *reinterpret_cast<half2 *>(&k_shared[col_warp_id][s_col0_id]) = data;
        }
        __syncthreads();

        const int tid2 = threadIdx.y * blockDim.x + threadIdx.x;
        for (int idx = tid2; idx < ROWS * KV_TOKEN_TILE; idx += 128) {
            int i = idx / KV_TOKEN_TILE;
            int j = idx % KV_TOKEN_TILE;
            float sum = 0.0f;
#pragma unroll
            for (int hd = 0; hd < 128; hd += 2) {
                const half2 *q = reinterpret_cast<const half2 *>(&q_shared[i][hd]);
                const half2 *k = reinterpret_cast<const half2 *>(&k_shared[j][hd]);
                sum += __half2float(q->x) * __half2float(k->x) +
                       __half2float(q->y) * __half2float(k->y);
            }
            s_shared[i][j] = sum;
#if defined(MY_OPS_DEBUG)
            if (s_out != nullptr) {
                int idx4 = (((blockIdx.x * LOOP_KV) + tile_id) * ROWS + i) * KV_TOKEN_TILE + j;
                s_out[idx4] = sum;
            }
#endif
        }
        __syncthreads();

        int w_id = threadIdx.y;
        int l_id = threadIdx.x;
        int row_start_w = w_id * ROWS_PER_WARP;
        int row_end_w = min(row_start_w + ROWS_PER_WARP, ROWS);

        for (int r = row_start_w; r < row_end_w; ++r) {
            float s = s_shared[r][l_id];
            float row_max = warp_reduce_xor_max(s);
            float exp_val = expf(s - row_max);
            float row_sum = warp_reduce_xor_sum(exp_val);
#if defined(MY_OPS_DEBUG)
            if (exp_val_out != nullptr) {
                int idx4 = (((blockIdx.x * LOOP_KV) + tile_id) * ROWS + r) * 32 + l_id;
                exp_val_out[idx4] = exp_val;
            }
#endif
            float scale_new = 0.0f;
            if (l_id == 0) {
                const float l_old = l[r];
                const float m_new = fmaxf(m[r], row_max);
                const float scale_old = expf(m[r] - m_new);
                scale_new = expf(row_max - m_new);
                const float l_new = l_old * scale_old + row_sum * scale_new;
                stream_num_scale[r] = l_old * scale_old;
                stream_l_new[r] = l_new;
                m[r] = m_new;
                l[r] = l_new;
#if defined(MY_OPS_DEBUG)
                if (row_sum_out != nullptr && scale_old_out != nullptr &&
                    scale_new_out != nullptr) {
                    int idx3 = ((blockIdx.x * LOOP_KV) + tile_id) * ROWS + r;
                    row_sum_out[idx3] = row_sum;
                    scale_old_out[idx3] = scale_old;
                    scale_new_out[idx3] = scale_new;
                }
#endif
            }
            scale_new = __shfl_sync(0xFFFFFFFFu, scale_new, 0);
            s_shared[r][l_id] = exp_val * scale_new;
        }
        __syncthreads();

        for (int token_id = 0; token_id < KV_TOKEN_TILE; token_id += 2) {
            int col_warp_id = warp_row_id + token_id;
            const half *row_base = V + col_warp_id * HEAD_DIM + tile_id * KV_TOKEN_TILE * HEAD_DIM +
                                   kv_head_block_offset * 256 * HEAD_DIM;
            const half2 *row_h2 = reinterpret_cast<const half2 *>(row_base);
            half2 data = row_h2[s_col0_id / 2];
            *reinterpret_cast<half2 *>(&v_shared[col_warp_id][s_col0_id]) = data;
        }
        __syncthreads();

        // warp stall
        for (int dst_row = 0; dst_row < ROWS; ++dst_row) {
            int col = block_tid;
            float sum_pv = 0.0f;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                float v = __half2float(v_shared[k][col]);
                sum_pv += s_shared[dst_row][k] * v;
            }
            const float l_new_r = stream_l_new[dst_row];
            dst_shared[dst_row][col] =
                (stream_num_scale[dst_row] * dst_shared[dst_row][col] + sum_pv) / l_new_r;
        }
        __syncthreads();
    }

#if defined(MY_OPS_DEBUG)
    if (m_out != nullptr && l_out != nullptr) {
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < ROWS;
             i += blockDim.x * blockDim.y) {
            m_out[blockIdx.x * ROWS + i] = m[i];
            l_out[blockIdx.x * ROWS + i] = l[i];
        }
        __syncthreads();
    }
#endif

    if (block_tid < HEAD_DIM) {
        for (int r = 0; r < ROWS; ++r) {
            const int qhead_local = r / TOKENS_PER_Q;
            const int token_id = r % TOKENS_PER_Q;
            const int qhead_global = blockIdx.x * QHEAD_PER_BLOCK + qhead_local;
            if (qhead_global < 16) {
                const int hd = block_tid;
                const int dst_id =
                    hd + HEAD_DIM * token_id + HEAD_DIM * TOKENS_PER_Q * qhead_global;
                dst[dst_id] = dst_shared[r][hd];
            }
        }
    }
    __syncthreads();
    (void)scale;
}

extern "C" void fa_one_pass(const uint16_t *q_host, const uint16_t *k_host, const uint16_t *v_host,
                            float *dst_host, float scale) {
    // q: (head_dim, n_token, n_qhead)    = (128,13,16)
    // kv: (head_dim, n_token, n_kv_head) = (128,256,8)

    using half_t = half;
    constexpr int HEAD_DIM = 128;
    constexpr int TOKENS_PER_Q = 13;
    constexpr int Q_HEADS = 16;
    constexpr int KV_TOKENS = 256;
    constexpr int KV_HEADS = 8;

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

    dim3 threads(32, 4, 1);
    dim3 blocks(8, 1, 1);

#if defined(MY_OPS_DEBUG)
    fa_kernel_one_pass<<<blocks, threads, 0, stream>>>(
        d_q, d_k, d_v, d_dst, scale, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
#else
    fa_kernel_one_pass<<<blocks, threads, 0, stream>>>(d_q, d_k, d_v, d_dst, scale);
#endif
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

#if defined(MY_OPS_DEBUG)
extern "C" void fa_debug(const uint16_t *q_host, const uint16_t *k_host, const uint16_t *v_host,
                         float *dst_host, float scale, float *m_host, float *l_host, float *s_host,
                         float *row_sum_host, float *scale_old_host, float *scale_new_host,
                         float *exp_val_host) {

    using half_t = half;
    constexpr int HEAD_DIM = 128;
    constexpr int TOKENS_PER_Q = 13;
    constexpr int Q_HEADS = 16;
    constexpr int KV_TOKENS = 256;
    constexpr int KV_HEADS = 8;
    constexpr int LOOP_KV = 8;
    constexpr int ROWS = 26;

    const size_t q_elems = static_cast<size_t>(HEAD_DIM) * TOKENS_PER_Q * Q_HEADS;
    const size_t kv_elems = static_cast<size_t>(HEAD_DIM) * KV_TOKENS * KV_HEADS;
    const size_t dst_elems = q_elems;
    const size_t ml_elems = static_cast<size_t>(KV_HEADS) * ROWS;
    const size_t s_elems = static_cast<size_t>(KV_HEADS) * LOOP_KV * ROWS * 32;
    const size_t rsn_elems = static_cast<size_t>(KV_HEADS) * LOOP_KV * ROWS;
    const size_t exp_elems = static_cast<size_t>(KV_HEADS) * LOOP_KV * ROWS * 32;

    half_t *d_q = nullptr;
    half_t *d_k = nullptr;
    half_t *d_v = nullptr;
    float *d_dst = nullptr;
    float *d_m = nullptr;
    float *d_l = nullptr;
    float *d_s = nullptr;
    float *d_row_sum = nullptr;
    float *d_scale_old = nullptr;
    float *d_scale_new = nullptr;
    float *d_exp_val = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_q, q_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_k, kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_v, kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, dst_elems * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_m, ml_elems * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_l, ml_elems * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_s, s_elems * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_row_sum, rsn_elems * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scale_old, rsn_elems * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scale_new, rsn_elems * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_exp_val, exp_elems * sizeof(float), stream));

    CUDA_CHECK(
        cudaMemcpyAsync(d_q, q_host, q_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_k, k_host, kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_v, v_host, kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));

    dim3 threads(32, 4, 1);
    dim3 blocks(8, 1, 1);

    fa_kernel_one_pass<<<blocks, threads, 0, stream>>>(
        d_q, d_k, d_v, d_dst, scale, d_m, d_l, d_s, d_row_sum, d_scale_old, d_scale_new, d_exp_val);
    LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, dst_elems * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(
        cudaMemcpyAsync(m_host, d_m, ml_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(l_host, d_l, ml_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(s_host, d_s, s_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(row_sum_host, d_row_sum, rsn_elems * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(scale_old_host, d_scale_old, rsn_elems * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(scale_new_host, d_scale_new, rsn_elems * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(exp_val_host, d_exp_val, exp_elems * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeAsync(d_q, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_k, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_v, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_dst, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_m, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_l, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_s, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_row_sum, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_scale_old, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_scale_new, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_exp_val, nullptr));
}
#endif
