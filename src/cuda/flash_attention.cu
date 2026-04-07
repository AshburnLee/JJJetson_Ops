#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "cuda_fp16.h"   // __device__ __host__ __half2 make_half2(__half x, __half y);
#include "cuda_utils.cuh"

// Tensor 坐标均按 **列主序 (column-major)** 解释，
// 即第一个维度元素变化最快。

__global__ void flash_attn_tile_kernel(
                const half* __restrict__ Q,  // [128, 13, 16, 1]
                const half* __restrict__ K,  // [128, 256, 8, 1] 
                const half* __restrict__ V,  // [128, 256, 8, 1]
                float* __restrict__ dst,      // 输出 [128, 13, 16, 1]
                const float scale             // 缩放因子
#if defined(MY_OPS_DEBUG)
                , float* __restrict__ m_out,       // [8,26] optional (可 nullptr)
                float* __restrict__ l_out,       // [8,26] optional (可 nullptr)
                float* __restrict__ s_out,       // [8,8,26,32] optional (可 nullptr)
                float* __restrict__ row_sum_out,    // [8,8,26] optional (可 nullptr)
                float* __restrict__ scale_old_out,  // [8,8,26] optional (可 nullptr)
                float* __restrict__ scale_new_out,  // [8,8,26] optional (可 nullptr)
                float* __restrict__ exp_val_out     // [8,8,26,32] optional (可 nullptr)
#endif
                ){
    constexpr int HEAD_DIM = 128;
    constexpr int TOKENS_PER_Q = 13;
    constexpr int QHEAD_PER_BLOCK = 2; // 16/8 =2 每个block处理连续2个Qhead
    constexpr int EITEM_PER_THREAD = 2; // 每个thread处理两个数值 
    constexpr int KV_TOKEN_TILE = 32;
    constexpr int LOOP_KV = 256 / KV_TOKEN_TILE; // 256/32 =8 即总共需要 8 次循环来加载整个 K 和 V

    __shared__ half q_shared[TOKENS_PER_Q * QHEAD_PER_BLOCK][HEAD_DIM]; // [13*2][128]
    // kernel 中所有的 threadIdx 相关的变量都是 block-wise 的变量
    // insight：warp id 根本上是这个warp中32个**thread所在warp的id**，故这个id是32个thread 的！
    // insight: lane_id 根本上是这个warp中32个**thread所在的lane的id**，故这个id是32个thread 的！
    int tid_blockwise = threadIdx.x + blockDim.x * threadIdx.y;   // 0~127  [0,1,2,3,...,125,126,127]
    int warp_id = tid_blockwise / 32; 
    int lane_id = tid_blockwise % 32; 
    // 这里出现 blockIdx，故这个变量就是 grid wise 的变量了
    // blockIdx 与 threadIdx 完全无关
    int qhead_block_offset = blockIdx.x * QHEAD_PER_BLOCK; // [0,2,4,6,8,10,12,14]

    // 此3个都是一个block 中 block-wise 的变量，
    // 故 qheadid_in_block 有4个值表示4个warp负责的 Q head id
    int qheadid_in_block = warp_id / 2;     // block-wise
    int half_head        = warp_id % 2;     // block-wise
    int row_start        = half_head * 64;  // block-wise

    // grid wise 变量，表示当前 block 负责的 Q head id:
    int qhead_id_global = qhead_block_offset + qheadid_in_block;
    // 跨过一个 qhead 所有元素后的 id，即Global的 Q head 起始位置
    const half* qhead_elem_start = Q + qhead_id_global * 13 * 128;

    int s_col0_id = lane_id * 2 + row_start;
    int s_col1_id = s_col0_id + 1;

    /// 1. Load Q tile（每一个thread 读连续两个值）
#pragma unroll
    for (int token_id = 0; token_id < TOKENS_PER_Q/*=13*/; token_id++) {
        if (s_col1_id < 128) {
            const half* src0 = qhead_elem_start + s_col0_id * 1 + token_id * 128;
            const half* src1 = qhead_elem_start + s_col1_id * 1 + token_id * 128;
            half2 data = make_half2(*src0, *src1);

            // shared memory 是 block-wise 的
            int s_row_id = qheadid_in_block * TOKENS_PER_Q + token_id;
            q_shared[s_row_id][s_col0_id] = data.x;
            q_shared[s_row_id][s_col1_id] = data.y;
        }
    }
    __syncthreads();

    /// loop_8_times_{load K_tile -> compute Q_tile * K_tile -> compute M&L}
    __shared__ half k_shared[KV_TOKEN_TILE][HEAD_DIM+2];                        // [32][128+2]
    __shared__ half v_shared[KV_TOKEN_TILE][HEAD_DIM+2];                        // [32][128+2]
    __shared__ float s_shared[TOKENS_PER_Q * QHEAD_PER_BLOCK][KV_TOKEN_TILE]; // [13*2][32]
    //__shared__ float m_l[TOKENS_PER_Q * QHEAD_PER_BLOCK][2];                  // [13*2][2] 
    __shared__ float dst_shared[TOKENS_PER_Q * QHEAD_PER_BLOCK][HEAD_DIM];    //[13*2][128]
    int kv_head_block_offset = blockIdx.x; // [0,1,2,3,4,5,6,7]

    // [0,0,0,...,0,0]    [0,0,0,...,0,0]      [1,1,1,...,1,1]    [1,1,1,...,1,1]
    int warp_row_id = qheadid_in_block;

    /// softmax start
    __shared__ float m[26];
    __shared__ float l[26];

    if (threadIdx.x < 26) {
        m[threadIdx.x] = -INFINITY;
        l[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    // 每个 warp (threadIdx.x=0~31) 负责若干行
    int ROWS = 26;
    int WARPS = blockDim.y;
    int ROWS_PER_WARP = (ROWS + WARPS - 1) / WARPS; // ceil(26/4)=7
    /// softmax end

    // initialize dst_shared
    int block_tid = threadIdx.x + blockDim.x * threadIdx.y;  // 0,1,2,...,127
    for (int i = block_tid; i < TOKENS_PER_Q * QHEAD_PER_BLOCK * HEAD_DIM/*=26x128*/; i += HEAD_DIM/*=128*/) {
        int row = i / 128;
        int col = i % 128;
        dst_shared[row][col] = 0.0f;
    }
    __syncthreads();

    /// 2. loop over 8 times to get global m&l
    for(int tile_id = 0; tile_id < LOOP_KV/*=8*/; tile_id++) {
        /// 2.1. Load K tile, 不同的 tile_id 对应 K 不同的 分块
        for(int token_id = 0; token_id < KV_TOKEN_TILE/*=32*/; token_id += 2) {
            int col_warp_id = warp_row_id + token_id;
            const half* src0 = K + s_col0_id + col_warp_id * 128 + tile_id * 32 * 128 + kv_head_block_offset *  256 * 128;
            const half* src1 = K + s_col1_id + col_warp_id * 128 + tile_id * 32 * 128 + kv_head_block_offset *  256 * 128;

            half2 data = make_half2(*src0, *src1);
            k_shared[col_warp_id][s_col0_id] = data.x;
            k_shared[col_warp_id][s_col1_id] = data.y;
        }
        __syncthreads();

        /// 2.2 compute Q_tile * K_tile 
        // block 中每一个 thread 负责一个 s_shared，然后需要loop stride 128
        const int tid2 = threadIdx.y * blockDim.x + threadIdx.x;
        for (int idx = tid2; idx < 13*2*32/*=可以达到的最大值*/; idx += 128/*=loop stride*/) {
            int i = idx / 32;
            int j = idx % 32;
            float sum = 0.0f;
#pragma unroll
            for (int hd = 0; hd < 128; hd += 2) {  // vectorise
                const half2* q = reinterpret_cast<const half2*>(&q_shared[i][hd]); // hd 连续变化
                const half2* k = reinterpret_cast<const half2*>(&k_shared[j][hd]); // hd 连续变化
                sum += __half2float(q->x) * __half2float(k->x) + __half2float(q->y) * __half2float(k->y);
            }
            s_shared[i][j] = sum;
#if defined(MY_OPS_DEBUG)
            // 仅 debug 编译路径下，把 logits(对应的 s_shared) dump 出来
            if (s_out != nullptr) {
                int block = blockIdx.x;
                int tile  = tile_id;
                int idx4  = (((block * LOOP_KV) + tile) * (TOKENS_PER_Q * QHEAD_PER_BLOCK) + i) * KV_TOKEN_TILE + j;
                s_out[idx4] = sum;
            }
#endif
        }
        __syncthreads();

        /// 2.3 紧接着更新基于当前 s_shared 的 m 和 l
        int warp_id = threadIdx.y;         // 0~3
        int lane_id = threadIdx.x;         // 0~31

        int row_start = warp_id * ROWS_PER_WARP;
        int row_end = min(row_start + ROWS_PER_WARP, ROWS);

        // r 表示这个warp 负责的第一行，warp=0 负责前7行，warp2 负责第8~14行
        for (int r = row_start; r < row_end; ++r) {
            // 2.3.1: 所有 32 线程加载 s_shared[r][lane_id]
            float s = s_shared[r][lane_id];

            // 2.3.2: Warp 内归约求 max
            // float row_max = warp_reduce_down_max(s);
            float row_max = warp_reduce_xor_max(s);

            // 2.3.3: Warp 内计算 exp(s - row_max) 并归约 sum
            float exp_val = expf(s - row_max);
#if defined(MY_OPS_DEBUG)
            // 仅 debug 编译路径下，把每个 (row, col) 的 exp(logit) dump 出来
            if (exp_val_out != nullptr) {
                int idx4 = (((blockIdx.x * LOOP_KV) + tile_id) * 26 + r) * 32 + lane_id;
                exp_val_out[idx4] = exp_val;
            }
#endif
            // float row_sum = warp_reduce_down_sum(exp_val);
            float row_sum = warp_reduce_xor_sum(exp_val);

            // 2.3.4: 只让 lane 0 更新全局 m/l
            // https://arxiv.org/pdf/1805.02867 
            if (lane_id == 0) {
                float m_new = fmaxf(m[r], row_max);
                float scale_old = expf(m[r] - m_new);
                float scale_new = expf(row_max - m_new);
                l[r] = l[r] * scale_old + row_sum * scale_new;
                m[r] = m_new;

#if defined(MY_OPS_DEBUG)
                // 仅 debug 编译路径下 dump 归一化所需中间量
                if (row_sum_out != nullptr && scale_old_out != nullptr && scale_new_out != nullptr) {
                    int idx3 = ((blockIdx.x * LOOP_KV) + tile_id) * 26 + r; // [8,8,26]
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
    // debug: dump m/l after pass1
    if (m_out != nullptr && l_out != nullptr) {
        // 8 blocks in x dimension
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < 26; i += blockDim.x * blockDim.y) {
            m_out[blockIdx.x * 26 + i] = m[i];
            l_out[blockIdx.x * 26 + i] = l[i];
        }
        __syncthreads();
    }
#endif

    /// 3. 
    // Loop_8_times_{
    //         load K_tile -> compute Q_tile * K_tile -> S_tile=softmax() -> load V_tile -> compute S_tile * V_tile -> write to dst_shared
    // }  
    // -> write dst_shared to global dst
    for(int tile_id = 0; tile_id < LOOP_KV/*=8*/; tile_id++) {
        // 3.1. load K_tile
        for(int token_id = 0; token_id < KV_TOKEN_TILE/*=32*/; token_id += 2) {
            int col_warp_id = warp_row_id + token_id;
            const half* src0 = K + s_col0_id + col_warp_id * 128 + tile_id * 32 * 128 + kv_head_block_offset *  256 * 128;
            const half* src1 = K + s_col1_id + col_warp_id * 128 + tile_id * 32 * 128 + kv_head_block_offset *  256 * 128;

            half2 data = make_half2(*src0, *src1);
            k_shared[col_warp_id][s_col0_id] = data.x;
            k_shared[col_warp_id][s_col1_id] = data.y;
        }
        __syncthreads();

        // 3.2. compute Q_tile * K_tile
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // [0,1,2,3...,127]
        for (int idx = tid; idx < 13*2*32/*=可以达到的最大值*/; idx += 128/*=loop stride*/) {
            int i = idx / 32;
            int j = idx % 32;
            float sum = 0.0f;
#pragma unroll
            for (int hd = 0; hd < 128; hd += 2) {  // vectorise 前提：q_shared & k_shared 的 hd 必须连续 ！
                const half2* q = reinterpret_cast<const half2*>(&q_shared[i][hd]); // hd 连续 ! 
                const half2* k = reinterpret_cast<const half2*>(&k_shared[j][hd]); // hd 连续 !
                sum += __half2float(q->x) * __half2float(k->x) + __half2float(q->y) * __half2float(k->y);
            }
            s_shared[i][j] = sum;  // 基于当前 s_shared 的 sum
        }
        __syncthreads();

        // for (int v_tile_id = 0; v_tile_id < LOOP_KV; ++v_tile_id){}
        // 3.3. apply m&l to s_shared to update s_shared=softmax()
        int warp_id = threadIdx.y;         // 0~3
        int lane_id = threadIdx.x;         // 0~31

        int row_start = warp_id * ROWS_PER_WARP;
        int row_end = min(row_start + ROWS_PER_WARP, ROWS);
        for (int r = row_start; r < row_end; ++r) {
            s_shared[r][lane_id] = expf(s_shared[r][lane_id] - m[r]) / l[r];
        }
        __syncthreads();

        // 3.4. load V_tile (方式同load K_tile 应为两者的shape相同)
        for(int token_id = 0; token_id < KV_TOKEN_TILE/*=32*/; token_id += 2) {
            int col_warp_id = warp_row_id + token_id; 
            const half* src0 = V + s_col0_id + col_warp_id * 128 + tile_id * 32 * 128 + kv_head_block_offset *  256 * 128;
            const half* src1 = V + s_col1_id + col_warp_id * 128 + tile_id * 32 * 128 + kv_head_block_offset *  256 * 128;
            half2 data = make_half2(*src0, *src1);

            v_shared[col_warp_id][s_col0_id] = data.x;
            v_shared[col_warp_id][s_col1_id] = data.y;
        }
        __syncthreads();

        // 3.5. compute dst = S_tile * V_tile。不是 “warp-level GEMM” 是 thread-level GEMM
        // loop over 26 rows to cover all output
        for (int dst_row = 0; dst_row < TOKENS_PER_Q * QHEAD_PER_BLOCK/*=26*/; ++dst_row ) {
            int col = block_tid; // 1,2,3,...127
            float sum = 0.0f;
            // dot product. 不可以向量化，因为v_shared[k][col] 的col 不是连续的, 同一个col不同行的元素相差128，不连续！
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                // sum += s_shared[dst_row][k] * v_shared[k][col];  // 注意 float * fp16
                float v = __half2float(v_shared[k][col]);
                sum += s_shared[dst_row][k] * v;
            }
            dst_shared[dst_row][col] += sum; // dst_shared 是最外层循环的累加结果 
        }
        __syncthreads();
    }

    // 3.6. write dst_shared -> global dst (col-major: [HEAD_DIM, TOKENS_PER_Q, Q_HEADS, 1])
    // dst_shared 的 26 行 = 2 个 qhead * 13 个 token，不能把 (qhead, token) 混成连续的 26 维
    if (block_tid < HEAD_DIM) {
        for (int r = 0; r < TOKENS_PER_Q/*=13*/ * QHEAD_PER_BLOCK/*=2*/; ++r) {
            const int qhead_local  = r / TOKENS_PER_Q;          // 0 , 1
            const int token_id     = r % TOKENS_PER_Q;          // 0,1,2,3,4,...,11,12
            // qhead_global:
            // (0,1,2,3...,6,7) * 2 + (0,1)
            const int qhead_global = blockIdx.x * QHEAD_PER_BLOCK + qhead_local; // 0..15
            if (qhead_global < 16) {
                const int hd = block_tid; // 0..127
                const int dst_id = hd + HEAD_DIM * token_id + HEAD_DIM * TOKENS_PER_Q * qhead_global;
                dst[dst_id] = dst_shared[r][hd];
            }
        }
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// 入口函数
//   - 输入 Q, K, V 为列主序 (column-major)，第一个维度元素变化最快
//   - Q 形状:  [128, 13, 16]   -> 128 (head_dim) x 13 (tokens_q) x 16 (q_heads)
//   - K/V 形状:[128, 256, 8]   -> 128 (head_dim) x 256 (tokens_kv) x 8 (kv_heads)
//   - dst 形状:[128, 13, 16]
//   - Q/K/V 在 host 侧以 2 字节标量存储
// -----------------------------------------------------------------------------
extern "C" void flash_attention(
                const uint16_t* q_host,  // host: e.g. np.float16, column-major
                const uint16_t* k_host,
                const uint16_t* v_host,
                float* dst_host,         // host: float32, column-major
                float scale) {

    using half_t = half;

    constexpr int HEAD_DIM       = 128;
    constexpr int TOKENS_PER_Q   = 13;
    constexpr int Q_HEADS        = 16;
    constexpr int KV_TOKENS      = 256;
    constexpr int KV_HEADS       = 8;

    const size_t q_elems   = static_cast<size_t>(HEAD_DIM) * TOKENS_PER_Q * Q_HEADS;   // 128 * 13 * 16
    const size_t kv_elems  = static_cast<size_t>(HEAD_DIM) * KV_TOKENS * KV_HEADS;     // 128 * 256 * 8
    const size_t dst_elems = q_elems;                                                  // 同 Q

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

    dim3 threads(32, 4, 1);  // 32 * 4 = 128 threads / block
    dim3 blocks(8, 1, 1);    // 8 blocks -> 8 * 2 = 16 Q heads（QHEAD_PER_BLOCK=2）

#if defined(MY_OPS_DEBUG)
    std::printf(
        "Kernel launch config: block=(%u,%u,%u), grid=(%u,%u,%u)\n",
        threads.x, threads.y, threads.z,
        blocks.x, blocks.y, blocks.z);
    std::fflush(stdout);
    // debug 构建下，flash_attention 本身只做 dst 计算，debug 参数保持空
    flash_attn_tile_kernel<<<blocks, threads, 0, stream>>>(d_q, d_k, d_v, d_dst, scale,
                                                     nullptr, nullptr, nullptr,
                                                     nullptr, nullptr, nullptr,
                                                     nullptr);
#else
    flash_attn_tile_kernel<<<blocks, threads, 0, stream>>>(d_q, d_k, d_v, d_dst, scale);
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

#if defined(MY_OPS_DEBUG)
// debug 入口：额外输出每个 block 的 m/l(float32, shape [8,26])和 S_tile(float32, shape [8,8,26,32])
extern "C" void flash_attention_debug(
                const uint16_t* q_host,
                const uint16_t* k_host,
                const uint16_t* v_host,
                float* dst_host,
                float scale,
                float* m_host,
                float* l_host,
                float* s_host,
                float* row_sum_host,
                float* scale_old_host,
                float* scale_new_host,
                float* exp_val_host) {

    using half_t = half;

    constexpr int HEAD_DIM       = 128;
    constexpr int TOKENS_PER_Q   = 13;
    constexpr int Q_HEADS        = 16;
    constexpr int KV_TOKENS      = 256;
    constexpr int KV_HEADS       = 8;
    constexpr int LOOP_KV        = 8;

    const size_t q_elems   = static_cast<size_t>(HEAD_DIM) * TOKENS_PER_Q * Q_HEADS;
    const size_t kv_elems  = static_cast<size_t>(HEAD_DIM) * KV_TOKENS * KV_HEADS;
    const size_t dst_elems = q_elems;
    const size_t ml_elems  = static_cast<size_t>(KV_HEADS) * 26;                 // 8*26
    const size_t s_elems   = static_cast<size_t>(KV_HEADS) * LOOP_KV * 26 * 32; // 8*8*26*32
    const size_t rsn_elems = static_cast<size_t>(KV_HEADS) * LOOP_KV * 26;      // 8*8*26
    const size_t exp_elems = static_cast<size_t>(KV_HEADS) * LOOP_KV * 26 * 32; // 8*8*26*32

    half_t* d_q   = nullptr;
    half_t* d_k   = nullptr;
    half_t* d_v   = nullptr;
    float*  d_dst = nullptr;
    float*  d_m   = nullptr;
    float*  d_l   = nullptr;
    float*  d_s   = nullptr;
    float*  d_row_sum = nullptr;
    float*  d_scale_old = nullptr;
    float*  d_scale_new = nullptr;
    float*  d_exp_val = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_q,   q_elems  * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_k,   kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_v,   kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, dst_elems * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_m,   ml_elems  * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_l,   ml_elems  * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_s,   s_elems   * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_row_sum,   rsn_elems   * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scale_old, rsn_elems   * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scale_new, rsn_elems   * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_exp_val,   exp_elems   * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_q,   q_host,  q_elems  * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_k,   k_host,  kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v,   v_host,  kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));

    dim3 threads(32, 4, 1);
    dim3 blocks(8, 1, 1);

    std::printf(
        "Kernel launch config: block=(%u,%u,%u), grid=(%u,%u,%u)\n",
        threads.x, threads.y, threads.z,
        blocks.x, blocks.y, blocks.z);
    flash_attn_tile_kernel<<<blocks, threads, 0, stream>>>(d_q, d_k, d_v, d_dst, scale,
                                                     d_m, d_l, d_s,
                                                     d_row_sum, d_scale_old, d_scale_new,
                                                     d_exp_val);
    LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, dst_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(m_host, d_m, ml_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(l_host, d_l, ml_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(s_host, d_s, s_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(row_sum_host, d_row_sum, rsn_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(scale_old_host, d_scale_old, rsn_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(scale_new_host, d_scale_new, rsn_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(exp_val_host, d_exp_val, exp_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeAsync(d_q,   nullptr));
    CUDA_CHECK(cudaFreeAsync(d_k,   nullptr));
    CUDA_CHECK(cudaFreeAsync(d_v,   nullptr));
    CUDA_CHECK(cudaFreeAsync(d_dst, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_m,   nullptr));
    CUDA_CHECK(cudaFreeAsync(d_l,   nullptr));
    CUDA_CHECK(cudaFreeAsync(d_s,   nullptr));
    CUDA_CHECK(cudaFreeAsync(d_row_sum,   nullptr));
    CUDA_CHECK(cudaFreeAsync(d_scale_old, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_scale_new, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_exp_val,   nullptr));
}
#endif  // MY_OPS_DEBUG
