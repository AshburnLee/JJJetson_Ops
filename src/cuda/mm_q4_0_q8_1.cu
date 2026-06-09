// Q4_0 权值 @ Q8_1(MMQ布局) 激活 的矩阵乘
// W 来自模型的 Q4_0 量化值，X 需要量化成 Q8_1，后做MMQ
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cuda_utils.cuh"

// Q4_0 权值 quant block (block_q4_0)
#define QK4_0 32 // 本 kernel Q4_0 quant block 粒度，沿 K 覆盖 32 个量化值（权值）
#define QR4_0 2  // 压缩比：1 字节存 2 个 4-bit，故 QR=2
// 打包粒度：qs 按 int32 载入做 dp4a 时，每 quant block 占 4 个 int32
#define QI4_0 (QK4_0 / (4 * QR4_0)) // 槽（=32/(4×2)）

// Q8_1 子块 (32 宽一组 d/sum)
#define QK8_1 32 // 本 kernel Q8_1 quant block 粒度，沿 K 覆盖 32 个量化值（激活值）
#define QR8_1 1  // 1 个 int8 对应 1 个激活，故 QR=1
// 8, 每子块占 8 个 int32 槽（=32/(4×1)），供 smem_x_qs 按 int32 读取
#define QI8_1 (QK8_1 / (4 * QR8_1))

#define QK8_1_MMQ 128
#define QK8_1_MMQ_SUB 4

#define WARP_SIZE 32

#define VDR_Q4_0_Q8_1_MMQ 4 // 每次调用 dot，每个 thread 连续处理多少个 int32

#define MMQ_TILE_TOKEN 4    // 每个 CUDA block 覆盖 Y 的 token 列数（原 MMQ_X）
#define MMQ_TILE_HEADDIM 32 // 每个 CUDA block 覆盖 Y 的行数（原 MMQ_Y）
#define NWARPS_Q4_0 4

// 存储粒度 32 个量化值，占 16+2=18 字节
struct block_q4_0 {
    half d;
    uint8_t qs[QK4_0 / 2]; // qs[16] 16字节，每字节（8-bit）含 2个 4-bit 对象，故 32 个量化值
};

// 存储粒度 128 个数，占 128+16=144 字节
struct block_q8_1_mmq {
    half2 ds4[QK8_1_MMQ_SUB];
    int8_t qs[QK8_1_MMQ_SUB * QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "block_q8_1_mmq size mismatch");
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");

// 打包 Q4_0 权值读取，（来自 vecdotq.cuh）, uint8 -> int32
static __device__ __forceinline__ int32_t get_int_from_uint8(const uint8_t *bytes, const int i32) {
    const uint16_t *p16 = (const uint16_t *)(bytes + sizeof(int32_t) * i32);
    int32_t x32 = 0;
    x32 |= static_cast<int32_t>(p16[0]) << 0;
    x32 |= static_cast<int32_t>(p16[1]) << 16;
    return x32;
}

// 打包 Q8_1 激活值，int8 -> int32
static __device__ __forceinline__ int32_t get_int_from_int8_aligned(const int8_t *bytes,
                                                                    const int i32) {
    return *reinterpret_cast<const int32_t *>(bytes + sizeof(int32_t) * i32);
}

// Q4_0（权值 W）x Q8_1（激活 X）量化值点积；用激活 X 的 scale（d）与 sum 做 zero-point 校正
// sm_61 及以上支持 __dp4a
/*
该函数做的是一个 Q4_0 量化块和一个 Q8_1 量化块的元素（32个量化值）按在 W 和 X 中的位置做点积，
所以点积结果是 32 个量化值的乘累加，
函数返回结果是 float 并用 Q8_1 的 scale 和 sum 做零点修正。
*/
template <int vdr>
static __device__ __forceinline__ float
q4_0_q8_1_vec_dot_impl(const int32_t *w_qs, const int32_t *x_qs_packed, const float w_scale,
                       const half2 x_ds) {
    int32_t sumi = 0;
#pragma unroll
    // __dp4a() 要求一次调用只能算 4 个数（int8），所以需要 vdr（4）
    // 次循环，4*8（两次__dp4a()）=32个数点积
    for (int i = 0; i < vdr /*=4*/; ++i) {
        // 1 个 int32 含 8 个量化值，拆成 w_lo（间隔的4个量化值）/w_hi（间隔的4个量化值） 各 4 个供
        // __dp4a
        const int32_t w_lo = w_qs[i] & 0b00001111000011110000111100001111; // 每8-bit的低 4-bit
        // const int32_t w_hi = w_qs[i] & 0b11110000111100001111000011110000;    // error 不等价
        const int32_t w_hi =
            (w_qs[i] >> 4) & 0b00001111000011110000111100001111; // 每8-bit的高 4-bit
        // x_qs_packed 是 Q8_1 的 qs 按 int32 读入，
        // 对应位置元素 dot，结果 sumi 就是 8（低4+高4）个 int32 的 dot 结果
        sumi = __dp4a(w_lo, x_qs_packed[2 * i + 0], sumi); // sm_61 及以上支持 __dp4a
        sumi = __dp4a(w_hi, x_qs_packed[2 * i + 1], sumi);
    }
    const float2 x_dsf = __half22float2(x_ds);
    return w_scale * (sumi * /*激活的 scale=*/x_dsf.x - (8 * vdr / QI4_0) * /*激活的 sum=*/x_dsf.y);
}

// 在函数内声明 __shared__，再用 int32_t**/float** 把地址回传给 kernel 里的指针
template <int tile_rows>
static __device__ __forceinline__ void allocate_smem_w(int32_t **smem_w_qs, float **smem_w_scale) {
    __shared__ int32_t smem_w_qs_buf[tile_rows * WARP_SIZE + tile_rows];
    __shared__ float smem_w_scale_buf[tile_rows * (WARP_SIZE / QI4_0) + tile_rows / QI4_0];
    *smem_w_qs = smem_w_qs_buf;
    *smem_w_scale = smem_w_scale_buf;
}

// 将 w load 到对应的 smem：smem_w_qs & smem_w_scale
template <int tile_rows, int nwarps, bool need_check>
static __device__ __forceinline__ void
load_smem_w(const block_q4_0 *__restrict__ w, int32_t *__restrict__ smem_w_qs,
            float *__restrict__ smem_w_scale, const int row_offset, const int row_max,
            const int k_slot, const int w_qblocks_per_row) {
    const int w_qblock_idx = k_slot / QI4_0;
    const int w_qs_subslot = k_slot % QI4_0;

#pragma unroll
    for (int r0 = 0; r0 < tile_rows; r0 += nwarps) {
        int row_w = r0 + row_offset;
        if (need_check) {
            row_w = min(row_w, row_max);
        }
        const block_q4_0 *w_blk = w + row_w * w_qblocks_per_row + w_qblock_idx;
        smem_w_qs[row_w * (WARP_SIZE + 1) + k_slot] = get_int_from_uint8(w_blk->qs, w_qs_subslot);
    }

    const int qblocks_per_smem_row = WARP_SIZE / QI4_0;
    const int w_scale_qblock = k_slot % qblocks_per_smem_row;

#pragma unroll
    for (int r0 = 0; r0 < tile_rows; r0 += nwarps * QI4_0) {
        int row_w = r0 + row_offset * QI4_0 + k_slot / qblocks_per_smem_row;
        if (need_check) {
            row_w = min(row_w, row_max);
        }
        const block_q4_0 *w_blk = w + row_w * w_qblocks_per_row + w_scale_qblock;
        smem_w_scale[row_w * (WARP_SIZE / QI4_0) + row_w / QI4_0 + w_scale_qblock] =
            __half2float(w_blk->d);
    }
}

/*
数值定位 -> 构建与 Q4_0 同量化值数量的 Q8_1（x_qs_packed）-> 执行 vec-dot
*/
static __device__ __forceinline__ float
vec_dot(const int32_t *__restrict__ smem_w_qs, const float *__restrict__ smem_w_scale,
        const int32_t *__restrict__ smem_x_qs, const half2 *__restrict__ smem_x_ds, const int row_w,
        const int col_batch, const int k_slot) {
    const int x_qs_base = k_slot % (QI8_1 / 2) + QI8_1 * (k_slot / (QI8_1 / 2));

    // x_qs_packed 存 8 个 int32 槽，之后的循环写入，保证其中 32 个 int8 量化值
    int32_t x_qs_packed[2 * VDR_Q4_0_Q8_1_MMQ];
#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        // x_qs_packed 中 4 个值，再加 4 个值
        x_qs_packed[2 * l + 0] = smem_x_qs[col_batch * WARP_SIZE + (x_qs_base + l) % WARP_SIZE];
        x_qs_packed[2 * l + 1] =
            smem_x_qs[col_batch * WARP_SIZE + (x_qs_base + l + QI4_0 /*=4*/) % WARP_SIZE];
    }

    return q4_0_q8_1_vec_dot_impl<VDR_Q4_0_Q8_1_MMQ>(
        &smem_w_qs[row_w * (WARP_SIZE + 1) + k_slot], x_qs_packed,
        smem_w_scale[row_w * (WARP_SIZE / QI4_0) + row_w / QI4_0 + k_slot / QI4_0],
        smem_x_ds[col_batch * (WARP_SIZE / QI8_1) + (2 * k_slot / QI8_1) % (WARP_SIZE / QI8_1)]);
}

// mul_mat_q 核心：x_q8_1 为 block_q8_1_mmq（与 q8_1.cu 对齐）
template <bool need_check>
__global__ void q4_0_q8_1_mat_mul_kernel(const void *__restrict__ w_q4_0,
                                         const void *__restrict__ x_q8_1, float *__restrict__ y,
                                         const int ncols_k, const int nrows_w, const int batch,
                                         const int nrows_y) {
    constexpr int tile_batch = MMQ_TILE_TOKEN;  // 4
    constexpr int tile_rows = MMQ_TILE_HEADDIM; // 32
    constexpr int nwarps = NWARPS_Q4_0;         // 4

    const block_q4_0 *w = (const block_q4_0 *)w_q4_0;
    const block_q8_1_mmq *x = (const block_q8_1_mmq *)x_q8_1;

    const int w_qblocks_per_row = ncols_k / QK4_0;
    const int x_mmq_superblocks_per_col = ncols_k / QK8_1_MMQ;
    const int q4_blocks_per_warp = WARP_SIZE / QI4_0; // 32/4=8

    const int row_y_0 = blockIdx.x * tile_rows;
    const int row_w_0 = row_y_0;
    const int col_y_0 = blockIdx.y * tile_batch;
    const int col_batch_0 = col_y_0;

    // 两个指向 smem 的 Device指针
    int32_t *smem_w_qs = nullptr;
    float *smem_w_scale = nullptr;
    allocate_smem_w<tile_rows>(&smem_w_qs, &smem_w_scale);
    // 大小 4列x32行，类型 int32_t 非 int8，故每列存 128 个量化值
    __shared__ int32_t smem_x_qs[tile_batch * WARP_SIZE];
    __shared__ half2 smem_x_ds[tile_batch * (WARP_SIZE / QI8_1)];

    // acc 为二维数组，为了更大的tile，每thread 负责多个累加器
    // 8 轮 ib0 循环 K 后，每个 thread 把自己的 acc 写到对应的
    float acc[tile_rows / WARP_SIZE][tile_batch / nwarps];
#pragma unroll
    for (int j = 0; j < tile_batch / nwarps /*=4/4*/; ++j) {
#pragma unroll
        for (int i = 0; i < tile_rows / WARP_SIZE /*=32/32*/; ++i) {
            acc[i][j] = 0.0f;
        }
    }

    //   ib0 = 0,8,16,24,32,40,48,56，共 8 轮，8×256=2048 表示W K方向本轮的起点id
    // 即每一次的 ib0 包含8个Q4_0
    for (int ib0 = 0; ib0 < w_qblocks_per_row /*=64*/; ib0 += q4_blocks_per_warp /*=8*/) {
        // 每一个 ib0（一轮K 字块），load W 到 smem中（smem_w_qs & smem_w_scale）
        load_smem_w</*32=*/tile_rows, /*4=*/nwarps, need_check>(
            w + row_w_0 * w_qblocks_per_row + ib0, smem_w_qs, smem_w_scale,
            /*row_offset=*/threadIdx.y,
            /*row_max=*/nrows_w - row_w_0 - 1,
            /*k_slot=*/threadIdx.x, w_qblocks_per_row);

#pragma unroll
        // ir = 0,1, 8个ib0 ，共16次ir循环，（这是一种分块方式，可以不分ir，直接点积）
        // smem_x_qs 容量是128个量化值（W smem中 存256个量化值），所以这里要分 ir =0,1
        for (int ir = 0;
             ir < QR4_0 /*=2*/ && ib0 + ir * q4_blocks_per_warp / QR4_0 < w_qblocks_per_row; ++ir) {
            const int kqs = ir * WARP_SIZE + threadIdx.x;
            const int x_subblock_in_strip = kqs / QI8_1;

#pragma unroll
            // load X 的 qs 到 smem_x_qs 中，由于复用 smem_x，必然与 ir 绑定
            // t = 0
            for (int t = 0; t < /*4=*/tile_batch; t += /*4=*/nwarps) {
                const int col_batch = min(col_batch_0 + threadIdx.y + t, batch - 1);
                const int k_qblock = ib0 + x_subblock_in_strip;
                const int mmq_super = k_qblock / QK8_1_MMQ_SUB;
                const int mmq_sub = k_qblock % QK8_1_MMQ_SUB;
                const block_q8_1_mmq *x_blk = &x[col_batch * x_mmq_superblocks_per_col + mmq_super];
                const int smem_x_idx = (threadIdx.y + t) * WARP_SIZE + kqs % WARP_SIZE;
                const int x_qs_chunk = mmq_sub * QI8_1 + (threadIdx.x % QI8_1);
                smem_x_qs[smem_x_idx] = get_int_from_int8_aligned(x_blk->qs, x_qs_chunk);
            }

#pragma unroll
            // load X 的 ds 到 smem_x_ds 中
            for (int ids0 = 0; ids0 < /*4=*/tile_batch; ids0 += /*4*8=*/nwarps * QI8_1) {
                const int col_tile =
                    (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE / QI8_1)) % tile_batch;
                const int x_ds_slot = threadIdx.x % (WARP_SIZE / QI8_1);
                const int col_batch = min(col_batch_0 + col_tile, batch - 1);
                const int k_qblock = ib0 + ir * (WARP_SIZE / QI8_1) + x_ds_slot;
                const int mmq_super = k_qblock / QK8_1_MMQ_SUB;
                const int mmq_sub = k_qblock % QK8_1_MMQ_SUB;
                const block_q8_1_mmq *x_blk = &x[col_batch * x_mmq_superblocks_per_col + mmq_super];
                smem_x_ds[col_tile * (WARP_SIZE / QI8_1) + x_ds_slot] = x_blk->ds4[mmq_sub];
            }

            __syncthreads();
            // ir=0 时，k_slot = 0,4,8,12
            // ir=1 时，k_slot = 16,20,24,28
            // 步长是 4 表示 一次点积固定消耗 W 上连续 4 个 int32 槽，即 32 个w的量化值
            // 32个量化值，正好对应最内层点积的粒度
            // k_slot 的值保证了 x tile与 w tile的“循环点积的节奏”相同
            for (int k_slot = /* =(0,1)*16 */ ir * WARP_SIZE / QR4_0;
                 k_slot < /* =((0,1)+1)*16=(16,32) */ (ir + 1) * WARP_SIZE / QR4_0;
                 k_slot += VDR_Q4_0_Q8_1_MMQ /*=4*/) {
#pragma unroll
                for (int j = 0; j < tile_batch; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < /*32=*/tile_rows; i += WARP_SIZE) {
                        // 计算 32 个 W 量化值 x 32 个 X 量化值
                        acc[i / WARP_SIZE][j / nwarps] +=
                            vec_dot(smem_w_qs, smem_w_scale, smem_x_qs, smem_x_ds, threadIdx.x + i,
                                    threadIdx.y + j, k_slot);
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < tile_batch; j += nwarps) {
        const int col_y = col_y_0 + j + threadIdx.y;
        if (col_y >= batch) {
            return;
        }
#pragma unroll
        for (int i = 0; i < tile_rows; i += WARP_SIZE) {
            const int row_y = row_y_0 + threadIdx.x + i;
            if (row_y >= nrows_y) {
                continue;
            }
            y[col_y * nrows_y + row_y] = acc[i / WARP_SIZE][j / nwarps];
        }
    }
}

static void launch_mat_mul_q4_0_q8_1(const void *d_w, const void *d_x_q8_1, float *d_y,
                                     const int ncols_k, const int nrows_w, const int batch,
                                     cudaStream_t stream) {
    constexpr int tile_batch = MMQ_TILE_TOKEN;
    constexpr int tile_rows = MMQ_TILE_HEADDIM;
    constexpr int nwarps = NWARPS_Q4_0;

    const int grid_x = (nrows_w + tile_rows - 1) / tile_rows;
    const int grid_y = (batch + tile_batch - 1) / tile_batch;
    const dim3 grid(grid_x, grid_y, 1);
    const dim3 block(WARP_SIZE, nwarps, 1);

    if (nrows_w % tile_rows == 0) {
        q4_0_q8_1_mat_mul_kernel<false>
            <<<grid, block, 0, stream>>>(d_w, d_x_q8_1, d_y, ncols_k, nrows_w, batch, nrows_w);
    } else {
        q4_0_q8_1_mat_mul_kernel<true>
            <<<grid, block, 0, stream>>>(d_w, d_x_q8_1, d_y, ncols_k, nrows_w, batch, nrows_w);
    }
    LAUNCH_CHECK();
}

extern "C" void q8_1(float *input, uint8_t *output, std::vector<int> &input_dims);

// weight_q4_0: Q4_0 权值，col-major，shape [nrows_out, ncols_k]
// input:       float 激活，col-major，dims = [ncols_k, batch, 1, 1]（与 q8_1 一致）
// output:      float [nrows_out, batch] col-major
extern "C" void mat_mul_q4_0_q8_1(const void *weight_q4_0, float *input, uint8_t *quant_workspace,
                                  float *output, std::vector<int> &input_dims, int nrows_out,
                                  int ncols_k) {
    const int64_t ne0 = input_dims[0];
    const int64_t ne1 = input_dims[1];
    const int64_t ne2 = input_dims.size() > 2 ? input_dims[2] : 1;
    const int64_t ne3 = input_dims.size() > 3 ? input_dims[3] : 1;

    if (ne0 != ncols_k) {
        std::fprintf(stderr, "mat_mul_q4_0_q8_1: input_dims[0]=%lld must equal ncols_k=%d\n",
                     (long long)ne0, ncols_k);
        return;
    }
    if ((ncols_k % QK4_0) != 0) {
        std::fprintf(stderr, "mat_mul_q4_0_q8_1: ncols_k must be multiple of %d\n", QK4_0);
        return;
    }
    if ((ncols_k % QK8_1_MMQ) != 0) {
        std::fprintf(stderr,
                     "mat_mul_q4_0_q8_1: ncols_k must be multiple of %d for block_q8_1_mmq\n",
                     QK8_1_MMQ);
        return;
    }

    const int batch = (int)(ne1 * ne2 * ne3);
    const int nrows_w = nrows_out;

    const int64_t n_input = ne0 * ne1 * ne2 * ne3;
    const int64_t n_weight_blocks = ((int64_t)nrows_out * ncols_k) / QK4_0;
    const int64_t n_quant_blocks = batch * ((ncols_k + QK8_1_MMQ - 1) / QK8_1_MMQ);
    const int64_t quant_bytes = n_quant_blocks * (int64_t)sizeof(block_q8_1_mmq);
    const int64_t n_output = (int64_t)nrows_out * batch;

    // q8_1() 内部自管 H2D/D2H，使用 host 上的 input / quant_workspace
    q8_1(input, quant_workspace, input_dims);

    void *d_w = nullptr;
    void *d_x = nullptr;
    float *d_y = nullptr;
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMallocAsync(&d_w, n_weight_blocks * sizeof(block_q4_0), stream));
    CUDA_CHECK(cudaMallocAsync(&d_x, quant_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, n_output * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_w, weight_q4_0, n_weight_blocks * sizeof(block_q4_0),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, quant_workspace, quant_bytes, cudaMemcpyHostToDevice, stream));

    // MMQ 入口
    launch_mat_mul_q4_0_q8_1(d_w, d_x, d_y, ncols_k, nrows_w, batch, stream);

    CUDA_CHECK(
        cudaMemcpyAsync(output, d_y, n_output * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_w, stream));
    CUDA_CHECK(cudaFreeAsync(d_x, stream));
    CUDA_CHECK(cudaFreeAsync(d_y, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    (void)n_input;
}

// 纯 Device 入口：不做 X 量化、不做 mem 分配、不做 H2D/D2H，只 launch kernel
// 用途：推理 pipeline 中 W/X 已在 GPU 上，只需 Y = W*X
/*
调用方约束：
  d_w_q4_0：Q4_0 权值，col-major [nrows_out, ncols_k]，与 block_q4_0 布局一致
  d_x_q8_1：已量化为 block_q8_1_mmq 布局（与 q8_1.cu 一致），非原始 float
  d_x_q8_1 缓冲 >= batch * ceil(ncols_k / QK8_1_MMQ) * sizeof(block_q8_1_mmq)
           = batch * ceil(ncols_k / 128) * 144 字节
  ncols_k 须为 QK4_0(32) 与 QK8_1_MMQ(128) 的公倍数约束（同 mat_mul_q4_0_q8_1 校验）
  d_y：device 上 float 输出 [nrows_out, batch] col-major
*/

extern "C" void mat_mul_q4_0_q8_1_device(const void *d_w_q4_0, const void *d_x_q8_1, float *d_y,
                                         const int ncols_k, const int nrows_out, const int batch,
                                         cudaStream_t stream) {
    launch_mat_mul_q4_0_q8_1(d_w_q4_0, d_x_q8_1, d_y, ncols_k, nrows_out, batch, stream);
}
