#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

#define CUDA_CPY_BLOCK_SIZE 64
#define CUDA_CPY_TILE_DIM_2D 32 // 2D tile dimension for transposed blocks
#define CUDA_CPY_BLOCK_NM 8     // block size of 3rd dimension if available
#define CUDA_CPY_BLOCK_ROWS 8   // block dimension for marching through rows

template <typename T>
static __global__ void cpy_transpose_kernel(const char *csrc, char *cdst, const int ne,
                                            const int ne0_0, const int ne0_1, const int ne0_2) {
    // 将按字节访问的内存 按 T 元素读写，即步长为sizeof(T)
    const T *src = reinterpret_cast<const T *>(csrc);
    T *dst = reinterpret_cast<T *>(cdst);

    const int64_t nmat = ne / (ne0_0 * ne0_1);

    // 逻辑 (dim0, dim1, batch)：sx/sy 覆盖子块的两维；blockIdx.x -> dim0，blockIdx.y -> dim1
    // sx: (0,1) * 32 + (0,1,2,...,31) => (0,1,2,...,31),(32,33,...,63)
    // sy: (0) * 32 + (0,1,2,...,7)  => (0,1,2,...,7)
    const int sx = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int sy = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;
    // 交换 dim0、dim1, tx 沿原 dim1，ty 沿原 dim0
    const int tx = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.x; // 画出示意图就明了了
    const int ty = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.y; // 画出示意图就明了了

    // block wise 每一个block都有一个 [32][32+1] 的 tile
    __shared__ float tile[CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D + 1];

#pragma unroll
    for (int i = 0; i < CUDA_CPY_BLOCK_NM /*=8*/; ++i) {
        const unsigned int imat = blockIdx.z * CUDA_CPY_BLOCK_NM /*=8*/ + i;
        if (imat >= nmat)
            break;

#pragma unroll
        // j = 0/8/16/24：沿 dim1 分条带；数据写入 tile[t_dim1][t_dim0]
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D /*=32*/; j += CUDA_CPY_BLOCK_ROWS /*=8*/) {
            // src 数据逻辑下标 (i0, i1)= (sx, (sy+j)), 边界处理，适配多种数据规模
            if (sx < ne0_0 && (sy + j) < ne0_1) {
                // tile index：tile[t_dim1][t_dim0]
                const int t_dim1 = threadIdx.y + j;                         // (0~7) + j
                const int t_dim0 = threadIdx.x * sizeof(float) / sizeof(T); // (0,1,..,31)
                // 与《核心公式》呼应：累加{每一个维度idx * 该维度数据stride}
                // 另，src_idx 中 sx 变化最快，而 sx（含 threadIdx.x）是连续1变化的 =>
                // “global是合并读”
                const int src_idx = sx + (sy + j) * ne0_0 + imat * (ne0_0 * ne0_1);
                // 逻辑上，写入 tile[t_dim1][t_dim0] ,shared 是row-major，故这里
                // t_dim0 连续1变化，<=>正好 t_dim0 函 threadIdx.x 也是连续1变化。
                // 故 tile 的写是延行连续的。
                T *tile2 = reinterpret_cast<T *>(tile[t_dim1]);
                tile2[t_dim0] = src[src_idx];
            }
        }

        __syncthreads();

#pragma unroll
        //
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D /*=32*/; j += CUDA_CPY_BLOCK_ROWS /*=8*/) {
            // dst数据索引：(tx, (ty+j))
            if (tx < ne0_1 && (ty + j) < ne0_0) {
                // 访问 tile 的 idx
                // Key: 与load 解阶段的idx 互换，相当于在 tile 做idx互换
                const int t_dim1 = threadIdx.x; // 连续1变化
                const int t_dim0 = (threadIdx.y + j) * sizeof(float) / sizeof(T);
                // 与《核心公式》呼应
                // dst 的写入，因tx 是连续1变化，故写入“Global是合并写”
                const int dst_idx = tx + (ty + j) * ne0_1 + imat * (ne0_0 * ne0_1);

                // 逻辑上 tile[t_dim1][t_dim0], t_dim0 连续 32 变化，故事 stride
                // access，stride大小是33
                const T *tile2 = reinterpret_cast<const T *>(tile[t_dim1]);
                dst[dst_idx] = tile2[t_dim0];
            }
        }
    }
}

extern "C" void cpy_transpose(const char *csrc, char *cdst, const std::vector<int> &src_dims) {

    // src_dims = (dim0, dim1, batch)；转置交换 dim0 与 dim1
    const int ne0_0 = src_dims[0];
    const int ne0_1 = src_dims[1];
    const int ne0_2 = src_dims[2];

    const int n_elem = ne0_0 * ne0_1 * ne0_2; // ne[3] is 1 assumed

    char *d_csrc = nullptr;
    char *d_cdst = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMallocAsync(&d_csrc, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_cdst, n_elem * sizeof(float), stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_csrc, csrc, n_elem * sizeof(float), cudaMemcpyHostToDevice, stream));

    // num_threads: (32,8,1)
    dim3 threads(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS, 1);
    // 根据数据量，和Block，决定grid
    dim3 blocks((ne0_0 + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D,
                (ne0_1 + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D,
                (n_elem / (ne0_0 * ne0_1) + CUDA_CPY_BLOCK_NM - 1) / CUDA_CPY_BLOCK_NM);
#if defined(MY_OPS_DEBUG)
    std::printf("Kernel launch config: Blcok=(%u,%u,%u), grid=(%u,%u,%u))\n", threads.x, threads.y,
                threads.z, blocks.x, blocks.y, blocks.z);
    std::fflush(stdout);
#endif
    cpy_transpose_kernel<float>
        <<<blocks, threads, 0, stream>>>(d_csrc, d_cdst, n_elem, ne0_0, ne0_1, ne0_2);
    LAUNCH_CHECK();

    CUDA_CHECK(
        cudaMemcpyAsync(cdst, d_cdst, n_elem * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeAsync(d_csrc, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_cdst, nullptr));
}
