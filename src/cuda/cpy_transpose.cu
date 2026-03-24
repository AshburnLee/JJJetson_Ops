#include <cuda_runtime.h>
#include <vector>
#include "cuda_utils.cuh"

#define CUDA_CPY_BLOCK_SIZE 64
#define CUDA_CPY_TILE_DIM_2D 32  // 2D tile dimension for transposed blocks
#define CUDA_CPY_BLOCK_NM 8      // block size of 3rd dimension if available
#define CUDA_CPY_BLOCK_ROWS 8    // block dimension for marching through rows

template <typename T>
static __global__ void cpy_transpose_kernel(
        const char * csrc,
        char * cdst,
        const int ne,
        const int ne0_0,
        const int ne0_1,
        const int ne0_2) {
    // 将按字节访问的内存 按 T 元素读写，即步长为sizeof(T)
    const T* src = reinterpret_cast<const T*>(csrc);
    T* dst = reinterpret_cast<T*>(cdst);

    const int64_t nmat = ne / (ne0_0 * ne0_1);

    // 转置前索引
    const int sx = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int sy = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;
    // 转置之后的索引 从读写示意图推导出
    const int tx = blockIdx.y/*画出示意图就明了了*/ * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int ty = blockIdx.x/*画出示意图就明了了*/ * CUDA_CPY_TILE_DIM_2D + threadIdx.y;

    // block wise 每一个block都有一个 [32][32+1] 的 tile
    __shared__ float tile[CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D + 1];

#pragma unroll
    for (int i = 0; i < CUDA_CPY_BLOCK_NM/*=8*/; ++i) {
        const unsigned int imat = blockIdx.z * CUDA_CPY_BLOCK_NM/*=8*/ + i;
        if (imat >= nmat)
            break;

#pragma unroll
        // j = 0/8/16/24，将 src[] 内容循环地读到 tile[row][col]，读转之前，故与 x，y 相关
        // row_id/col_id 是数据 tile 的索引；src_idx 是输入数据索引
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D/*=32*/; j += CUDA_CPY_BLOCK_ROWS/*=8*/) {
            if(sx < ne0_1 && sy + j < ne0_0) {
                const int row_id = threadIdx.y + j; // (0~7) + j
                const int col_id = threadIdx.x * sizeof(float)/sizeof(T); 
                // 转之前 [ne0_0][ne0_1][ne0_2] 
                T *tile2 = reinterpret_cast<T*>(tile[row_id]);
                // row-major indx
                const int src_idx = sx + (sy + j) * /*num_col=*/ne0_1 + imat * ne0_0 * ne0_1;
                // col-major indx
                // const int src_idx = (sy + j)              // 行 r
                //                     + sx * ne0_0           // 列 c
                //                     + imat * ne0_0 * ne0_1; // batch m
                tile2[col_id] = src[src_idx];
            }
        }

        __syncthreads();

#pragma unroll
        // j = 0/8/16/24，将 tile[row][col] 内容写入 dst[]，读转之后，故与 tx,ty 相关
        // row_id/col_id 是数据 tile 的索引；dst_idx 是目标数据索引
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D/*=32*/; j += CUDA_CPY_BLOCK_ROWS/*=8*/) {
            if (ty + j < ne0_1 && tx < ne0_0) {
                const int row_id = threadIdx.x; // 0~31
                const int col_id = (threadIdx.y + j) * sizeof(float)/sizeof(T);
                // 转之后 [ne0_1][ne0_0][ne0_2]
                // row-major indx
                const int dst_idx = tx + (ty + j) * /*转之后num_col=*/ne0_0 + imat * ne0_0 * ne0_1;
                // col-major indx
                // const int dst_idx = (ty + j)              // 行' = 列 c
                //                     + tx * ne0_1           // 列' = 行 r
                //                     + imat * ne0_0 * ne0_1; // batch m 不变
                const T *tile2 = reinterpret_cast<const T*>(tile[row_id]);
                dst[dst_idx] = tile2[col_id];
            }
        }
    }
}

extern "C" void cpy_transpose(
        const char * csrc, 
        char * cdst,
        const std::vector<int>& src_dims) {

    // 转置 ne0_0 和 ne0_1
    const int ne0_2 = src_dims[0];
    const int ne0_0 = src_dims[1];
    const int ne0_1 = src_dims[2];

    const int n_elem = ne0_0 * ne0_1 * ne0_2; // ne[3] is 1 assumed

    char* d_csrc = nullptr;
    char* d_cdst = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMallocAsync(&d_csrc, n_elem * sizeof(float),stream));
    CUDA_CHECK(cudaMallocAsync(&d_cdst, n_elem * sizeof(float),stream));
    CUDA_CHECK(cudaMemcpyAsync(d_csrc, csrc, n_elem * sizeof(float), cudaMemcpyHostToDevice,stream));

    // num_blocks: ((ne0_1 + 32-1)/32, (ne0_0 + 32-1)/32, (ne/(ne0_1*ne0_0) + 8-1)/8)
    dim3 blocks( (ne0_1 + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D,
                    (ne0_0 + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D,
                    (n_elem/(ne0_1*ne0_0) + CUDA_CPY_BLOCK_NM - 1) / CUDA_CPY_BLOCK_NM);
    // num_threads: (32,8,1)
    dim3 threads(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS, 1);

    cpy_transpose_kernel<float><<<blocks, threads, 0, stream>>>
        (d_csrc, d_cdst, n_elem, ne0_0, ne0_1, ne0_2);
    LAUNCH_CHECK();
    
    CUDA_CHECK(cudaMemcpyAsync(cdst, d_cdst, n_elem * sizeof(float),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeAsync(d_csrc, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_cdst, nullptr));  
}
