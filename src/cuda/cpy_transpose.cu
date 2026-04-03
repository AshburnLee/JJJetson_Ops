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

    // 逻辑 (dim0, dim1, batch)：sx/sy 覆盖子块的两维；blockIdx.x -> dim0，blockIdx.y -> dim1
    const int sx = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int sy = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;
    // 交换 dim0、dim1, tx 沿原 dim1，ty 沿原 dim0
    const int tx = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.x; //画出示意图就明了了
    const int ty = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.y; //画出示意图就明了了

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
            if (sx < ne0_0 && sy + j < ne0_1) {
                const int row_id = threadIdx.y + j; // (0~7) + j
                const int col_id = threadIdx.x * sizeof(float)/sizeof(T); 
                // 列主序 (dim0,dim1,batch)
                T *tile2 = reinterpret_cast<T*>(tile[row_id]);
                const int src_idx = sx + (sy + j) * ne0_0 + imat * (ne0_0 * ne0_1);
                tile2[col_id] = src[src_idx];
            }
        }

        __syncthreads();

#pragma unroll
        // j = 0/8/16/24，将 tile[row][col] 内容写入 dst[]，读转之后，故与 tx,ty 相关
        // row_id/col_id 是数据 tile 的索引；dst_idx 是目标数据索引
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D/*=32*/; j += CUDA_CPY_BLOCK_ROWS/*=8*/) {
            if (tx < ne0_1 && ty + j < ne0_0) {
                const int row_id = threadIdx.x;
                const int col_id = (threadIdx.y + j) * sizeof(float)/sizeof(T);
                const int dst_idx = tx + (ty + j) * ne0_1 + imat * (ne0_0 * ne0_1);
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

    // src_dims = (dim0, dim1, batch)；转置交换 dim0 与 dim1
    const int ne0_0 = src_dims[0];
    const int ne0_1 = src_dims[1];
    const int ne0_2 = src_dims[2];

    const int n_elem = ne0_0 * ne0_1 * ne0_2; // ne[3] is 1 assumed

    char* d_csrc = nullptr;
    char* d_cdst = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMallocAsync(&d_csrc, n_elem * sizeof(float),stream));
    CUDA_CHECK(cudaMallocAsync(&d_cdst, n_elem * sizeof(float),stream));
    CUDA_CHECK(cudaMemcpyAsync(d_csrc, csrc, n_elem * sizeof(float), cudaMemcpyHostToDevice,stream));

    // grid.x -> dim0，grid.y -> dim1，grid.z -> batch 条带
    dim3 blocks( (ne0_0 + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D,
                    (ne0_1 + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D,
                    (n_elem/(ne0_0*ne0_1) + CUDA_CPY_BLOCK_NM - 1) / CUDA_CPY_BLOCK_NM);
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
