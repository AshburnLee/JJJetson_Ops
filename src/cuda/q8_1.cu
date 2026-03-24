#include <cuda_runtime.h>
#include <cstring>       // memcpy
#include <stdio.h>
#include <vector>
#include <cstdint>       // uint8_t
#include <cuda_fp16.h>   // half2
#include "cuda_utils.cuh"

#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128  // 量化块大小
#define QK8_1 32  // 32个float值共享一组 scale和sum
#define WARP_SIZE 32

struct block_q8_1_mmq {
    half2 ds4[4];  // DS4 布局：4 个 (d, sum) 对，每一个（d,sum）对儿是一个half2
    int8_t qs[4 * QK8_1];  // 128 个待量化值, char 占 8 bit
};

__global__ void quantize_q8_1_kernel(
    const float* __restrict__ x,          // 输入浮点数组
    void* __restrict__ vy,                // 输出量化数组 输出是void
    const int64_t ne0_0,  // 2048  (第0维读)变化最快的维度元素个数           
    const int64_t s0_1,   // 2048  第一维度 相邻元素偏移个数 2048个
    const int64_t s0_2,   // 26624 第二维度 相邻元素偏移个数 26624个
    const int64_t s0_3,   // 26624 第三维度 相邻元素偏移个数 26624个
    const int64_t ne0,   // 2048  变化最快的维度元素个数
    const int64_t ne1,   // 13    第二个维度元素个数
    const int64_t ne2    // 1     第三个维度元素个数
) {
    // 2. 布局参数，不同的layout需要不同的分组大小，控制线程id的范围
    // 2. 表示多少个输入值共享一个额放缩因子
    constexpr int vals_per_scale = 32; 
    constexpr int vals_per_sum = 32;  

    // 3. 计算线程索引 - 具体是 grid 中一列的索引 4 个block(128 个 thread),
    // 线程索引 *4 就是访问的数据索引
    // scope 是一列即 4 个block的所有线程
    // grid 的y方向有4个block，每个block有128个线程，每个thread负责4个float，正好覆盖输入变化最快的维度 2048。
    const int64_t i0 = ((int64_t)blockIdx.y * blockDim.x  + threadIdx.x) * 4;
    if (i0 >= ne0) return;  // 超出范围退出

    // 输入数据每一个维度的索引
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;


    const float4* x4 = (const float4*)x;

    float4 xi = (i0 < ne0_0) ? x4[(i3 * s0_3 + i2 * s0_2 + i1 * s0_1 + i0) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    float amax = fmaxf(fabsf(xi.x), fmaxf(fabsf(xi.y), fmaxf(fabsf(xi.z), fabsf(xi.w))));

#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum = 0.0f;
    sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }

    // 数学中：scale = amax / 127.0f
    const float d_inv = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    // 目标类型 char4
    char4 q;
    // 数学中：q_i = round(x_i / scale)
    // roundf: float 类型浮点数 四舍五入到 最近的整数
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);

    // 7. 计算输出块索引 和 其中的 qs 索引
    block_q8_1_mmq* y = (block_q8_1_mmq*)vy;

    const int64_t id_block_ext = blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x / QK8_1); 
    const int64_t id_block = (i0 / (4 * QK8_1)) * ne1 + blockIdx.x;  // 自然
    const int64_t id_q8_1_block = id_block + id_block_ext;
    const int64_t id_qs = i0 % (4 * QK8_1);  // 自然

    // id_q8_1_block 是输出 qblock 的索引

    // 8. 存储量化值
    char4* y_qs4 = (char4*)y[id_q8_1_block].qs;
    y_qs4[id_qs / 4] = q;  // qs 类型是1byte, q 的类型是4byte，故id_qs / 4。

    // 存储缩放因子 d_inv 和 sum
    // 这个方法就表示，只存储在指定的
    if (id_qs % 32 == 0) {
        y[id_q8_1_block].ds4[id_qs / 32] = make_half2(1.0f / d_inv, sum);
    }
}

// in_dims = [2048, 13, 1, 1] (col-major，dim0 最快变化)
// 使用 uint8 缓冲
extern "C" void q8_1(float* input, uint8_t* output, std::vector<int>& input_dims) {
    const int64_t ne0_0 = input_dims[0];  // 2048
    const int64_t ne0_1 = input_dims[1];  // 13
    const int64_t ne0_2 = input_dims[2];  // 1
    const int64_t ne0_3 = input_dims[3];  // 1

    const int64_t s0_1 = ne0_0;                     // 2048
    const int64_t s0_2 = ne0_0 * ne0_1;              // 26624
    const int64_t s0_3 = ne0_0 * ne0_1 * ne0_2;       // 26624

    const int64_t n_elem   = ne0_0 * ne0_1 * ne0_2 * ne0_3;   // 26624
    // block_q8_1_mmq 的个数：2048 对应 208 个这样的block
    const int64_t n_blocks = ne0_1 * ((ne0_0 + 128 - 1) / 128);    // 13*16 = 208

    float *d_x = nullptr;
    struct block_q8_1_mmq *d_y = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_x, n_elem * sizeof(float),stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, n_blocks * sizeof(block_q8_1_mmq),stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, input, n_elem * sizeof(float), cudaMemcpyHostToDevice,stream));

    const int block_size = CUDA_QUANTIZE_BLOCK_SIZE_MMQ;   // 128
    const int blocks_y   = (ne0_0 + 4*block_size - 1) / (4*block_size); // 4
    const dim3 blocks(ne0_1, blocks_y, ne0_2*ne0_3);         // (13,4,1)
    const dim3 threads(block_size, 1, 1);                 // (128,1,1)

    quantize_q8_1_kernel<<<blocks, threads, 0, stream>>>(
            d_x, d_y,
            ne0_0, s0_1, s0_2, s0_3,
            ne0_0, ne0_1, ne0_2);
    LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpyAsync(output, d_y, n_blocks * sizeof(block_q8_1_mmq),
        cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeAsync(d_x,nullptr));
    CUDA_CHECK(cudaFreeAsync(d_y,nullptr));
}

// 在调用 q8_1 之后，解析 output，将uint8_t的output parse得到 unit8_t 的 qs 和 16位的 scale 和 16位的 sum
// 只不过，用 fp32 来存储 scale 和 sum，因为后续的计算需要用到 float32 的精度
extern "C" void parse_q8_1_output(
    const uint8_t* input,           // output from q8_1
    int8_t* qs_out,                 // [n_blocks * 128]
    float* scale_out,               // [n_blocks * 4]
    float* sum_out,                 // [n_blocks * 4]
    int64_t n_blocks                // 原始 float 可以分为多少个 128，208
) {
    auto* blocks = reinterpret_cast<const block_q8_1_mmq*>(input);
    for (int64_t b = 0; b < n_blocks; ++b) {
        // 1. 拷贝 128 个 int8
        std::memcpy(qs_out + b*128, blocks[b].qs, 128);

        // 2. 解析 4 个 half2 得到 scale 和 sum
        for (int g = 0; g < 4; ++g) {
            // 将 16 bit(half) 的 scale 转换为 float32 的值, 数值转换
            // 正是 __half2float 的含义，将 FP16 转换为 FP32
            scale_out[b*4 + g] = __half2float(blocks[b].ds4[g].x);  // d
            sum_out[b*4 + g]   = __half2float(blocks[b].ds4[g].y);  // sum
        }
    }
}
