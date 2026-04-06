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

// 4 组 Q8_1 为一个 结构，故有4个（d, m）对
struct block_q8_1_mmq {
    half2 ds4[4];  // DS4 布局：4 个 (d, sum) 对，每一个（d,sum）对儿是一个half2
    // qs 里连续 128 个 int8_t
    int8_t qs[4 * QK8_1];  // 128 个待量化值, char 占 1 字节
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

    // 3. 计算线程索引 - 具体是 grid 中一列的索引 4 个block(共 4*128 个 thread),
    // 线程索引 *4 即 每个线程负责沿第 0 维连续的 4 个 float
    // grid 的y方向有4个block，每个block有128个线程，每个thread负责4个float，正好覆盖输入变化最快的维度 2048。
    const int64_t i0 = ((int64_t)blockIdx.y * blockDim.x  + threadIdx.x) * 4;
    if (i0 >= ne0) return;  // 超出范围退出

    // 输入数据每一个维度的索引
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    // 每个 float4 对应内存里 连续 4 个 float，由一个 thread 一次读入
    const float4* x4 = (const float4*)x;

    float4 xi = (i0 < ne0_0) ? x4[(i3 * s0_3 + i2 * s0_2 + i1 * s0_1 + i0) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    // 每个 thread 负责4个数的局部max
    float amax = fmaxf(fabsf(xi.x), fmaxf(fabsf(xi.y), fmaxf(fabsf(xi.z), fabsf(xi.w))));
    // 8 个 thread cover 32个数，warp 中4路warp-level reduce 得到4组，每组32个数的max 
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    // 每个thread 负责4个数的局部sum
    float sum = 0.0f;
    sum = xi.x + xi.y + xi.z + xi.w;
    // 8 个 thread cover 32个数，warp 中4路 warp-level reduce 得到4组，每组32个数的sum  
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
    // qs 在结构体里是 int8_t 数组；每个线程写 4 个 int8，CUDA 无 int4，故用 char4（4个1 字节）
    //  nvcc 里多数情况下 char 按有符号 8 位用, 并且下游按 int8* 解引用，原本语义不变
    // 将同一块内存视为“每 4 字节一组”的向量，布局与连续 4 个 int8_t 一致，便于一次写入 q。
    char4* y_qs4 = (char4*)y[id_q8_1_block].qs;
    // 一个thread写入4个字节， id_qs/4 表示了这连续4个字节的首地址。
    y_qs4[id_qs / 4] = q;  

    // 存储缩放因子 d_inv 和 sum
    // 这个方法就表示，只存储在每一个 32 组的第一个位置
    if (id_qs % 32 == 0) {
        y[id_q8_1_block].ds4[id_qs / 32] = make_half2(1.0f / d_inv, sum);
    }
}
// 给出thread 和 输出的映射
/*
|线程索引 (`threadIdx.x`)| 输入索引 (`i0`)|输出索引 (qblock 索引`ib`, qblock中qs索引`iqs`)|存储位置 (`y[ib].qs`)|存储内容 (`char4`)|`ds4`存储位置索引`ds4_id`|`ds4` 存储|
|---|---|---|---|---|---|---|
|0|0|`ib=0`,`iqs=0`|`y[0].qs[0:3]`|`q(x[0:3])`|0| `ds4[0]` (d,sum)|
|1|4|`ib=0`,`iqs=4`|`y[0].qs[4:7]`|`q(x[4:7])`|-|-|
|2|8|`ib=0`,`iqs=8`|`y[0].qs[8:11]`|`q(x[8:11])`|-|-|
|3|12|`ib=0`,`iqs=12`|`y[0].qs[12:15]`|`q(x[12:15])`|-|-|
|4|16|`ib=0`,`iqs=16`|`y[0].qs[16:19]`|`q(x[16:19])`|-|-|
|5|20|`ib=0`,`iqs=20`|`y[0].qs[20:23]`|`q(x[20:23])`|-|-|
|6|24|`ib=0`,`iqs=24`|`y[0].qs[24:27]`|`q(x[24:27])`|-|-|
|7|28|`ib=0`,`iqs=28`|`y[0].qs[28:31]`|`q(x[28:31])`|-|-|
|8|32|`ib=0`,`iqs=32`|`y[0].qs[32:35]`|`q(x[32:35])`|1|`ds4[1]` (d,sum)|
|9|36|`ib=0`,`iqs=36`|`y[0].qs[36:39]`|`q(x[36:39])`|-|-|
|10|40|`ib=0`,`iqs=40`|`y[0].qs[40:43]`|`q(x[40:43])`|-|-|
|11|44|`ib=0`,`iqs=44`|`y[0].qs[44:47]`|`q(x[44:47])`|-|-|
|12|48|`ib=0`,`iqs=48`|`y[0].qs[48:51]`|`q(x[48:51])`|-|-|
|13|52|`ib=0`,`iqs=52`|`y[0].qs[52:55]`|`q(x[52:55])`|-|-|
|14|56|`ib=0`,`iqs=56`|`y[0].qs[56:59]`|`q(x[56:59])`|-|-|
|15|60|`ib=0`,`iqs=60`|`y[0].qs[60:63]`|`q(x[60:63])`|-|-|
|16|64|`ib=0`,`iqs=64`|`y[0].qs[64:67]`|`q(x[64:67])`| 2|`ds4[2]` (d,sum)|
|17|68|`ib=0`,`iqs=68`|`y[0].qs[68:71]`|`q(x[68:71])`|-|-|
|18|72|`ib=0`,`iqs=72`|`y[0].qs[72:75]`|`q(x[72:75])`|-|-|
|19|76|`ib=0`,`iqs=76`|`y[0].qs[76:79]`|`q(x[76:79])`|-|-|
|20|80|`ib=0`,`iqs=80`|`y[0].qs[80:83]`|`q(x[80:83])`|-|-|
|21|84|`ib=0`,`iqs=84`|`y[0].qs[84:87]`|`q(x[84:87])`|-|-|
|22|88|`ib=0`,`iqs=88`|`y[0].qs[88:91]`|`q(x[88:91])`|-|-|
|23|92|`ib=0`,`iqs=92`|`y[0].qs[92:95]`|`q(x[92:95])`|-|-|
|24|96|`ib=0`,`iqs=96`|`y[0].qs[96:99]`|`q(x[96:99])`|3|`ds4[3]` (d,sum)|
|25|100|`ib=0`,`iqs=100`|`y[0].qs[100:103]`|`q(x[100:103])`|-|-|
|26|104|`ib=0`,`iqs=104`|`y[0].qs[104:107]`|`q(x[104:107])`|-|-|
|27|108|`ib=0`,`iqs=108`|`y[0].qs[108:111]`|`q(x[108:111])`|-|-|
|28|112|`ib=0`,`iqs=112`|`y[0].qs[112:115]`|`q(x[112:115])`|-|-|
|29|116|`ib=0`,`iqs=116`|`y[0].qs[116:119]`|`q(x[116:119])`|-|-|
|30|120|`ib=0`,`iqs=120`|`y[0].qs[120:123]`|`q(x[120:123])`|-|-|
|31|124|`ib=0`,`iqs=124`|`y[0].qs[124:127]`|`q(x[124:127])`|-|-|
|32|128|`ib=1`,`iqs=0`|`y[1].qs[0:3]`|`q(x[0:3])`|0| `ds4[0]` (d,sum)|

*/

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
    const dim3 threads(block_size, 1, 1);                    // (128,1,1)

#if defined(MY_OPS_DEBUG)
    std::printf(
        "Kernel launch config: block=(%u,%u,%u), grid=(%u,%u,%u)\n",
        threads.x, threads.y, threads.z,
        blocks.x, blocks.y, blocks.z
        );
    std::fflush(stdout);
#endif
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
