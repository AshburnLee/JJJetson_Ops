#include <cuda_runtime.h>
#include <cstring>       // memcpy
#include <stdio.h>
#include <vector>
#include "cuda_utils.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256 

struct rope_corr_dims {
    float v[2];
};

template<bool forward>
static __global__ void rope_neox_kernel(
        const float * x,                     // 对 x 进行 RoPE 操作
        float * dst,                         // 将结果写入 dst
        const int ne0,                   // 128，变化最快（第一个维度）的维度大小
        const int ne1,                   // 16，变化其次块（第二个维度）的维度大小
        const int s1,                    // stride Of dim1：128
        const int s2,                    // stride of dim2：128*16=2048
        const int n_dims,                // 128 每个 head 的前 n_dims 维参与旋转，后半部分直接拷贝
        const int *pos,
        const float freq_scale,          // 1，
        const float ext_factor,          // 0，
        const float attn_factor,         // 1，
        const rope_corr_dims corr_dims,  // [24,41],
        const float theta_scale
) {
    // y 方向，即变化最快的实际值索引 0~255
    const int id_fast = threadIdx.y + blockDim.y * blockIdx.y ;

    if (id_fast >= ne0) {
        return;
    }
    // x 方向（将head和token合并为1维度），表示（num_head=16, token=13）索引 0~207
    const int id_flat_ht = threadIdx.x + blockDim.x * blockIdx.x ;
    // 因为是 col-major，故这里除数是 ne1, 即两者变化较快的维度
    const int id_head       = id_flat_ht % ne1;
    const int id_token      = id_flat_ht / ne1;

    // 扁平化后的输出数据索引。与核心公式1一致
    const int id_dest = id_fast + ne0 * id_flat_ht;
    // 扁平化后的输入 x 索引。与核心公式1一致
    const int ix      = id_fast + s1 * id_head + s2 * id_token;

    // 超出 n_dims的位置元素直接copy， 与方向只有64个 thread 在工作
    // [TODO]: id_dest和ix 中 已经有了 id_fast 的信息，这里还要 id_dest + id_fast有问题 ？
    if (id_fast*2 >= n_dims) {
        dst[id_dest + id_fast + 0] = x[ix + id_fast + 0];
        dst[id_dest + id_fast + 1] = x[ix + id_fast + 1];
        return;
    }

    // 不同的 token pos 不同，pos 的长度与 token个数相同
    // theta_base 随id_fast 和 id_token 不同而不同
    const float theta_base = pos[id_token] * powf(theta_scale, (float)id_fast);
    // const float freq_factor = 1.0f;
    float cos_theta;
    float sin_theta;
    // 计算 cos_theta & sin_theta 带 $\theta$
    cos_theta = cosf(theta_base);
    sin_theta = sinf(theta_base);
    // 或 使用YaRN 得到 sing_theta 和 cos_theta
    if (!forward) {
        sin_theta *= -1.0f;
    }
    // Rotation 核心计算
    // 1. 获取旋转对
    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[id_dest + 0]        = x0*cos_theta - x1*sin_theta;
    dst[id_dest + n_dims/2] = x0*sin_theta + x1*cos_theta;
}


/*
输入 x 的维度是 [128,16,13,1]（col-major），

  - 第0维 128 `head_dim`：每个 head 的 embedding 维度。旋转对：(0,64),(1,65),...,(63,127)
  - 第1维 16 `n_heads`：多头数量（每个 head 独立 RoPE）
  - 第2维 13 `seq_len`：当前 prompt 的 token 数量（共 13 个位置）
*/
extern "C" void rope(float* input, int* pos, float* output, std::vector<int>& input_dims) {
    const int64_t ne0_0 = input_dims[0];  // 128
    const int64_t ne0_1 = input_dims[1];  // 16
    const int64_t ne0_2 = input_dims[2];  // 13
    const int64_t ne0_3 = input_dims[3];  // 1

    const int64_t s0_1 = ne0_0;                            // 128
    const int64_t s0_2 = ne0_0 * ne0_1;                     // 128*16 = 2048
    const int64_t n_elem = ne0_0 * ne0_1 * ne0_2 * ne0_3;    // 26624
    const int64_t nr = ne0_1 * ne0_2 * ne0_3;               // 208

    float *d_x = nullptr;
    int *d_pos = nullptr;
    float *d_y = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_x, n_elem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_pos, ne0_2 * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_y, n_elem * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_x, input, n_elem * sizeof(float), cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos, pos, ne0_2 * sizeof(int), cudaMemcpyHostToDevice,stream));

    const dim3 threads(1, CUDA_ROPE_BLOCK_SIZE, 1); // (1,256,1)
    const int n_blocks_x = (ne0_0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 blocks(nr, n_blocks_x, 1);  // (208,1,1)
    
    // 旋转角度
    int n_dims = 128;
    const float freq_base = 10000.0f;
    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    // 在YaRN 中使用的参数：
    const float freq_scale = 1.0f;
    const float ext_factor = 0.f; 
    const rope_corr_dims corr_dims = {24.f, 41.f};
    const float attn_factor = 1.f;

    rope_neox_kernel<true><<<blocks, threads, 0, stream>>>(d_x, d_y, ne0_0, ne0_1, s0_1, s0_2, n_dims, 
        d_pos,
        freq_scale,
        ext_factor,
        attn_factor,
        corr_dims,
        theta_scale
    );
    LAUNCH_CHECK();

    CUDA_CHECK(cudaMemcpyAsync(output, d_y, n_elem * sizeof(float), cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeAsync(d_x, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_pos, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_y, nullptr));
}
