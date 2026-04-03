#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "cuda_utils.cuh"

static __device__ __forceinline__ float op_relu(float x) {
    return fmaxf(x, 0);
}
static __device__ __forceinline__ float op_gelu(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}
static __device__ __forceinline__ float op_silu(float x) {
    return x / (1.0f + expf(-x));
}
static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;
    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}
static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;

    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

#define CUDA_GLU_BLOCK_SIZE 256

/*
出发 case:
src0_p  (6144,13,1,1)   只有第一个维度可以有padding
src1_p  (6144,13,1,1)   只有第一个维度可以有padding
nc = n = ne0 = 6144         dst->ne[0] == nc
k = dst.num_elem = 79872    输出元素个数
o0 = 6144              src_0->nb[1]/sizeof(T)
o1 = 6144              src_1->nb[1]/sizeof(T) = 6144*4/4=6144（这是连续的情况下），即第二个维度元素的 stride
num_block = 312
num_thread = 256
*/
template <float (*op)(float), typename T>
static __global__ void unary_gated_op_kernel(const T * src, const T * g, T * dst, 
                                             const int64_t k, 
                                             const int64_t ne0, 
                                             const int64_t o0, 
                                             const int64_t o1) {
    const int64_t i = threadIdx.x + blockDim.x * blockIdx.x;
    // k 是不含 padding 的元素个数
    // 为什么会遍历所有真实值，并且不会将 padding 写入dst？
    // 答：因为有 o0 存在，得到的 j0 不是连续的：j = (0,1,2,3,8,9,10,11,16,17,18,19)，中间跳过的就是padding值
    // 故 dst 中只有真实值
    if (i >= k) {
        return;
    }
    // (i / ne0) 是行号，乘以 o0 就是跳过前整行（含 padding）
    const int64_t j0 = (i / ne0) * o0 + (i % ne0);
    const int64_t j1 = o0 == o1 ? j0 : (i / ne0) * o1 + (i % ne0);

    dst[i] = (T)(op((float)src[j0]) * (float)g[j1]);
}


template <float (*op)(float), typename T>
static void unary_gated(const T * src0, const T * src1, T * dst, 
                             const std::vector<int>& dst_dims,
                             const std::vector<int>& src0_nb, 
                             const std::vector<int>& src1_nb
) {
    // src0 src1 dst 的 shape 相同，dst 是连续的，不含 padding 的
    // src0/src1 在“可 padding 维度”上物理长度可能更大（o0/o1），需按物理大小分配和拷贝
    int64_t dst_elem = dst_dims[0] * dst_dims[1] * dst_dims[2] * dst_dims[3];
    int64_t dst_ne0 = dst_dims[0];
    int64_t o0=src0_nb[1]/sizeof(T);  // 这个维度 包括padding的元素数stride 即该维度可以含有padding
    int64_t o1=src1_nb[1]/sizeof(T);  // 这个维度 包括padding的元素数stride 即该维度可以含有padding
    // 注意这里要通过 物理布局，即含有padding的布局分配空间
    int64_t src0_elem = o0 * dst_dims[1] * dst_dims[2] * dst_dims[3];
    int64_t src1_elem = o1 * dst_dims[1] * dst_dims[2] * dst_dims[3];

    T* d_src0 = nullptr;
    T* d_src1 = nullptr;
    T* d_dst  = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // src0 src1 要分配含有padding的空间
    CUDA_CHECK(cudaMallocAsync(&d_src0, src0_elem * sizeof(T),stream));
    CUDA_CHECK(cudaMallocAsync(&d_src1, src1_elem * sizeof(T),stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, dst_elem * sizeof(T),stream));

    CUDA_CHECK(cudaMemcpyAsync(d_src0, src0, src0_elem * sizeof(T), cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(d_src1, src1, src1_elem * sizeof(T), cudaMemcpyHostToDevice,stream));

    // num_block = 312
    const int64_t blocks = (dst_elem + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const int64_t threads = CUDA_GLU_BLOCK_SIZE;
#if defined(MY_OPS_DEBUG)
    std::printf(
        "Kernel launch config: block=(%lld, 1, 1), grid=(%lld, 1, 1)\n",
        static_cast<long long>(threads), static_cast<long long>(blocks));
    std::fflush(stdout);
#endif
    unary_gated_op_kernel<op><<<blocks, threads, 0, stream>>>(
        d_src0, d_src1, d_dst, dst_elem, dst_ne0, o0, o1);
    LAUNCH_CHECK();
    
    CUDA_CHECK(cudaMemcpyAsync(dst, d_dst, dst_elem * sizeof(T), cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeAsync(d_src0, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_src1, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_dst, nullptr));
    
}


extern "C" void relu_gated(const float* src0, const float* src1, float* dst, 
                             const std::vector<int>& dst_dims,
                             const std::vector<int>& src0_nb, 
                             const std::vector<int>& src1_nb) {
    unary_gated<op_relu, float>(src0, src1, dst, dst_dims, src0_nb, src1_nb);
}

extern "C" void gelu_gated(const float* src0, const float* src1, float* dst, 
                             const std::vector<int>& dst_dims,
                             const std::vector<int>& src0_nb, 
                             const std::vector<int>& src1_nb) {
    unary_gated<op_gelu, float>(src0, src1, dst, dst_dims, src0_nb, src1_nb);
}

extern "C" void silu_gated(const float* src0, const float* src1, float* dst, 
                             const std::vector<int>& dst_dims,
                             const std::vector<int>& src0_nb, 
                             const std::vector<int>& src1_nb) {
    unary_gated<op_silu, float>(src0, src1, dst, dst_dims, src0_nb, src1_nb);
}

extern "C" void gelu_erf_gated(const float* src0, const float* src1, float* dst, 
                             const std::vector<int>& dst_dims,
                             const std::vector<int>& src0_nb, 
                             const std::vector<int>& src1_nb) {
    unary_gated<op_gelu_erf, float>(src0, src1, dst, dst_dims, src0_nb, src1_nb);
}

extern "C" void gelu_quick_gated(const float* src0, const float* src1, float* dst, 
                             const std::vector<int>& dst_dims,
                             const std::vector<int>& src0_nb, 
                             const std::vector<int>& src1_nb) {
    unary_gated<op_gelu_quick, float>(src0, src1, dst, dst_dims, src0_nb, src1_nb);
}
