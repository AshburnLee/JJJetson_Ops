#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstring>       // memcpy
#include <stdio.h>
#include <vector>
#include <stdexcept>   // std::runtime_error
#include <cassert>
#include "utils.h"
#include "cuda_utils.cuh"

# define CUDA_CPY_BLOCK_SIZE 64

template<typename dst_t, typename src_t>
__host__ __device__ inline dst_t cuda_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
    // 如果 dst_t 是 bf16，先将 src_t 转为 float，仅为转为 bf16
    } else if constexpr(std::is_same_v<dst_t, nv_bfloat16>) {
        return __float2bfloat16(float(x));
    // 如果 src_t 是 bf16，先将bf16 转为 float，再隐式转换
    } else if constexpr(std::is_same_v<src_t, nv_bfloat16>) {
        return __bfloat162float(x);
    } else if constexpr(std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);
    } else {
        return float(x);
    }
}

template<typename src_t, typename dst_t>
static __device__ void cuda_cast_t(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = cuda_cast<dst_t>(*(const src_t *) cxi);
}

/*
元素个数 26624 个
src： ne0_0=128 ne0_1=16 ne0_2=13 ne0_3=1
      nb0_0=4 nb0_1=6656 nb0_2=512 nb0_3=106496

dst： ne1_0=2048 ne1_1=13 ne1_2=1 ne1_3=1
      nb1_0=4 nb1_1=8192 nb1_2=106496 nb1_3=106496

即：
src:  ne[128, 16, 13, 1]    nb[4, 6656, 512,    106496]
dst:  ne[2048, 13, 1, 1]    nb[4, 8192, 106496, 106496]

src 如果连续存储，那么：
- nb0_0 = 4
- nb0_1 = nb0_0 × ne0_0 = 4 × 128 = 512
- nb0_2 = nb0_1 × ne0_1 = 512 × 16 = 8192
- nb0_3 = nb0_2 × ne0_2 = 8192 × 13 = 106496

但实际上：
- nb0_0 = 4
- nb0_1 = 6656      6656/4=1664
- nb0_2 = 512       512/4=128
- nb0_3 = 106496

所以，可能存在 Permute 和 Padding，导致 nb 变化。但 shape 不会变化，是因为实际数值还是那些，
只是 padding 了许多没有的数据。128×16×13 = 26624 个元素的值其实全都在，一个都不少，
只是被故意“以某种方式” 摆在了一块更大的内存里，中间插满了 6144 字节的 padding。

不连续的存储直接使用公式：`offset = i10*nb1_0 + i11*nb1_1 + i12*nb1_2 + i13 * nb1_3;` 即可，

因为无论有没有 padding，无论张量是不是 contiguous，无论有没有被 permute/view 过，
只要你知道当前的 ne[] 和 nb[]，这个公式永远能给你算出逻辑坐标 (i03, i02, i01, i00) 
对应的元素的真实字节偏移。

**padding 已经被 nb 吃进去了, 完全不用手动处理**，只会便利有效数，因为 thread 只有 
ne0_0 * ne0_1 * ne0_2 * ne0_3 个，故不会遍历 padding。

但必须先用 `char*` 完成字节级偏移，再强转成目标类型。
*/
typedef void (*cpy_kernel_t)(const char * src, char * dst);
template <cpy_kernel_t cpy_1>
static __global__ void cpy_continue_kernel(const char * src, char * dst, const int ne,
                                           const int ne0_0, const int ne0_1, const int ne0_2, const int ne0_3,
                                           const int nb0_0, const int nb0_1, const int nb0_2, const int nb0_3, 
                                           const int ne1_0, const int ne1_1, const int ne1_2, const int ne1_3,
                                           const int nb1_0, const int nb1_1, const int nb1_2, const int nb1_3
) {
    // 这里不能使用 reinterpret_cast 因为没有表示类型的模版参数，要在其调用者处
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    /// TODO: 我的公式有问题，why
    // const int64_t i03 = (i / ne0_0*ne0_1*ne0_2) % ne0_3;
    // const int64_t i02 = (i / ne0_0*ne0_1) % ne0_2;
    // const int64_t i01 = (i / ne0_0) % ne0_1;
    // const int64_t i00 = (i / 1) % ne0_0;   // fastest change
    // const int64_t src_offset = i00*nb0_0 + i01*nb0_1 + i02*nb0_2 + i03 * nb0_3;

    // const int64_t i13 = (i / ne1_0*ne1_1*ne1_2) % ne1_3;
    // const int64_t i12 = (i / ne1_0*ne1_1) % ne1_2;
    // const int64_t i11 = (i / ne1_0) % ne1_1;
    // const int64_t i10 = (i / 1) % ne1_0;   // fastest change
    // const int64_t dst_offset = i10*nb1_0 + i11*nb1_1 + i12*nb1_2 + i13 * nb1_3;

    const int64_t i03 = i/(ne0_0 * ne0_1 * ne0_2);
    const int64_t i02 = (i - i03 * ne0_0 * ne0_1 * ne0_2 ) / (ne0_0 * ne0_1);
    const int64_t i01 = (i - i03 * ne0_0 * ne0_1 * ne0_2  -  i02 * ne0_1 * ne0_0) / ne0_0;
    const int64_t i00 = i - i03 * ne0_0 * ne0_1 * ne0_2 - i02 * ne0_1 * ne0_0 - i01 * ne0_0;
    // 始终对应的有效元素在内存中的真实位置
    const int64_t src_offset = i00 * nb0_0 + i01 * nb0_1 + i02 * nb0_2 + i03 * nb0_3;

    const int64_t i13 = i/(ne1_0 * ne1_1 * ne1_2);
    const int64_t i12 = (i - i13 * ne1_0 * ne1_1 * ne1_2) / (ne1_0 * ne1_1);
    const int64_t i11 = (i - i13 * ne1_0 * ne1_1 * ne1_2 - i12 * ne1_0 * ne1_1) / ne1_0;
    const int64_t i10 = i - i13 * ne1_0 * ne1_1 * ne1_2 - i12 * ne1_0 * ne1_1 - i11 * ne1_0;
    // dst 的nb值表示 dst元素是连续的，无padding
    const int64_t dst_offset = i10 * nb1_0 + i11 * nb1_1 + i12 * nb1_2 + i13 * nb1_3;
    // src + src_offset 准确指向 src tensor 中逻辑索引为 (i03,i02,i01,i00) 的有效值的内存位置
    cpy_1(src + src_offset, dst + dst_offset);
}

template<typename src_t, typename dst_t>
static void cpy_conitune_impl(
    const char * csrc, char * cdst,
    const std::vector<int>& src_dims,
    const std::vector<int>& dst_dims,
    const std::vector<int>& src_stride,
    const std::vector<int>& dst_stride
) {
    // 
    const src_t* src = reinterpret_cast<const src_t*>(csrc);
    dst_t* dst = reinterpret_cast<dst_t*>(cdst);

    assert(src_dims.size() <= 4 && "src_dims must be <= 4");
    assert(dst_dims.size() <= 4 && "dst_dims must be <= 4");
    // array is row-major in python, kernel is col-major 
    const int ne0_0 = src_dims[0];
    const int ne0_1 = src_dims[1];
    const int ne0_2 = src_dims[2];
    const int ne0_3 = src_dims[3];

    const int ne1_0 = dst_dims[0];
    const int ne1_1 = dst_dims[1];
    const int ne1_2 = dst_dims[2];
    const int ne1_3 = dst_dims[3];

    const int nb0_0 = src_stride[0];
    const int nb0_1 = src_stride[1];
    const int nb0_2 = src_stride[2];
    const int nb0_3 = src_stride[3];

    const int nb1_0 = dst_stride[0];
    const int nb1_1 = dst_stride[1];
    const int nb1_2 = dst_stride[2];
    const int nb1_3 = dst_stride[3];

    const int n_elem = ne0_0 * ne0_1 * ne0_2 * ne0_3;

    src_t* d_src = nullptr; 
    dst_t* d_dst = nullptr; 

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMallocAsync(&d_src, n_elem * sizeof(src_t),stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, n_elem * sizeof(dst_t),stream));
    CUDA_CHECK(cudaMemcpyAsync(d_src, src, n_elem * sizeof(src_t), cudaMemcpyHostToDevice,stream));

    const int num_blocks = (n_elem + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    // cpy_continue_kernel 函数没有表示类型的模版参数，我出纳入的 src_t* 的 d_src 和 dst_t* 的d_dst，肯定报错
    // 下面的 src_t, dst_t 仅仅是 cuda_cast_t 的模版参数。
    // 所以只能传入 cpy_continue_kernel const char*的d_src 和 char* 的 d_dst！！ 
#if defined(MY_OPS_DEBUG)
    std::printf(
        "Kernel launch config: block=(%lld,1,1), grid=(%lld,1,1)\n",
        static_cast<long long>(CUDA_CPY_BLOCK_SIZE),
        static_cast<long long>(num_blocks));
    std::fflush(stdout);
#endif
    cpy_continue_kernel<cuda_cast_t<src_t, dst_t>><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (reinterpret_cast<const char*>(d_src), 
         reinterpret_cast<char*>(d_dst), n_elem,
         ne0_0, ne0_1, ne0_2, ne0_3, nb0_0, nb0_1, nb0_2, nb0_3,
         ne1_0, ne1_1, ne1_2, ne1_3, nb1_0, nb1_1, nb1_2, nb1_3);
    LAUNCH_CHECK();
    
    CUDA_CHECK(cudaMemcpyAsync(dst, d_dst, n_elem * sizeof(dst_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeAsync(d_src, nullptr));
    CUDA_CHECK(cudaFreeAsync(d_dst, nullptr));
    
}

extern "C" void cpy_continue(
    const char * src, char * dst,
    const std::vector<int>& src_dims,
    const std::vector<int>& dst_dims,
    const std::vector<int>& src_stride,
    const std::vector<int>& dst_stride,
    data_type src_dt, data_type dst_dt 
) {
    if (src_dt == data_type::DT_F32 && dst_dt == data_type::DT_F32) {
        cpy_conitune_impl<float, float>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    // } else if (src_dt == data_type::DT_BF16 && dst_dt == data_type::DT_BF16) {
    //     cpy_conitune_impl<nv_bfloat16, nv_bfloat16>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    // } else if (src_dt == data_type::DT_F32 && dst_dt == data_type::DT_BF16) {
    //     cpy_conitune_impl<float, nv_bfloat16>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    // } else if (src_dt == data_type::DT_I32 && dst_dt == data_type::DT_BF16) {
    //     cpy_conitune_impl<int32_t, nv_bfloat16>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    // } else if (src_dt == data_type::DT_F16 && dst_dt == data_type::DT_BF16) {
    //     cpy_conitune_impl<half, nv_bfloat16>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    // } else if (src_dt == data_type::DT_BF16 && dst_dt == data_type::DT_F32) {
    //     cpy_conitune_impl<nv_bfloat16, float>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    // } else if(src_dt == data_type::DT_BF16 && dst_dt == data_type::DT_F16) {
    //     cpy_conitune_impl<nv_bfloat16, half>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    // }else if (src_dt == data_type::DT_BF16 && dst_dt == data_type::DT_I32) {
    //     cpy_conitune_impl<nv_bfloat16, int32_t>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    } else if (src_dt == data_type::DT_F32 && dst_dt == data_type::DT_F16) {
        cpy_conitune_impl<float, half>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    } else if (src_dt == data_type::DT_F16 && dst_dt == data_type::DT_F32) {
        cpy_conitune_impl<half, float>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    } else if (src_dt == data_type::DT_I32 && dst_dt == data_type::DT_F32) {
        cpy_conitune_impl<int32_t, float>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    } else if (src_dt == data_type::DT_F32 && dst_dt == data_type::DT_I32) {
        cpy_conitune_impl<float, int32_t>(src, dst, src_dims,dst_dims,src_stride,dst_stride);
    } else {
        throw std::runtime_error("Cast not supported!");
    }
    
}
