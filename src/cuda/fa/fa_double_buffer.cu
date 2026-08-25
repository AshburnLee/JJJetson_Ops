// ★ 生产 FA kernel（engine 唯一选用）
// 使用双缓冲：
// 当前 tile 的 V cp.async 与 QK WMMA 异步，隐藏 load V
// 下一 tile 的 K prefetch 与 softmax 异步，隐藏 load K
// 两次 wait 后做 WMMA PV

// 泛化 shape：head_dim 模板实例 {32,64,128}；seq / head 数运行时传入

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#include "cuda_fp16.h"
#include "cuda_utils.cuh"
#include "fa.h"

#include "fa_double_buffer_kernel.cuh"

namespace {

bool fa_is_supported_head_dim(int head_dim) {
    return head_dim == 32 || head_dim == 64 || head_dim == 128;
}

fa_db::FaDoubleBufferKernelParams fa_make_kernel_params(const FaDoubleBufferShape *shape,
                                                        float scale) {
    fa_db::FaDoubleBufferKernelParams kparams{};
    kparams.num_q_tokens = shape->num_q_tokens;
    kparams.num_kv_tokens = shape->num_kv_tokens;
    kparams.num_q_heads = shape->num_q_heads;
    kparams.num_kv_heads = shape->num_kv_heads;
    kparams.scale = scale;
    return kparams;
}

void fa_double_buffer_launch(cudaStream_t stream, const FaDoubleBufferShape *shape, const half *d_q,
                             const half *d_k, const half *d_v, float *d_dst, float scale) {
    const fa_db::FaDoubleBufferKernelParams kparams = fa_make_kernel_params(shape, scale);
    switch (shape->head_dim) {
    case 32:
        fa_db::fa_double_buffer_launch_templated<32>(stream, d_q, d_k, d_v, d_dst, kparams);
        break;
    case 64:
        fa_db::fa_double_buffer_launch_templated<64>(stream, d_q, d_k, d_v, d_dst, kparams);
        break;
    case 128:
        fa_db::fa_double_buffer_launch_templated<128>(stream, d_q, d_k, d_v, d_dst, kparams);
        break;
    default:
        break;
    }
}

} // namespace

extern "C" int fa_double_buffer_validate_shape(const FaDoubleBufferShape *shape) {
    if (shape == nullptr) {
        std::fprintf(stderr, "fa_double_buffer_validate_shape: shape is null\n");
        return -1;
    }
    if (!fa_is_supported_head_dim(shape->head_dim)) {
        std::fprintf(stderr,
                     "fa_double_buffer_validate_shape: unsupported head_dim=%d (use 32/64/128)\n",
                     shape->head_dim);
        return -1;
    }
    if (shape->head_dim % fa_db::kWmmaN != 0) {
        std::fprintf(stderr, "fa_double_buffer_validate_shape: head_dim must be multiple of 16\n");
        return -1;
    }
    if (shape->num_q_tokens <= 0 || shape->num_q_tokens > fa_db::kMaxQTokensPerBlock) {
        std::fprintf(stderr,
                     "fa_double_buffer_validate_shape: num_q_tokens=%d out of range [1,%d]\n",
                     shape->num_q_tokens, fa_db::kMaxQTokensPerBlock);
        return -1;
    }
    if (shape->num_kv_tokens <= 0) {
        std::fprintf(stderr, "fa_double_buffer_validate_shape: num_kv_tokens=%d invalid\n",
                     shape->num_kv_tokens);
        return -1;
    }
    if (shape->num_kv_heads <= 0 || shape->num_q_heads <= 0) {
        std::fprintf(stderr, "fa_double_buffer_validate_shape: invalid head counts q=%d kv=%d\n",
                     shape->num_q_heads, shape->num_kv_heads);
        return -1;
    }
    if (shape->num_q_heads % shape->num_kv_heads != 0) {
        std::fprintf(stderr,
                     "fa_double_buffer_validate_shape: num_q_heads must be multiple of "
                     "num_kv_heads (got q=%d kv=%d)\n",
                     shape->num_q_heads, shape->num_kv_heads);
        return -1;
    }
    // 每 block 固定 2 个 Q，分组 g=q/kv 须为偶数。g=2 即原来的 2:1。
    const int gqa_g = shape->num_q_heads / shape->num_kv_heads;
    if (gqa_g < 2 || (gqa_g % 2) != 0) {
        std::fprintf(stderr,
                     "fa_double_buffer_validate_shape: GQA group g=q/kv must be even and >=2 "
                     "(got q=%d kv=%d g=%d)\n",
                     shape->num_q_heads, shape->num_kv_heads, gqa_g);
        return -1;
    }
    return 0;
}

// -========================-- 生产 (device) --========================-
extern "C" int fa_double_buffer_forward_device(void *stream, const FaDoubleBufferShape *shape,
                                               const uint16_t *d_q, const uint16_t *d_k,
                                               const uint16_t *d_v, float *d_dst, float scale) {
    if (stream == nullptr) {
        std::fprintf(stderr, "fa_double_buffer_forward_device: stream is null\n");
        return -1;
    }
    if (d_q == nullptr || d_k == nullptr || d_v == nullptr || d_dst == nullptr) {
        std::fprintf(stderr, "fa_double_buffer_forward_device: d_q/d_k/d_v/d_dst is null\n");
        return -1;
    }
    if (fa_double_buffer_validate_shape(shape) != 0) {
        return -1;
    }

    fa_double_buffer_launch(
        static_cast<cudaStream_t>(stream), shape, reinterpret_cast<const half *>(d_q),
        reinterpret_cast<const half *>(d_k), reinterpret_cast<const half *>(d_v), d_dst, scale);
    return 0;
}

static FaDoubleBufferShape fa_legacy_test_shape() {
    FaDoubleBufferShape shape{};
    shape.head_dim = 128;
    shape.num_q_tokens = 13;
    shape.num_kv_tokens = 256;
    shape.num_q_heads = 16;
    shape.num_kv_heads = 8;
    return shape;
}

// ======================== 仅供 Python 测试 ================================
extern "C" void fa_double_buffer_forward_host_legacy(const uint16_t *q_host, const uint16_t *k_host,
                                                     const uint16_t *v_host, float *dst_host,
                                                     float scale) {
    const FaDoubleBufferShape shape = fa_legacy_test_shape();
    using half_t = half;

    const size_t q_elems =
        static_cast<size_t>(shape.head_dim) * shape.num_q_tokens * shape.num_q_heads;
    const size_t kv_elems =
        static_cast<size_t>(shape.head_dim) * shape.num_kv_tokens * shape.num_kv_heads;
    const size_t dst_elems = q_elems;

    half_t *d_q = nullptr;
    half_t *d_k = nullptr;
    half_t *d_v = nullptr;
    float *d_dst = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_q, q_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_k, kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_v, kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, dst_elems * sizeof(float), stream));

    CUDA_CHECK(
        cudaMemcpyAsync(d_q, q_host, q_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_k, k_host, kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_v, v_host, kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));

    if (fa_double_buffer_forward_device(stream, &shape, reinterpret_cast<const uint16_t *>(d_q),
                                        reinterpret_cast<const uint16_t *>(d_k),
                                        reinterpret_cast<const uint16_t *>(d_v), d_dst,
                                        scale) != 0) {
        CUDA_CHECK(cudaFreeAsync(d_q, stream));
        CUDA_CHECK(cudaFreeAsync(d_k, stream));
        CUDA_CHECK(cudaFreeAsync(d_v, stream));
        CUDA_CHECK(cudaFreeAsync(d_dst, stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, dst_elems * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_q, stream));
    CUDA_CHECK(cudaFreeAsync(d_k, stream));
    CUDA_CHECK(cudaFreeAsync(d_v, stream));
    CUDA_CHECK(cudaFreeAsync(d_dst, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

extern "C" void fa(const uint16_t *q_host, const uint16_t *k_host, const uint16_t *v_host,
                   float *dst_host, float scale) {
    fa_double_buffer_forward_host_legacy(q_host, k_host, v_host, dst_host, scale);
}

extern "C" void fa_double_buffer_forward_host(const FaDoubleBufferShape *shape,
                                              const uint16_t *q_host, const uint16_t *k_host,
                                              const uint16_t *v_host, float *dst_host,
                                              float scale) {
    if (shape == nullptr || q_host == nullptr || k_host == nullptr || v_host == nullptr ||
        dst_host == nullptr) {
        std::fprintf(stderr, "fa_double_buffer_forward_host: null pointer argument\n");
        return;
    }
    if (fa_double_buffer_validate_shape(shape) != 0) {
        return;
    }

    using half_t = half;
    const size_t q_elems =
        static_cast<size_t>(shape->head_dim) * shape->num_q_tokens * shape->num_q_heads;
    const size_t kv_elems =
        static_cast<size_t>(shape->head_dim) * shape->num_kv_tokens * shape->num_kv_heads;
    const size_t dst_elems = q_elems;

    half_t *d_q = nullptr;
    half_t *d_k = nullptr;
    half_t *d_v = nullptr;
    float *d_dst = nullptr;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_q, q_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_k, kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_v, kv_elems * sizeof(half_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_dst, dst_elems * sizeof(float), stream));

    CUDA_CHECK(
        cudaMemcpyAsync(d_q, q_host, q_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_k, k_host, kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_v, v_host, kv_elems * sizeof(half_t), cudaMemcpyHostToDevice, stream));

    if (fa_double_buffer_forward_device(stream, shape, reinterpret_cast<const uint16_t *>(d_q),
                                        reinterpret_cast<const uint16_t *>(d_k),
                                        reinterpret_cast<const uint16_t *>(d_v), d_dst,
                                        scale) != 0) {
        CUDA_CHECK(cudaFreeAsync(d_q, stream));
        CUDA_CHECK(cudaFreeAsync(d_k, stream));
        CUDA_CHECK(cudaFreeAsync(d_v, stream));
        CUDA_CHECK(cudaFreeAsync(d_dst, stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, dst_elems * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_q, stream));
    CUDA_CHECK(cudaFreeAsync(d_k, stream));
    CUDA_CHECK(cudaFreeAsync(d_v, stream));
    CUDA_CHECK(cudaFreeAsync(d_dst, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
