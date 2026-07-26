#include "elementwise.h"

#include <cuda_runtime.h>
#include <cstdio>

#include "cuda_utils.cuh"

#define CUDA_ELEMENTWISE_BLOCK_SIZE 256

struct ElementwiseAddOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};

struct ElementwiseSubOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a - b; }
};

struct ElementwiseMulOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a * b; }
};

struct ElementwiseDivOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a / b; }
};

template <typename Op>
static __global__ void elementwise_binary_kernel(const float *__restrict__ a,
                                                 const float *__restrict__ b,
                                                 float *__restrict__ out, int n_elem) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elem) {
        Op op{};
        out[i] = op(a[i], b[i]);
    }
}

static int elementwise_check_n_elem(int n_elem) {
    if (n_elem <= 0) {
        std::fprintf(stderr, "elementwise_binary_forward_device: invalid n_elem=%d\n", n_elem);
        return -1;
    }
    return 0;
}

template <typename Op>
static void elementwise_binary_launch(cudaStream_t stream, const float *d_a, const float *d_b,
                                      float *d_out, int n_elem) {
    const int threads = CUDA_ELEMENTWISE_BLOCK_SIZE;
    const int blocks = (n_elem + threads - 1) / threads;

#if defined(MY_OPS_DEBUG)
    std::printf("elementwise_binary launch: blocks=%d threads=%d n_elem=%d\n", blocks, threads,
                n_elem);
    std::fflush(stdout);
#endif

    elementwise_binary_kernel<Op><<<blocks, threads, 0, stream>>>(d_a, d_b, d_out, n_elem);
    LAUNCH_CHECK();
}

static int elementwise_binary_launch_op(void *stream, ElementwiseBinaryOp op, const float *d_a,
                                        const float *d_b, float *d_out, int n_elem) {
    if (elementwise_check_n_elem(n_elem) != 0) {
        return -1;
    }
    if (d_a == nullptr || d_b == nullptr || d_out == nullptr) {
        std::fprintf(stderr, "elementwise_binary_forward_device: null pointer argument\n");
        return -1;
    }

    cudaStream_t s = stream != nullptr ? static_cast<cudaStream_t>(stream) : nullptr;
    if (s == nullptr) {
        std::fprintf(stderr, "elementwise_binary_forward_device: stream is null\n");
        return -1;
    }

    switch (op) {
    case ELEMENTWISE_ADD:
        elementwise_binary_launch<ElementwiseAddOp>(s, d_a, d_b, d_out, n_elem);
        break;
    case ELEMENTWISE_SUB:
        elementwise_binary_launch<ElementwiseSubOp>(s, d_a, d_b, d_out, n_elem);
        break;
    case ELEMENTWISE_MUL:
        elementwise_binary_launch<ElementwiseMulOp>(s, d_a, d_b, d_out, n_elem);
        break;
    case ELEMENTWISE_DIV:
        elementwise_binary_launch<ElementwiseDivOp>(s, d_a, d_b, d_out, n_elem);
        break;
    default:
        std::fprintf(stderr, "elementwise_binary_forward_device: unknown op=%d\n",
                     static_cast<int>(op));
        return -1;
    }
    return 0;
}

extern "C" int elementwise_binary_forward_device(void *stream, ElementwiseBinaryOp op,
                                                 const float *d_a, const float *d_b, float *d_out,
                                                 int n_elem) {
    return elementwise_binary_launch_op(stream, op, d_a, d_b, d_out, n_elem);
}

extern "C" int elementwise_add_forward_device(void *stream, const float *d_a, const float *d_b,
                                              float *d_out, int n_elem) {
    return elementwise_binary_forward_device(stream, ELEMENTWISE_ADD, d_a, d_b, d_out, n_elem);
}

extern "C" int elementwise_sub_forward_device(void *stream, const float *d_a, const float *d_b,
                                              float *d_out, int n_elem) {
    return elementwise_binary_forward_device(stream, ELEMENTWISE_SUB, d_a, d_b, d_out, n_elem);
}

extern "C" int elementwise_mul_forward_device(void *stream, const float *d_a, const float *d_b,
                                              float *d_out, int n_elem) {
    return elementwise_binary_forward_device(stream, ELEMENTWISE_MUL, d_a, d_b, d_out, n_elem);
}

extern "C" int elementwise_div_forward_device(void *stream, const float *d_a, const float *d_b,
                                              float *d_out, int n_elem) {
    return elementwise_binary_forward_device(stream, ELEMENTWISE_DIV, d_a, d_b, d_out, n_elem);
}

// ======================== 仅供 Python 测试 ================================
extern "C" void elementwise_binary_forward_host(const float *a_host, const float *b_host,
                                                float *out_host, int n_elem,
                                                ElementwiseBinaryOp op) {
    if (elementwise_check_n_elem(n_elem) != 0) {
        return;
    }
    if (a_host == nullptr || b_host == nullptr || out_host == nullptr) {
        std::fprintf(stderr, "elementwise_binary_forward_host: null pointer argument\n");
        return;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const size_t bytes = static_cast<size_t>(n_elem) * sizeof(float);
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_a, bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_b, bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_out, bytes, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_a, a_host, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b_host, bytes, cudaMemcpyHostToDevice, stream));

    if (elementwise_binary_forward_device(stream, op, d_a, d_b, d_out, n_elem) != 0) {
        CUDA_CHECK(cudaFreeAsync(d_a, stream));
        CUDA_CHECK(cudaFreeAsync(d_b, stream));
        CUDA_CHECK(cudaFreeAsync(d_out, stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(out_host, d_out, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_a, stream));
    CUDA_CHECK(cudaFreeAsync(d_b, stream));
    CUDA_CHECK(cudaFreeAsync(d_out, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
