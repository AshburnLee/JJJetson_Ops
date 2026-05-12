#include <cuda_runtime.h>
#include <vector>

#include "cublas_utils.cuh"
#include "cuda_utils.cuh"

extern "C" void moe_dispatch_launch_cuda(cudaStream_t stream, const float *d_x,
                                         const int *d_expert_ids, int num_tokens, int top_k,
                                         int hidden_size, int num_experts, float *d_permuted_x,
                                         int *d_source_token, int *d_source_k,
                                         int *d_expert_offsets, int *d_counts, int *d_next_slot);

extern "C" void moe_combine_launch_cuda(cudaStream_t stream, const float *d_expert_out,
                                        const int *d_source_token, const int *d_source_k,
                                        const float *d_route_weights, float *d_y, int num_routes,
                                        int hidden_size, int top_k);

static __device__ __forceinline__ float silu_f(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

static __global__ void moe_swiglu_silu_mul_kernel(const float *gate, const float *up, float *mid,
                                                  int n_elem) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elem) {
        mid[i] = silu_f(gate[i]) * up[i];
    }
}

static void moe_swiglu_experts_forward_cublas_device(
    cudaStream_t stream, cublasHandle_t handle, const float *d_x, const int *d_expert_ids,
    const float *d_route_weights, const float *d_w_gate, const float *d_w_up, const float *d_w_down,
    float *d_y, int num_tokens, int hidden_size, int intermediate_size, int num_experts, int top_k,
    float *d_perm, int *d_src_t, int *d_src_k, int *d_off, int *d_counts, int *d_next,
    float *d_gate, float *d_up, float *d_mid, float *d_expert_out, int *h_off_host) {
    const int H = hidden_size;
    const int I = intermediate_size;
    const int R = num_tokens * top_k;

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    moe_dispatch_launch_cuda(stream, d_x, d_expert_ids, num_tokens, top_k, H, num_experts, d_perm,
                             d_src_t, d_src_k, d_off, d_counts, d_next);

    CUDA_CHECK(cudaMemcpyAsync(h_off_host, d_off,
                               static_cast<size_t>(num_experts + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const float alpha = 1.f;
    const float beta = 0.f;
    // 首先定位每一个 e 对应的输入、输出的 “片”，然后 call API
    for (int e = 0; e < num_experts; ++e) {
        // 当前 e 的开始
        const int start = h_off_host[e];
        // 当前 e 的偏移
        const int n = h_off_host[e + 1] - start;
        if (n <= 0) {
            continue;
        }
        const float *Xe = d_perm + static_cast<size_t>(start) * H;
        const float *Wg = d_w_gate + static_cast<size_t>(e) * I * H;
        const float *Wu = d_w_up + static_cast<size_t>(e) * I * H;
        float *Ge = d_gate + static_cast<size_t>(start) * I;
        float *Ue = d_up + static_cast<size_t>(start) * I;

        // C = alpha * op(A) * op(B)  + beta*C，
        // OP 表示是否转置
        // A、B、C 约定为col-major
        // op(A) : m x k
        // op(B) : k x n
        // C     : m x n
        // C 物理上的 row-major 3x2 在API看来是 col-major的 2x3 !!! 这是理解 row-major
        // 数据和col-major API的关键 对于 md 中 case e=1 时的参数列表： 2x4, 4x3 = 2x3, 故OP(A)
        // 是Wg^T, OP(B) 是Xe，输出在row-major时 是 3x2 的,所以 是C（Ge） 决定了 A 是权值， B 是x
        // !!!
        CUBLAS_CHECK(cublasSgemm(
            handle,      /* handle：当前 cuBLAS 上下文 */
            CUBLAS_OP_T, /* transa：对 A 做转置 op(A)=Wg^T，原始形状 IxH (2x4) */
            CUBLAS_OP_N, /* transb：B 侧不转置 op(B)=Xe 原始shape nxH (3x4)*/
            I,           /* m=I (2)：OP(A) 和 C 相同的行数 */
            n,           /* n=n (3)：OP(B) 和 C 相同的列数 */
            H,           /* k=H (4)：OP(A) 列数和 OP(B) 行数 */
            &alpha,      /* alpha=1：系数 */
            Wg,          /* A=Wg：权重，指向本 expert 的 WG[e]*/
            H,     /* lda=H（4）：A 的leading dimension长度，与A是否转置无关，实际显存中的位置*/
            Xe,    /* B=Xe：输入批指针，d_perm 中本段起始行，形状与此次 GEMM 对应 nxH（3x4） */
            H,     /* ldb=H（4）：B 的leading dimension长度,每行4个值连续排列 */
            &beta, /* beta：输出 C 系数, 这里表示不保留 C 旧内容 */
            Ge,    /* C=Ge：gate 输出缓冲 Ge，写入与 d_perm 对齐的“片”，形状 Ixn（col-major） */
            I));   /* ldc=I（2）：C 的leading dimension长度，每行2个值连续排列 */

        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, I, n, H, &alpha, Wu, H, Xe, H,
                                 &beta, Ue, I));
    }

    const int threads = 256;
    const int blocks_silu = (R * I + threads - 1) / threads;
    moe_swiglu_silu_mul_kernel<<<blocks_silu, threads, 0, stream>>>(d_gate, d_up, d_mid, R * I);
    LAUNCH_CHECK();

    for (int e = 0; e < num_experts; ++e) {
        const int start = h_off_host[e];
        const int n = h_off_host[e + 1] - start;
        if (n <= 0) {
            continue;
        }
        const float *Wd = d_w_down + static_cast<size_t>(e) * H * I;
        const float *Me = d_mid + static_cast<size_t>(start) * I;
        float *Oe = d_expert_out + static_cast<size_t>(start) * H;

        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, H, n, I, &alpha, Wd, I, Me, I,
                                 &beta, Oe, H));
    }

    CUDA_CHECK(
        cudaMemsetAsync(d_y, 0, static_cast<size_t>(num_tokens) * H * sizeof(float), stream));
    moe_combine_launch_cuda(stream, d_expert_out, d_src_t, d_src_k, d_route_weights, d_y, R, H,
                            top_k);
}

extern "C" void moe_swiglu_experts_forward(const float *x_host, const int *expert_ids_host,
                                           const float *route_weights_host,
                                           const float *w_gate_host, const float *w_up_host,
                                           const float *w_down_host, float *y_host, int num_tokens,
                                           int hidden_size, int intermediate_size, int num_experts,
                                           int top_k, cudaStream_t stream) {
    const int H = hidden_size;
    const int I = intermediate_size;
    const int R = num_tokens * top_k;
    const size_t x_sz = static_cast<size_t>(num_tokens) * H;
    const size_t y_sz = x_sz;
    const size_t rk_sz = static_cast<size_t>(num_tokens) * top_k;
    const size_t w_gu_sz = static_cast<size_t>(num_experts) * I * H;
    const size_t w_d_sz = static_cast<size_t>(num_experts) * H * I;
    const size_t ri_sz = static_cast<size_t>(R) * I;

    float *d_x = nullptr, *d_y = nullptr, *d_rw = nullptr, *d_wg = nullptr, *d_wu = nullptr,
          *d_wd = nullptr;
    float *d_perm = nullptr, *d_gate = nullptr, *d_up = nullptr, *d_mid = nullptr, *d_out = nullptr;
    int *d_ids = nullptr, *d_src_t = nullptr, *d_src_k = nullptr, *d_off = nullptr,
        *d_counts = nullptr, *d_next = nullptr;

    std::vector<int> h_off(static_cast<size_t>(num_experts) + 1);

    const bool own_stream = (stream == nullptr);
    cudaStream_t s = stream;
    if (own_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaMalloc(&d_x, x_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, y_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ids, rk_sz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rw, rk_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wg, w_gu_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wu, w_gu_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wd, w_d_sz * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_perm, static_cast<size_t>(R) * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src_t, rk_sz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_src_k, rk_sz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_off, static_cast<size_t>(num_experts + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, static_cast<size_t>(num_experts) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next, static_cast<size_t>(num_experts) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gate, ri_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up, ri_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mid, ri_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(R) * H * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(d_x, x_host, x_sz * sizeof(float), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(
        cudaMemcpyAsync(d_ids, expert_ids_host, rk_sz * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_rw, route_weights_host, rk_sz * sizeof(float),
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(
        cudaMemcpyAsync(d_wg, w_gate_host, w_gu_sz * sizeof(float), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(
        cudaMemcpyAsync(d_wu, w_up_host, w_gu_sz * sizeof(float), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(
        cudaMemcpyAsync(d_wd, w_down_host, w_d_sz * sizeof(float), cudaMemcpyHostToDevice, s));

    moe_swiglu_experts_forward_cublas_device(s, handle, d_x, d_ids, d_rw, d_wg, d_wu, d_wd, d_y,
                                             num_tokens, H, I, num_experts, top_k, d_perm, d_src_t,
                                             d_src_k, d_off, d_counts, d_next, d_gate, d_up, d_mid,
                                             d_out, h_off.data());

    CUDA_CHECK(cudaMemcpyAsync(y_host, d_y, y_sz * sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_ids));
    CUDA_CHECK(cudaFree(d_rw));
    CUDA_CHECK(cudaFree(d_wg));
    CUDA_CHECK(cudaFree(d_wu));
    CUDA_CHECK(cudaFree(d_wd));
    CUDA_CHECK(cudaFree(d_perm));
    CUDA_CHECK(cudaFree(d_src_t));
    CUDA_CHECK(cudaFree(d_src_k));
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_mid));
    CUDA_CHECK(cudaFree(d_out));

    if (own_stream) {
        CUDA_CHECK(cudaStreamDestroy(s));
    }
}
