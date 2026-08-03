#include "embed.h"

#include <cuda_runtime.h>
#include <cstdio>

#include <cuda_utils.cuh>

// Token embedding gather：按 token id 查表，无矩阵乘。
//
// 输入：
//   embed      row-major [vocab_size, hidden_size]，embed[v, h] 在 embed[v*H + h]
//   token_ids  [num_tokens]，第 t 个位置的 token id
// 输出：
//   hidden     col-major [hidden_size, num_tokens]，hidden[h, t] 在 hidden[h + t*H]
//
// 对每个 token 位置 t：
//   id = token_ids[t]
//   hidden[h, t] = embed[id, h]   （h = 0 .. hidden_size-1）
static __global__ void embed_forward_kernel(const float *__restrict__ embed,
                                            const int *__restrict__ token_ids,
                                            float *__restrict__ hidden, int hidden_size,
                                            int num_tokens) {
    const int t = blockIdx.x;
    if (t >= num_tokens) {
        return;
    }

    const int id = token_ids[t];
    // embed 第 id 行起点：row-major [vocab, hidden]
    const float *row = embed + static_cast<int64_t>(id) * hidden_size;
    // hidden 第 t 列起点：col-major [hidden, num_tokens]
    const int64_t out_base = static_cast<int64_t>(t) * hidden_size;

    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        hidden[out_base + h] = row[h];
    }
}

static int embed_check_shape(int hidden_size, int num_tokens) {
    if (hidden_size <= 0 || num_tokens <= 0) {
        std::fprintf(stderr, "embed_forward_device: invalid hidden_size=%d num_tokens=%d\n",
                     hidden_size, num_tokens);
        return -1;
    }
    return 0;
}

extern "C" int embed_forward_device(void *stream, const float *d_embed, const int *d_token_ids,
                                    float *d_hidden, int hidden_size, int num_tokens) {
    if (embed_check_shape(hidden_size, num_tokens) != 0) {
        return -1;
    }
    if (d_embed == nullptr || d_token_ids == nullptr || d_hidden == nullptr) {
        std::fprintf(stderr, "embed_forward_device: null pointer argument\n");
        return -1;
    }

    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int block_size = hidden_size < 256 ? hidden_size : 256;
    const dim3 threads(block_size, 1, 1);
    const dim3 blocks(static_cast<unsigned>(num_tokens), 1, 1);

#if defined(MY_OPS_DEBUG)
    std::printf("embed_forward_device launch: hidden=%d tokens=%d\n", hidden_size, num_tokens);
    std::fflush(stdout);
#endif

    embed_forward_kernel<<<blocks, threads, 0, s>>>(d_embed, d_token_ids, d_hidden, hidden_size,
                                                    num_tokens);
    LAUNCH_CHECK();
    return 0;
}

extern "C" void embed_forward_host(const float *embed_host, const int *token_ids_host,
                                   float *hidden_host, int hidden_size, int vocab_size,
                                   int num_tokens) {
    if (embed_check_shape(hidden_size, num_tokens) != 0 || vocab_size <= 0) {
        return;
    }
    if (embed_host == nullptr || token_ids_host == nullptr || hidden_host == nullptr) {
        std::fprintf(stderr, "embed_forward_host: null pointer argument\n");
        return;
    }

    const int64_t n_hidden = static_cast<int64_t>(hidden_size) * num_tokens;
    const int64_t n_embed = static_cast<int64_t>(vocab_size) * hidden_size;

    float *d_embed = nullptr;
    int *d_token_ids = nullptr;
    float *d_hidden = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMallocAsync(&d_embed, static_cast<size_t>(n_embed) * sizeof(float), stream));
    CUDA_CHECK(
        cudaMallocAsync(&d_token_ids, static_cast<size_t>(num_tokens) * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_hidden, static_cast<size_t>(n_hidden) * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_embed, embed_host, static_cast<size_t>(n_embed) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_token_ids, token_ids_host,
                               static_cast<size_t>(num_tokens) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    embed_forward_device(stream, d_embed, d_token_ids, d_hidden, hidden_size, num_tokens);

    CUDA_CHECK(cudaMemcpyAsync(hidden_host, d_hidden, static_cast<size_t>(n_hidden) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_embed, stream));
    CUDA_CHECK(cudaFreeAsync(d_token_ids, stream));
    CUDA_CHECK(cudaFreeAsync(d_hidden, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
