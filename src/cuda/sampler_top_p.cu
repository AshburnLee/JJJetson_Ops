#include "sampler_top_p.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "cuda_utils.cuh"
#include "cuda_utils.h"

#define SAMPLER_MAX_TOP_K 128

static __device__ __forceinline__ uint64_t sampler_mix_seed_dev(uint64_t seed) {
    return seed == 0 ? UINT64_C(0x123456789abcdef) : seed;
}

static __device__ __forceinline__ uint64_t sampler_xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static __device__ __forceinline__ float sampler_uniform01(uint64_t *state) {
    const uint64_t mantissa = (sampler_xorshift64(state) >> 11) & ((1ULL << 53) - 1);
    return static_cast<float>(mantissa) * (1.0f / static_cast<float>(1ULL << 53));
}

static __device__ __forceinline__ void sampler_topk_insert(float logit, int idx, float *top_logits,
                                                           int *top_indices, int k) {
    if (logit <= top_logits[k - 1]) {
        return;
    }
    top_logits[k - 1] = logit;
    top_indices[k - 1] = idx;
    for (int i = k - 1; i > 0; --i) {
        if (top_logits[i] <= top_logits[i - 1]) {
            break;
        }
        const float tmp_logit = top_logits[i - 1];
        const int tmp_idx = top_indices[i - 1];
        top_logits[i - 1] = top_logits[i];
        top_indices[i - 1] = top_indices[i];
        top_logits[i] = tmp_logit;
        top_indices[i] = tmp_idx;
    }
}

static __global__ void sampler_top_p_kernel(const float *logits, int vocab_size, float top_p,
                                            float temperature, int top_k, uint64_t seed,
                                            float *probs, int *indices, int *out_token) {
    // TODO(perf-topp): 现 <<<1,1>>> 单线程 + O(n^2) 选择排序 + 每步 cudaMallocAsync 临时 buffer。
    //   改：并行 softmax/sort（CUB/thrust）+ nucleus mask + fused sample；或与 top-k 合并 fused
    //   kernel （对齐 vLLM Triton / SGLang FlashInfer）。定位：grep
    //   TODO(perf-topp)；doc/guide/sampler-top-p.md
    if (threadIdx.x != 0) {
        return;
    }

    int n = vocab_size;
    if (top_k > 0 && top_k < vocab_size) {
        n = top_k;
        for (int i = 0; i < n; ++i) {
            probs[i] = -INFINITY;
            indices[i] = 0;
        }
        for (int v = 0; v < vocab_size; ++v) {
            sampler_topk_insert(logits[v], v, probs, indices, n);
        }
    } else {
        for (int v = 0; v < vocab_size; ++v) {
            probs[v] = logits[v];
            indices[v] = v;
        }
    }

    float max_logit = probs[0];
    for (int i = 1; i < n; ++i) {
        if (probs[i] > max_logit) {
            max_logit = probs[i];
        }
    }

    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        probs[i] = expf((probs[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.f) {
        out_token[0] = indices[0];
        return;
    }
    for (int i = 0; i < n; ++i) {
        probs[i] /= sum;
    }

    for (int i = 0; i < n - 1; ++i) {
        int best = i;
        for (int j = i + 1; j < n; ++j) {
            if (probs[j] > probs[best]) {
                best = j;
            }
        }
        if (best != i) {
            const float tmp_p = probs[i];
            const int tmp_idx = indices[i];
            probs[i] = probs[best];
            indices[i] = indices[best];
            probs[best] = tmp_p;
            indices[best] = tmp_idx;
        }
    }

    float cum = 0.f;
    int nucleus = 1;
    for (int i = 0; i < n; ++i) {
        cum += probs[i];
        nucleus = i + 1;
        if (cum >= top_p) {
            break;
        }
    }

    float nucleus_sum = 0.f;
    for (int i = 0; i < nucleus; ++i) {
        nucleus_sum += probs[i];
    }
    if (nucleus_sum <= 0.f) {
        out_token[0] = indices[0];
        return;
    }

    uint64_t rng_state = sampler_mix_seed_dev(seed);
    const float u = sampler_uniform01(&rng_state);
    float acc = 0.f;
    int chosen = indices[0];
    for (int i = 0; i < nucleus; ++i) {
        acc += probs[i] / nucleus_sum;
        if (u <= acc) {
            chosen = indices[i];
            break;
        }
    }
    out_token[0] = chosen;
}

static int sampler_top_p_check_args(const float *d_logits, int vocab_size, float top_p,
                                    float temperature, int top_k, int *d_out_token) {
    if (d_logits == nullptr || d_out_token == nullptr || vocab_size <= 0) {
        return -1;
    }
    if (top_p <= 0.f || top_p > 1.f) {
        std::fprintf(stderr, "sampler_top_p_device: top_p must be in (0, 1]\n");
        return -1;
    }
    if (temperature <= 0.f) {
        std::fprintf(stderr, "sampler_top_p_device: temperature must be positive\n");
        return -1;
    }
    if (top_k > SAMPLER_MAX_TOP_K) {
        std::fprintf(stderr, "sampler_top_p_device: top_k=%d exceeds max %d\n", top_k,
                     SAMPLER_MAX_TOP_K);
        return -1;
    }
    return 0;
}

extern "C" int sampler_top_p_device(void *stream, const float *d_logits, int vocab_size,
                                    float top_p, float temperature, int top_k, uint64_t seed,
                                    int *d_out_token) {
    if (sampler_top_p_check_args(d_logits, vocab_size, top_p, temperature, top_k, d_out_token) !=
        0) {
        return -1;
    }

    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const size_t n_bytes = static_cast<size_t>(vocab_size) * sizeof(float);
    const size_t i_bytes = static_cast<size_t>(vocab_size) * sizeof(int);

    float *d_probs = nullptr;
    int *d_indices = nullptr;
    // TODO(perf-topp): 每 call 分配 probs/indices；生产改 session 级 workspace 复用或 fused kernel
    // 无 scratch
    CUDA_CHECK(cudaMallocAsync(&d_probs, n_bytes, s));
    CUDA_CHECK(cudaMallocAsync(&d_indices, i_bytes, s));

    // TODO(perf-topp): launch config <<<1,1>>>；见 sampler_top_p_kernel 内注释
    sampler_top_p_kernel<<<1, 1, 0, s>>>(d_logits, vocab_size, top_p, temperature, top_k, seed,
                                         d_probs, d_indices, d_out_token);
    LAUNCH_CHECK();

    CUDA_CHECK(cudaFreeAsync(d_probs, s));
    CUDA_CHECK(cudaFreeAsync(d_indices, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return 0;
}

extern "C" int sampler_top_p_host(const float *logits_host, int vocab_size, float top_p,
                                  float temperature, int top_k, uint64_t seed) {
    if (logits_host == nullptr || vocab_size <= 0 || top_p <= 0.f || top_p > 1.f ||
        temperature <= 0.f) {
        return -1;
    }

    float *d_logits = nullptr;
    int *d_out_token = nullptr;
    int out_token = -1;

    CUDA_CHECK(cudaMalloc(&d_logits, static_cast<size_t>(vocab_size) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_token, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_logits, logits_host, static_cast<size_t>(vocab_size) * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (sampler_top_p_device(nullptr, d_logits, vocab_size, top_p, temperature, top_k, seed,
                             d_out_token) != 0) {
        cudaFree(d_logits);
        cudaFree(d_out_token);
        return -1;
    }

    CUDA_CHECK(cudaMemcpy(&out_token, d_out_token, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_out_token));
    return out_token;
}
