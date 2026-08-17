#include "sampler_top_k.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "cuda_utils.cuh"
#include "cuda_utils.h"

#define SAMPLER_BLOCK_SIZE 256
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

static __device__ __forceinline__ int warp_argmax(float val, int idx) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_val = __shfl_xor_sync(0xffffffff, val, offset);
        const int other_idx = __shfl_xor_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    return idx;
}

static __global__ void sampler_greedy_kernel(const float *logits, int vocab_size, int *out_token) {
    float local_max = -INFINITY;
    int local_idx = 0;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        const float logit = logits[v];
        if (logit > local_max) {
            local_max = logit;
            local_idx = v;
        }
    }

    local_idx = warp_argmax(local_max, local_idx);

    __shared__ int warp_best[SAMPLER_BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp_best[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float best_val = -INFINITY;
        int best_idx = 0;
        const int num_warps = (blockDim.x + 31) >> 5;
        if (lane < num_warps) {
            best_idx = warp_best[lane];
            best_val = logits[best_idx];
        }
        best_idx = warp_argmax(best_val, best_idx);
        if (lane == 0) {
            out_token[0] = best_idx;
        }
    }
}

static __global__ void sampler_top_k_kernel(const float *logits, int vocab_size, int top_k,
                                            float temperature, uint64_t seed, int *out_token) {
    // TODO(perf-topk): 现仅 thread 0 顺序扫 vocab（<<<1,1>>>），大词表 decode 性能差。
    //   改：每 thread 维护 local top-k -> block merge；或 CUB select-k / fused mask+softmax+sample
    //   （对齐 vLLM Triton / SGLang FlashInfer）。k==1 已用 sampler_greedy_kernel 并行 argmax。
    //   定位：grep TODO(perf-topk)；doc/guide/generate_loop_device_api.md
    __shared__ float sm_logits[SAMPLER_MAX_TOP_K];
    __shared__ int sm_indices[SAMPLER_MAX_TOP_K];

    if (threadIdx.x == 0) {
        for (int i = 0; i < top_k; ++i) {
            sm_logits[i] = -INFINITY;
            sm_indices[i] = 0;
        }
        for (int v = 0; v < vocab_size; ++v) {
            sampler_topk_insert(logits[v], v, sm_logits, sm_indices, top_k);
        }

        const float max_logit = sm_logits[0];
        float sum = 0.f;
        for (int i = 0; i < top_k; ++i) {
            sm_logits[i] = expf((sm_logits[i] - max_logit) / temperature);
            sum += sm_logits[i];
        }
        if (sum <= 0.f) {
            out_token[0] = sm_indices[0];
            return;
        }
        for (int i = 0; i < top_k; ++i) {
            sm_logits[i] /= sum;
        }

        uint64_t rng_state = sampler_mix_seed_dev(seed);
        const float u = sampler_uniform01(&rng_state);
        float acc = 0.f;
        int chosen = sm_indices[0];
        for (int i = 0; i < top_k; ++i) {
            acc += sm_logits[i];
            if (u <= acc) {
                chosen = sm_indices[i];
                break;
            }
        }
        out_token[0] = chosen;
    }
}

static int sampler_check_args(const float *d_logits, int vocab_size, int top_k, float temperature,
                              int *d_out_token) {
    if (d_logits == nullptr || d_out_token == nullptr || vocab_size <= 0 || top_k <= 0) {
        return -1;
    }
    if (temperature <= 0.f) {
        std::fprintf(stderr, "sampler_top_k_device: temperature must be positive\n");
        return -1;
    }
    if (top_k > SAMPLER_MAX_TOP_K) {
        std::fprintf(stderr, "sampler_top_k_device: top_k=%d exceeds max %d\n", top_k,
                     SAMPLER_MAX_TOP_K);
        return -1;
    }
    return 0;
}

extern "C" int sampler_top_k_device(void *stream, const float *d_logits, int vocab_size, int top_k,
                                    float temperature, uint64_t seed, int *d_out_token) {
    if (sampler_check_args(d_logits, vocab_size, top_k, temperature, d_out_token) != 0) {
        return -1;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    if (top_k == 1) {
        sampler_greedy_kernel<<<1, SAMPLER_BLOCK_SIZE, 0, s>>>(d_logits, vocab_size, d_out_token);
    } else {
        const int k = top_k > vocab_size ? vocab_size : top_k;
        // TODO(perf-topk): launch config <<<1,1>>>；见 sampler_top_k_kernel 内注释
        sampler_top_k_kernel<<<1, 1, 0, s>>>(d_logits, vocab_size, k, temperature, seed,
                                             d_out_token);
    }
    LAUNCH_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(s));
    return 0;
}

extern "C" int sampler_top_k_host(const float *logits_host, int vocab_size, int top_k,
                                  float temperature, uint64_t seed) {
    if (logits_host == nullptr || vocab_size <= 0 || top_k <= 0 || temperature <= 0.f) {
        return -1;
    }

    float *d_logits = nullptr;
    int *d_out_token = nullptr;
    int out_token = -1;

    CUDA_CHECK(cudaMalloc(&d_logits, static_cast<size_t>(vocab_size) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_token, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_logits, logits_host, static_cast<size_t>(vocab_size) * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (sampler_top_k_device(nullptr, d_logits, vocab_size, top_k, temperature, seed,
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
