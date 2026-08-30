#include "generate_loop.h"

#include <stdio.h>

#include "cuda_utils.h"
#include "inference_engine.h"
#include "nvtx_range.h"
#include "sampler_top_k.h"
#include "sampler_top_p.h"
#include "transformer_model.h"

// TODO(生产化)：骨架期 host <-> GPU 胶水；迁入 inference_engine_forward_token_last_logits，
//   d_token_ids/d_logits 用 Engine BufferPool 按 T 复用，去掉每步 cudaMalloc（见 roadmap 模块 4
//   生产化 / Phase 2.6）。
// TODO(perf-topk)：top_k>1 时 sampler_top_k.cu 为 <<<1,1>>> 单线程扫 vocab。
// TODO(perf-topp)：top_p<1 时 sampler_top_p.cu 为 <<<1,1>>> + O(n^2) sort + 每步 temp malloc。
//   大词表 decode 前需并行 kernel；grep TODO(perf-topk) / TODO(perf-topp)。
static int sample_token_device(void *stream, const float *d_logits_last, int vocab_size, int top_k,
                               float temperature, float top_p, uint64_t seed, int *d_out_token) {
    if (top_p < 1.f) {
        return sampler_top_p_device(stream, d_logits_last, vocab_size, top_p, temperature, top_k,
                                    seed, d_out_token);
    }
    return sampler_top_k_device(stream, d_logits_last, vocab_size, top_k, temperature, seed,
                                d_out_token);
}

// H2D token_ids -> forward_token_device -> sampler on末列 logits -> token id
static int forward_token_step(InferenceEngine *engine, const int *token_ids_host, int num_tokens,
                              int pos_offset, int vocab_size, int top_k, float temperature,
                              float top_p, uint64_t seed, int *out_token) {
    cudaStream_t stream = static_cast<cudaStream_t>(inference_engine_get_stream(engine));
    const size_t logits_bytes =
        static_cast<size_t>(vocab_size) * static_cast<size_t>(num_tokens) * sizeof(float);

    int *d_token_ids = nullptr;
    float *d_logits = nullptr;
    int *d_out_token = nullptr;

    CUDA_CHECK(
        cudaMallocAsync(&d_token_ids, static_cast<size_t>(num_tokens) * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_logits, logits_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_out_token, sizeof(int), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_token_ids, token_ids_host,
                               static_cast<size_t>(num_tokens) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    const int rc = inference_engine_forward_token_device(engine, d_token_ids, d_logits, num_tokens,
                                                         pos_offset);

    if (rc == 0) {
        const float *d_logits_last =
            d_logits + static_cast<size_t>(vocab_size) * static_cast<size_t>(num_tokens - 1);
        if (sample_token_device(stream, d_logits_last, vocab_size, top_k, temperature, top_p, seed,
                                d_out_token) != 0) {
            CUDA_CHECK(cudaFreeAsync(d_token_ids, stream));
            CUDA_CHECK(cudaFreeAsync(d_logits, stream));
            CUDA_CHECK(cudaFreeAsync(d_out_token, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return -1;
        }
        CUDA_CHECK(
            cudaMemcpyAsync(out_token, d_out_token, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaFreeAsync(d_token_ids, stream));
    CUDA_CHECK(cudaFreeAsync(d_logits, stream));
    CUDA_CHECK(cudaFreeAsync(d_out_token, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return rc;
}

// Module 4 GenerateLoop 入口：在已有 Engine session 上跑 prefill + decode 循环。
extern "C" int generate_loop_run(InferenceEngine *engine, const int *prompt_token_ids,
                                 int prompt_len, int max_new_tokens, int eos_token_id, int top_k,
                                 float temperature, float top_p, uint64_t seed, int *out_token_ids,
                                 int out_capacity) {
    if (engine == nullptr || prompt_token_ids == nullptr || prompt_len <= 0 ||
        max_new_tokens <= 0 || out_token_ids == nullptr || out_capacity < max_new_tokens ||
        top_k <= 0 || temperature <= 0.f || top_p <= 0.f || top_p > 1.f) {
        return -1;
    }

    const TransformerModel *model = inference_engine_get_model(engine);
    const ModelConfig *cfg = transformer_model_get_config(model);
    if (cfg == nullptr || transformer_model_is_weights_loaded(model) != 1) {
        return -1;
    }

    const int vocab_size = cfg->vocab_size;
    const int max_seq_len = cfg->max_seq_len;
    if (prompt_len > max_seq_len) {
        fprintf(stderr, "generate_loop_run: prompt_len exceeds max_seq_len\n");
        return -1;
    }
    if (prompt_len + max_new_tokens - 1 > max_seq_len) {
        fprintf(stderr, "generate_loop_run: would exceed max_seq_len\n");
        return -1;
    }

    name_engine_thread();
    NVTX_RANGE("generate");

    int num_generated = 0;
    int next_token = 0;
    // TODO: d_token_ids、d_logits、d_out_token ,Generate 10 个 token，就是 10 轮 malloc / memcpy /
    // free session、stream、GPU buffer 是 Engine 的东西，但是malloc/free 却在 loop 中
    {
        NVTX_RANGE("prefill");
        if (forward_token_step(engine, prompt_token_ids, prompt_len, 0, vocab_size, top_k,
                               temperature, top_p, seed, &next_token) != 0) {
            fprintf(stderr, "generate_loop_run: prefill forward failed\n");
            return -1;
        }
    }
    out_token_ids[num_generated++] = next_token;
    if (eos_token_id >= 0 && next_token == eos_token_id) {
        return num_generated;
    }

    int decode_token = next_token;
    {
        NVTX_RANGE("decode");
        for (int step = 1; step < max_new_tokens; ++step) {
            const int cache_len = inference_engine_kv_cache_len(engine);
            if (cache_len + 1 > max_seq_len) {
                fprintf(stderr, "generate_loop_run: decode exceeds max_seq_len\n");
                return -1;
            }
            if (forward_token_step(engine, &decode_token, 1, cache_len, vocab_size, top_k,
                                   temperature, top_p, seed, &next_token) != 0) {
                fprintf(stderr, "generate_loop_run: decode forward failed\n");
                return -1;
            }
            out_token_ids[num_generated++] = next_token;
            decode_token = next_token;
            if (eos_token_id >= 0 && next_token == eos_token_id) {
                break;
            }
        }
    }

    return num_generated;
}
