#include "generate_loop.h"

#include <stdio.h>
#include <vector>

#include "cuda_utils.h"
#include "inference_engine.h"
#include "transformer_model.h"

extern "C" int sampler_greedy_host(const float *logits, int vocab_size) {
    if (logits == nullptr || vocab_size <= 0) {
        return -1;
    }
    int best = 0;
    float best_val = logits[0];
    for (int v = 1; v < vocab_size; ++v) {
        if (logits[v] > best_val) {
            best_val = logits[v];
            best = v;
        }
    }
    return best;
}

// TODO(生产化)：骨架期 host <-> GPU 胶水；迁入 inference_engine_forward_token_last_logits，
//   d_token_ids/d_logits 用 Engine BufferPool 按 T 复用，去掉每步 cudaMalloc（见 roadmap 模块 4
//   生产化 / §2.6）。
// H2D token_ids -> forward_token_device -> D2H 末 token logits [vocab]
static int forward_token_step(InferenceEngine *engine, const int *token_ids_host, int num_tokens,
                              int pos_offset, int vocab_size, float *logits_last_host) {
    cudaStream_t stream = static_cast<cudaStream_t>(inference_engine_get_stream(engine));
    const size_t logits_bytes =
        static_cast<size_t>(vocab_size) * static_cast<size_t>(num_tokens) * sizeof(float);

    int *d_token_ids = nullptr;
    float *d_logits = nullptr;

    CUDA_CHECK(
        cudaMallocAsync(&d_token_ids, static_cast<size_t>(num_tokens) * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_logits, logits_bytes, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_token_ids, token_ids_host,
                               static_cast<size_t>(num_tokens) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    const int rc = inference_engine_forward_token_device(engine, d_token_ids, d_logits, num_tokens,
                                                         pos_offset);

    if (rc == 0) {
        const float *d_logits_last =
            d_logits + static_cast<size_t>(vocab_size) * static_cast<size_t>(num_tokens - 1);
        CUDA_CHECK(cudaMemcpyAsync(logits_last_host, d_logits_last,
                                   static_cast<size_t>(vocab_size) * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaFreeAsync(d_token_ids, stream));
    CUDA_CHECK(cudaFreeAsync(d_logits, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return rc;
}

// Module 4 GenerateLoop 入口：在已有 Engine session 上跑 prefill + decode 循环，greedy 出 token。
//
// Big picture 里它在哪？
//   Model + Engine(session) -> [本函数] -> out_token_ids（新生成 token 列表）
//   借用 engine 指针；不 create/destroy Engine、不 own KV/权重（全在 Engine 内）。
//   每步 forward 调 inference_engine_forward_token_device（生产 GPU 路径）。
//   TODO(生产化)：去掉 forward_token_step；GenerateLoop 只保留循环 + stop（见 roadmap 模块 4
//   生产化）。 host 仅负责 prompt/token id 与 greedy 采样；图纸：doc/design/phase2_lifecycle.md §4
//
// 函数内部顺序（例：prompt_len=3, max_new_tokens=2, cache 从 0 起）：
//   step 1) prefill：forward_token_device(prompt[3], pos=0) -> 末列 logits -> greedy -> token_0
//   step 2) decode：forward_token_device([token_0], T=1, pos=3) -> greedy -> token_1
//   KV 末态：cache_len = prompt_len + num_generated - 1
//
// 返回：写入 out_token_ids 的新 token 个数；eos_token_id<0 表示不启用 EOS 早停。
extern "C" int generate_loop_run(InferenceEngine *engine, const int *prompt_token_ids,
                                 int prompt_len, int max_new_tokens, int eos_token_id,
                                 int *out_token_ids, int out_capacity) {
    if (engine == nullptr || prompt_token_ids == nullptr || prompt_len <= 0 ||
        max_new_tokens <= 0 || out_token_ids == nullptr || out_capacity < max_new_tokens) {
        return -1;
    }

    const TransformerModel *model = inference_engine_get_model(engine);
    const ModelConfig *cfg = transformer_model_get_config(model);
    if (cfg == nullptr || transformer_model_is_weights_loaded(model) != 1) {
        return -1;
    }

    const int vocab_size = cfg->vocab_size;
    const int max_seq_len = cfg->max_seq_len;
    // step 0：max_seq 边界
    if (prompt_len > max_seq_len) {
        fprintf(stderr, "generate_loop_run: prompt_len exceeds max_seq_len\n");
        return -1;
    }
    if (prompt_len + max_new_tokens - 1 > max_seq_len) {
        fprintf(stderr, "generate_loop_run: would exceed max_seq_len\n");
        return -1;
    }

    // step 1：prefill + 首个 greedy token
    std::vector<float> logits_last(static_cast<size_t>(vocab_size));
    if (forward_token_step(engine, prompt_token_ids, prompt_len, 0, vocab_size,
                           logits_last.data()) != 0) {
        fprintf(stderr, "generate_loop_run: prefill forward failed\n");
        return -1;
    }

    int num_generated = 0;
    int next_token = sampler_greedy_host(logits_last.data(), vocab_size);
    if (next_token < 0) {
        return -1;
    }
    out_token_ids[num_generated++] = next_token;
    if (eos_token_id >= 0 && next_token == eos_token_id) {
        return num_generated;
    }

    // step 2：decode 循环（每步 T=1，pos=当前 cache_len）
    int decode_token = next_token;
    for (int step = 1; step < max_new_tokens; ++step) {
        const int cache_len = inference_engine_kv_cache_len(engine);
        if (cache_len + 1 > max_seq_len) {
            fprintf(stderr, "generate_loop_run: decode exceeds max_seq_len\n");
            return -1;
        }
        if (forward_token_step(engine, &decode_token, 1, cache_len, vocab_size,
                               logits_last.data()) != 0) {
            fprintf(stderr, "generate_loop_run: decode forward failed\n");
            return -1;
        }
        next_token = sampler_greedy_host(logits_last.data(), vocab_size);
        if (next_token < 0) {
            return -1;
        }
        out_token_ids[num_generated++] = next_token;
        decode_token = next_token;
        if (eos_token_id >= 0 && next_token == eos_token_id) {
            break;
        }
    }

    return num_generated;
}
