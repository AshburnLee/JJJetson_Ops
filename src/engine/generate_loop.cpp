#include "generate_loop.h"

#include <stdio.h>

#include "inference_engine.h"
#include "nvtx_range.h"
#include "transformer_model.h"

// Module 4 GenerateLoop 生产入口：在已有 Engine session 上跑完 [prefill 一次 + decode 若干次]，
// 把新生成的 token id 写进 out_token_ids，返回实际生成个数。
//
// Big picture 里它在哪？
//   Loader / Model(权重) / Engine.create 都已经做完；
//   调用方把 InferenceEngine* 借给 [本函数]，本函数只驱动循环和停条件。
//   每步调 ie_forward_token_sample（forward + 采样 + D2H token），
//   本文件不碰 stream / memcpy / sampler kernel。
//   不 own Engine / KV / 权重，也不 create、destroy、reset session。
//   对照：Engine 的 forward_token_device 是单步；本函数是把单步串成一次 generate。
//   图纸：doc/design/phase2_lifecycle.md §4；doc/guide/generate_loop_device_api.md
//
// 函数内部顺序（例：prompt_ids=[3,17,42]，prompt_len=3，max_new_tokens=6，eos 关闭）：
//   INPUT：cache_len=0，out 空
//   step 0. 校验指针、max_new、采样超参；prompt 与 [prompt+max_new-1] 不得超 max_seq
//   step 1. prefill：T=3、pos 从 0。Engine sample 一步，cache_len 变成 3，得到 x0
//   step 2. out[0]=x0。若 x0 等于 eos_token_id（且 eos>=0）则到此返回 1
//   step 3. decode 循环 step=1..5：每次 T=1，pos=当前 cache_len，输入是上一步的 token。
//           第 1 次 decode 输入 x0、cache_len=3；得到 x1，cache_len 变成 4；依此类推。
//           采到 eos 就提前停。OUTPUT：out=[x0..x5] 共 6 个 id，cache_len=8
//           （公式：len(prompt)+num_generated-1 = 3+6-1）
//
// 调用契约：engine 必须已 create 且权重已 load；out_token_ids 由调用方分配，容量 >= max_new。
// eos_token_id<0 表示不启用早停。失败返回 -1；成功返回写入 out 的个数（可能因 EOS 小于 max_new）。
extern "C" int generate_loop_run(InferenceEngine *engine, const int *prompt_token_ids,
                                 int prompt_len, int max_new_tokens, int eos_token_id, int top_k,
                                 float temperature, float top_p, uint64_t seed, int *out_token_ids,
                                 int out_capacity) {
    // step 0：入参与 session 合法性；后面按 prompt_len + max_new - 1 估最终 cache
    if (engine == nullptr || prompt_token_ids == nullptr || prompt_len <= 0 ||
        max_new_tokens <= 0 || out_token_ids == nullptr || out_capacity < max_new_tokens ||
        top_k <= 0 || temperature <= 0.f || top_p <= 0.f || top_p > 1.f) {
        return -1;
    }

    const TransformerModel *model = ie_get_model(engine);
    const ModelConfig *cfg = transformer_model_get_config(model);
    if (cfg == nullptr || transformer_model_is_weights_loaded(model) != 1) {
        return -1;
    }

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
    // step 1：prefill，T=prompt_len，pos 从 0；采出第一个新 token
    {
        NVTX_RANGE("prefill");
        if (ie_forward_token_sample(engine, prompt_token_ids, prompt_len, 0, top_k, temperature,
                                    top_p, seed, &next_token) != 0) {
            fprintf(stderr, "generate_loop_run: prefill forward failed\n");
            return -1;
        }
    }
    // step 2：写入 out[0]；prefill 就碰到 EOS 则不进 decode
    out_token_ids[num_generated++] = next_token;
    if (eos_token_id >= 0 && next_token == eos_token_id) {
        return num_generated;
    }

    int decode_token = next_token;
    // step 3：decode 循环，每次 T=1、pos=cache_len，输入是上一步 token；EOS 提前停
    {
        NVTX_RANGE("decode");
        for (int step = 1; step < max_new_tokens; ++step) {
            const int cache_len = ie_kv_cache_len(engine);
            if (cache_len + 1 > max_seq_len) {
                fprintf(stderr, "generate_loop_run: decode exceeds max_seq_len\n");
                return -1;
            }
            if (ie_forward_token_sample(engine, &decode_token, 1, cache_len, top_k, temperature,
                                        top_p, seed, &next_token) != 0) {
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
