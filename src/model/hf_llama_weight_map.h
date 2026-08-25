#pragma once

#include "weight_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

// 将 HF 模型中的 weight 名映射为该推理引擎中的对应名称（layer{i}.w_* / embed / final_norm /
// lm_head）。 2D 权重：HF PyTorch Linear 是 [out, in]，内部 fixture 是 row-major [in,
// out]，需要转置。 1D RMSNorm 权重：只改名，shape 不变。未识别的 key 打 stderr 警告并跳过。 0
// 成功；-1 表示 result 无效，或映射后出现重复内部名。
int hf_llama_map_weight_load_result(WeightLoadResult *result);

// 读 HF Llama 的 config.json，填进 ModelConfig。
// 例：hidden_size=2048、num_attention_heads=32、没有 head_dim 字段 -> head_dim=2048/32=64；
//     num_hidden_layers=1 -> num_layers=1；tie_word_embeddings true -> 1。
// 0 成功；-1 缺字段、无法整除或 validate 失败。
int hf_llama_parse_config_json(const char *path, ModelConfig *cfg);

#ifdef __cplusplus
}
#endif
