#pragma once

#include "weight_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

// 将 HF 模型中的 weight 名映射为该推理引擎中的对应名称（layer{i}.w_* / embed / final_norm /
// lm_head）。
//
// Linear（q/k/v/o/gate/up/down）：HF 已经是 PyTorch [out, in]，和
// linear_forward_device 一样，只改名不转置。
// 例：Q 的 in=4、out=2，HF q_proj.weight 行主序
//   [[1, 2, 3, 4],
//    [5, 6, 7, 8]]
// 就是 W[o, i]。linear 用 CUBLAS_OP_T、lda=in 读这块，算 y = W @ x。
// 若再转成 [in, out]，Engine 实际变成 W^T @ x，logits 会差一个数量级。
//
// lm_head：HF 是 [vocab, hidden]，untied_lm_head 要 [hidden, vocab]，仍转置。
// 1D RMSNorm：只改名。未识别 key 打 stderr 并跳过。
// 0 成功；-1 表示 result 无效，或映射后出现重复内部名。
int hf_llama_map_weight_load_result(WeightLoadResult *result);

// 读 HF Llama 的 config.json，填进 ModelConfig。
// 例：hidden_size=2048、num_attention_heads=32、没有 head_dim 字段 -> head_dim=2048/32=64；
//     num_hidden_layers=1 -> num_layers=1；tie_word_embeddings true -> 1。
// 0 成功；-1 缺字段、无法整除或 validate 失败。
int hf_llama_parse_config_json(const char *path, ModelConfig *cfg);

#ifdef __cplusplus
}
#endif
