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

#ifdef __cplusplus
}
#endif
