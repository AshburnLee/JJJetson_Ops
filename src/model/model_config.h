#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// LLaMA-style transformer 静态超参；Loader 填充后供 Model / Engine 只读使用。
typedef struct ModelConfig {
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    int max_seq_len;
    float freq_base;
    float rms_norm_epsilon;
    int tie_word_embeddings; // 0 = separate lm_head, 1 = tied with embed
} ModelConfig;

// 校验维度一致性；0 成功，-1 非法参数。
int model_config_validate(const ModelConfig *cfg);

#ifdef __cplusplus
}
#endif
