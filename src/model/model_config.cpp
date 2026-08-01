#include "model_config.h"

int model_config_validate(const ModelConfig *cfg) {
    if (cfg == nullptr) {
        return -1;
    }
    if (cfg->hidden_size <= 0 || cfg->intermediate_size <= 0 || cfg->num_layers <= 0 ||
        cfg->num_q_heads <= 0 || cfg->num_kv_heads <= 0 || cfg->head_dim <= 0 ||
        cfg->vocab_size <= 0 || cfg->max_seq_len <= 0) {
        return -1;
    }
    if (cfg->num_kv_heads > cfg->num_q_heads) {
        return -1;
    }
    if (cfg->num_q_heads * cfg->head_dim != cfg->hidden_size) {
        return -1;
    }
    if (cfg->freq_base <= 0.f || cfg->rms_norm_epsilon <= 0.f) {
        return -1;
    }
    if (cfg->tie_word_embeddings != 0 && cfg->tie_word_embeddings != 1) {
        return -1;
    }
    return 0;
}
