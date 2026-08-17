#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct InferenceEngine InferenceEngine;

#ifdef __cplusplus
extern "C" {
#endif

// Prefill prompt then decode loop. Writes newly generated tokens to out_token_ids.
// eos_token_id < 0 disables EOS early stop. top_k <= 0 or temperature <= 0 is invalid.
// top_k==1 is greedy (temperature ignored). Sampling uses sampler_top_k_device on GPU logits.
int generate_loop_run(InferenceEngine *engine, const int *prompt_token_ids, int prompt_len,
                      int max_new_tokens, int eos_token_id, int top_k, float temperature,
                      uint64_t seed, int *out_token_ids, int out_capacity);

#ifdef __cplusplus
}
#endif
