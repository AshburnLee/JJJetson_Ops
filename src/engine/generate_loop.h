#pragma once

#include <stddef.h>

typedef struct InferenceEngine InferenceEngine;

#ifdef __cplusplus
extern "C" {
#endif

// Greedy argmax on logits [vocab_size] (host).
int sampler_greedy_host(const float *logits, int vocab_size);

// Prefill prompt then decode loop. Writes newly generated tokens to out_token_ids.
// eos_token_id < 0 disables EOS early stop.
// Returns number of generated tokens, or -1 on error.
int generate_loop_run(InferenceEngine *engine, const int *prompt_token_ids, int prompt_len,
                      int max_new_tokens, int eos_token_id, int *out_token_ids, int out_capacity);

#ifdef __cplusplus
}
#endif
