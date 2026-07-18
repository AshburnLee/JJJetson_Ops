#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KVCache KVCache;

// --==============================- Session -=============================--

// 创建 session 级 KV cache：每层一块固定 GPU buffer，layout [head_dim, max_seq, num_kv_heads, 1]
KVCache *kv_cache_create(int max_seq, int head_dim, int num_kv_heads, int num_layers);

void kv_cache_destroy(KVCache *cache);

// 重置 cache_len=0，复用 GPU 内存
/*
reset 时间点：
新会话 / 新 request（换 prompt、清空对话），Prefill 开始前（新序列第一步）

不重置：
1. Prefill 切换到 Decode ，不需要 reset，
2. 同一序列每层 forward 各层公用一个 cache_len

timeline：
- create_cache          ->  cache_len = 0
- prefill (T=13)        ->  write + advance ->  cache_len = 13
- decode × N            ->  每步 +1 ->  14, 15, ...
- [用户开启新 session]
- reset (或 destroy + create) ->  cache_len = 0，开始下一个 Request
*/
void kv_cache_reset(KVCache *cache);

int kv_cache_get_len(const KVCache *cache);

int kv_cache_get_max_seq(const KVCache *cache);

int kv_cache_get_head_dim(const KVCache *cache);

int kv_cache_get_num_kv_heads(const KVCache *cache);

int kv_cache_get_num_layers(const KVCache *cache);

// 返回 layer 的 K/V device 基指针；有效长度为 kv_cache_get_len()
const float *kv_cache_get_k_device_ptr(const KVCache *cache, int layer);

const float *kv_cache_get_v_device_ptr(const KVCache *cache, int layer);

// -========================-- 生产（device）--========================-
// 将 d_k/d_v 写入 [offset, offset + n_tokens)，不修改 cache_len（一层 forward 内各 layer 共用
// offset） stream 为 void*，避免在 .h 中引入 cuda_runtime.h。在实现处，static_cast 为 cudaStream_t
int kv_cache_append_device(void *stream, KVCache *cache, int layer, const float *d_k,
                           const float *d_v, int n_tokens);

// 一个 forward step 的所有 layer 写完后，推进 cache_len
int kv_cache_advance_len(KVCache *cache, int n_tokens);

#ifdef __cplusplus
}
#endif
