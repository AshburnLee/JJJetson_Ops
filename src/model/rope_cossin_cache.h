#pragma once

#ifdef __cplusplus
extern "C" {
#endif
// 语义上是 per-model singleton, 每一个model一份Cache，其随model销毁
typedef struct RopeCosSinCache RopeCosSinCache;

RopeCosSinCache *rope_cossin_cache_create(int max_len, int n_dims, float freq_base);

void rope_cossin_cache_destroy(RopeCosSinCache *cache);

// 返回 Cache Device 指针，共 kernel中访问
const float *rope_cossin_cache_device_ptr(const RopeCosSinCache *cache);

int rope_cossin_cache_max_len(const RopeCosSinCache *cache);

int rope_cossin_cache_n_dims(const RopeCosSinCache *cache);

// 访问Cache前检查 pos[0..n_tokens) ，均在 [0, max_len)
int rope_cossin_cache_check_pos(const RopeCosSinCache *cache, const int *pos, int n_tokens);

#ifdef __cplusplus
}
#endif
