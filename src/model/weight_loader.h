#pragma once

#include <stddef.h>
#include <stdint.h>

#include "model_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum WeightDtype {
    WEIGHT_DTYPE_F32 = 0,
} WeightDtype;

// 单次 load 产出的 host 权重条目；data/dims/name 由 Loader 分配，result_destroy 释放。
typedef struct HostTensor {
    char *name;
    float *data;
    int64_t *dims;
    int ndim;
    WeightDtype dtype;
} HostTensor;

// Loader 输出：ModelConfig + name→tensor 表；无 GPU 对象，调用方 H2D 后交给 Model。
typedef struct WeightLoadResult {
    ModelConfig config;
    HostTensor *tensors;
    int num_tensors;
} WeightLoadResult;

void weight_load_result_init(WeightLoadResult *out);

void weight_load_result_destroy(WeightLoadResult *result);

const HostTensor *weight_load_result_find(const WeightLoadResult *result, const char *name);

// 骨架 API：读路径 → 填充 out；0 成功，-1 失败。细节（解析）后续实现。
int weight_loader_load_fixture(const char *path, WeightLoadResult *out);

int weight_loader_load_safetensors(const char *path, WeightLoadResult *out);

#ifdef __cplusplus
}
#endif
