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

// path 是一个目录。打开其中的 config.txt（例如 hidden_size=128），写入 out->config；
// 再按 manifest.txt 逐行找 .f32 文件，每行形如 [embed 2 128 256 weights/embed.f32]，
// 把 float 读进 out->tensors[]。目录不对或缺文件则返回 -1。
int weight_loader_load_fixture(const char *path, WeightLoadResult *out);

// path 是一个 .safetensors 文件。读 8 字节头长 + JSON，把每个 F32 tensor 拷到 out->tensors[]
// （名字用 JSON 里的 key，例如 embed、layer0.w_q，不会改成 HF 原名）。
// 若同目录还有 config.txt，顺带写入 out->config；没有则 config 保持全 0。
// safetensors 是真实模型权重的目标格式；当前完成了可读 safetensors + 可验证（步骤 1/2，单测为主）。
// 真实推理场景需从 HF safetensors + config 加载实际权重（步骤 3/4 名映射后接 Model H2D；见 roadmap
// 模块 1）。
int weight_loader_load_safetensors(const char *path, WeightLoadResult *out);

#ifdef __cplusplus
}
#endif
