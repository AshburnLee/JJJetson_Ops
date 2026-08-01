#include "weight_loader.h"

#include <stdlib.h>
#include <string.h>

void weight_load_result_init(WeightLoadResult *out) {
    if (out == nullptr) {
        return;
    }
    memset(out, 0, sizeof(*out));
}

static void host_tensor_free(HostTensor *tensor) {
    if (tensor == nullptr) {
        return;
    }
    free(tensor->name);
    free(tensor->data);
    free(tensor->dims);
    tensor->name = nullptr;
    tensor->data = nullptr;
    tensor->dims = nullptr;
    tensor->ndim = 0;
}

void weight_load_result_destroy(WeightLoadResult *result) {
    if (result == nullptr) {
        return;
    }
    if (result->tensors != nullptr) {
        for (int i = 0; i < result->num_tensors; ++i) {
            host_tensor_free(&result->tensors[i]);
        }
        free(result->tensors);
    }
    weight_load_result_init(result);
}

const HostTensor *weight_load_result_find(const WeightLoadResult *result, const char *name) {
    if (result == nullptr || name == nullptr || result->tensors == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < result->num_tensors; ++i) {
        const HostTensor *tensor = &result->tensors[i];
        if (tensor->name != nullptr && strcmp(tensor->name, name) == 0) {
            return tensor;
        }
    }
    return nullptr;
}

int weight_loader_load_fixture(const char *path, WeightLoadResult *out) {
    if (path == nullptr || out == nullptr) {
        return -1;
    }
    (void)path;
    // Skeleton: fixture 解析在细节阶段实现。
    return -1;
}

int weight_loader_load_safetensors(const char *path, WeightLoadResult *out) {
    if (path == nullptr || out == nullptr) {
        return -1;
    }
    (void)path;
    // Skeleton: safetensors 解析在细节阶段实现。
    return -1;
}
