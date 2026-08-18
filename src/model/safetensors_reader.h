#pragma once

#include "weight_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

// @see safetensors_reader.cpp — safetensors_read_file 块注释
int safetensors_read_file(const char *path, HostTensor **out_tensors, int *out_num_tensors);

#ifdef __cplusplus
}
#endif
