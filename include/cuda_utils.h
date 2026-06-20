#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_
// for Engine（host TU 翻译单元）
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t error = call;                                                                  \
        if (error != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,                      \
                    cudaGetErrorString(error));                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define LAUNCH_CHECK()                                                                             \
    do {                                                                                           \
        cudaError_t error = cudaGetLastError();                                                    \
        if (error != cudaSuccess) {                                                                \
            fprintf(stderr, "Kernel launch failed at %s:%d - %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(error));                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#endif // CUDA_UTILS_H_
