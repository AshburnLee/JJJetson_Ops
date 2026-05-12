#ifndef CUBLAS_UTILS_CUH_
#define CUBLAS_UTILS_CUH_

#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t state = call;                                                               \
        if (state != CUBLAS_STATUS_SUCCESS) {                                                      \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, (int)state);       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#endif // CUBLAS_UTILS_CUH_
