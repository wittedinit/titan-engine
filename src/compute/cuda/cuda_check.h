#pragma once

#include <cuda_runtime.h>
#include <cstdio>

// Debug-mode kernel launch error check.
// Place after every kernel launch in wrapper functions.
#ifndef NDEBUG
#define CUDA_CHECK_LAUNCH() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA launch error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)
#else
#define CUDA_CHECK_LAUNCH() ((void)0)
#endif
