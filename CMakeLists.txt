cmake_minimum_required(VERSION 3.24)
project(titan-engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# CUDA architecture — default to sm_90a (Blackwell) + sm_89 (Ada) + sm_80 (Ampere)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;89;90a")
endif()

# Options
option(TITAN_BUILD_TESTS "Build tests" ON)
option(TITAN_BUILD_TOOLS "Build Python tools" ON)
option(TITAN_BUILD_SERVER "Build HTTP API server" ON)
option(TITAN_USE_IO_URING "Use io_uring for NVMe I/O" ON)
option(TITAN_USE_AVX512 "Use AVX-512 CPU kernels" ON)

# Check for io_uring
if(TITAN_USE_IO_URING)
    find_library(URING_LIB uring)
    if(URING_LIB)
        add_definitions(-DTITAN_HAS_IO_URING)
        message(STATUS "io_uring: found (${URING_LIB})")
    else()
        message(WARNING "io_uring not found, falling back to pread")
        set(TITAN_USE_IO_URING OFF)
    endif()
endif()

# Check for AVX-512
if(TITAN_USE_AVX512)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" HAS_AVX512)
    if(HAS_AVX512)
        add_definitions(-DTITAN_HAS_AVX512)
        message(STATUS "AVX-512: supported")
    else()
        message(WARNING "AVX-512 not supported on this CPU")
        set(TITAN_USE_AVX512 OFF)
    endif()
endif()

# Core library
add_library(titan_core STATIC
    src/core/types.cpp
    src/core/config.cpp
    src/core/hardware.cpp
    src/core/logging.cpp
)
target_include_directories(titan_core PUBLIC src)
target_link_libraries(titan_core PUBLIC CUDA::cudart pthread)

# Memory manager
add_library(titan_memory STATIC
    src/memory/memory_manager.cpp
    src/memory/vram_pool.cpp
    src/memory/ram_pool.cpp
    src/memory/nvme_pool.cpp
    src/memory/prefetcher.cpp
)
target_include_directories(titan_memory PUBLIC src)
target_link_libraries(titan_memory PUBLIC titan_core)
if(TITAN_USE_IO_URING)
    target_link_libraries(titan_memory PUBLIC ${URING_LIB})
endif()

# CUDA compute kernels
add_library(titan_compute STATIC
    src/compute/cuda/dequant.cu
    src/compute/cuda/attention.cu
    src/compute/cuda/moe.cu
    src/compute/cuda/norm.cu
    src/compute/cuda/activation.cu
    src/compute/cuda/sampling.cu
    src/compute/cuda/sparse.cu
    src/compute/cuda/gemv.cu
    src/compute/cpu/matmul_avx.cpp
    src/compute/cpu/expert_cpu.cpp
)
target_include_directories(titan_compute PUBLIC src)
target_link_libraries(titan_compute PUBLIC titan_core CUDA::cublas CUDA::cublasLt)
if(TITAN_USE_AVX512)
    target_compile_options(titan_compute PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx512f -mavx512bw -mavx512vl>)
endif()

# Model runtime
add_library(titan_model STATIC
    src/model/loader.cpp
    src/model/quantizer.cpp
    src/model/dense.cpp
    src/model/moe.cpp
    src/model/tokenizer.cpp
    src/model/sparsity.cpp
)
target_include_directories(titan_model PUBLIC src)
target_link_libraries(titan_model PUBLIC titan_core titan_compute titan_memory)

# Inference engine
add_library(titan_inference STATIC
    src/inference/engine.cpp
    src/inference/scheduler.cpp
    src/inference/speculative.cpp
    src/inference/kv_cache.cpp
    src/inference/batch.cpp
)
target_include_directories(titan_inference PUBLIC src)
target_link_libraries(titan_inference PUBLIC titan_model titan_compute titan_memory)

# CLI executable
add_executable(titan
    src/api/cli.cpp
)
target_link_libraries(titan PRIVATE titan_inference titan_model titan_compute titan_memory titan_core)

# Tests
if(TITAN_BUILD_TESTS)
    enable_testing()

    add_executable(test_types tests/test_types.cpp)
    target_link_libraries(test_types PRIVATE titan_core)
    add_test(NAME test_types COMMAND test_types)

    add_executable(test_memory tests/test_memory.cpp)
    target_link_libraries(test_memory PRIVATE titan_memory titan_core)
    add_test(NAME test_memory COMMAND test_memory)

    add_executable(test_kernels tests/test_kernels.cu)
    target_link_libraries(test_kernels PRIVATE titan_compute titan_core CUDA::cudart)
    add_test(NAME test_kernels COMMAND test_kernels)
endif()
