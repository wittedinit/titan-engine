#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace titan {
namespace cuda {

// ============================================================================
// RMS Normalization
//
// out[i] = (x[i] / rms) * weight[i]
// where rms = sqrt(mean(x^2) + eps)
//
// Two-pass: first compute sum of squares, then normalize.
// Single-pass online version is possible but two-pass is simpler and
// equally fast for typical hidden dimensions (4096-8192).
// ============================================================================

__global__ void rmsnorm_kernel(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int dim, float eps
) {
    // One block per normalization (one token)
    extern __shared__ float shared[];

    // Pass 1: Sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = input[i];
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    // Final reduction (first warp)
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        sum_sq = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
        if (lane_id == 0) {
            shared[0] = rsqrtf(sum_sq / dim + eps);
        }
    }
    __syncthreads();

    float inv_rms = shared[0];

    // Pass 2: Normalize and scale
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

// ============================================================================
// Layer Normalization (for models that use it)
//
// out[i] = ((x[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]
// ============================================================================

__global__ void layernorm_kernel(
    float*       __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int dim, float eps
) {
    extern __shared__ float shared[];

    // Pass 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += input[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) shared[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        sum = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) shared[0] = sum / dim;
    }
    __syncthreads();
    float mean = shared[0];

    // Pass 2: Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float diff = input[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    }
    if (lane_id == 0) shared[warp_id] = var_sum;
    __syncthreads();
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        var_sum = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
        }
        if (lane_id == 0) shared[0] = rsqrtf(var_sum / dim + eps);
    }
    __syncthreads();
    float inv_std = shared[0];

    // Pass 3: Normalize, scale, shift
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float normalized = (input[i] - mean) * inv_std;
        output[i] = normalized * weight[i] + (bias ? bias[i] : 0.0f);
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void rmsnorm(float* output, const float* input, const float* weight,
             int dim, float eps, cudaStream_t stream) {
    int threads = (dim < 1024) ? dim : 1024;
    int shared_mem = 32 * sizeof(float); // For warp reduction
    rmsnorm_kernel<<<1, threads, shared_mem, stream>>>(
        output, input, weight, dim, eps
    );
    CUDA_CHECK_LAUNCH();
}

void layernorm(float* output, const float* input,
               const float* weight, const float* bias,
               int dim, float eps, cudaStream_t stream) {
    int threads = (dim < 1024) ? dim : 1024;
    int shared_mem = 32 * sizeof(float);
    layernorm_kernel<<<1, threads, shared_mem, stream>>>(
        output, input, weight, bias, dim, eps
    );
    CUDA_CHECK_LAUNCH();
}

} // namespace cuda
} // namespace titan
