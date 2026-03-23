#include "core/types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace titan {
namespace cuda {

// ============================================================================
// INT4 Affine Dequantization + Matrix-Vector Multiply
//
// Inspired by flash-moe's FMA-optimized kernel but adapted for CUDA.
// Each group of `group_size` elements shares one scale and one bias.
// weight_val = (uint4_nibble * scale + bias)
// output[row] = sum_col( weight[row][col] * input[col] )
//
// Optimized with FMA trick from flash-moe:
//   acc += fma(float(nibble), scale * x, bias * x)
//   instead of: acc += (float(nibble) * scale + bias) * x
// ============================================================================

__global__ void dequant_matvec_int4_kernel(
    const uint32_t* __restrict__ weights,    // Packed INT4 weights [rows, cols/8]
    const half*     __restrict__ scales,     // Per-group scales [rows, cols/group_size]
    const half*     __restrict__ biases,     // Per-group biases [rows, cols/group_size]
    const float*    __restrict__ input,      // Input vector [cols]
    float*          __restrict__ output,     // Output vector [rows]
    int rows, int cols, int group_size
) {
    // Each block handles one row (or a few rows with tiling)
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared_input[];

    // Load input into shared memory (all threads cooperate)
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        shared_input[i] = input[i];
    }
    __syncthreads();

    // Each thread accumulates a partial sum over a subset of columns
    float acc = 0.0f;

    int groups_per_row = cols / group_size;
    int packed_per_row = cols / 8; // 8 nibbles per uint32

    const uint32_t* row_weights = weights + (size_t)row * packed_per_row;
    const half* row_scales = scales + (size_t)row * groups_per_row;
    const half* row_biases = biases + (size_t)row * groups_per_row;

    for (int packed_idx = threadIdx.x; packed_idx < packed_per_row; packed_idx += blockDim.x) {
        uint32_t packed = row_weights[packed_idx];
        int base_col = packed_idx * 8;

        // Determine which group this belongs to
        int group = base_col / group_size;
        float scale = __half2float(row_scales[group]);
        float bias = __half2float(row_biases[group]);

        // Process 8 nibbles from the packed uint32
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int col = base_col + n;
            if (col >= cols) break;

            uint32_t nibble = (packed >> (n * 4)) & 0xF;
            float x = shared_input[col];

            // FMA-optimized dequant: fma(nibble, scale*x, bias*x)
            float sx = scale * x;
            float bx = bias * x;
            acc = fmaf((float)nibble, sx, acc + bx);
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // First thread in each warp writes to shared memory
    __shared__ float warp_sums[32]; // Max 32 warps per block
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    // Final reduction across warps (first warp only)
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        acc = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (lane_id == 0) {
            output[row] = acc;
        }
    }
}

// ============================================================================
// INT2 Dequant + MatVec (extreme compression)
// ============================================================================

__global__ void dequant_matvec_int2_kernel(
    const uint32_t* __restrict__ weights,    // Packed INT2 weights [rows, cols/16]
    const half*     __restrict__ scales,
    const half*     __restrict__ biases,
    const float*    __restrict__ input,
    float*          __restrict__ output,
    int rows, int cols, int group_size
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared_input[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        shared_input[i] = input[i];
    }
    __syncthreads();

    float acc = 0.0f;

    int groups_per_row = cols / group_size;
    int packed_per_row = cols / 16; // 16 x 2-bit values per uint32

    const uint32_t* row_weights = weights + (size_t)row * packed_per_row;
    const half* row_scales = scales + (size_t)row * groups_per_row;
    const half* row_biases = biases + (size_t)row * groups_per_row;

    for (int packed_idx = threadIdx.x; packed_idx < packed_per_row; packed_idx += blockDim.x) {
        uint32_t packed = row_weights[packed_idx];
        int base_col = packed_idx * 16;
        int group = base_col / group_size;
        float scale = __half2float(row_scales[group]);
        float bias = __half2float(row_biases[group]);

        #pragma unroll
        for (int n = 0; n < 16; n++) {
            int col = base_col + n;
            if (col >= cols) break;
            uint32_t val = (packed >> (n * 2)) & 0x3;
            float x = shared_input[col];
            float sx = scale * x;
            float bx = bias * x;
            acc = fmaf((float)val, sx, acc + bx);
        }
    }

    // Warp reduction (same as INT4)
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    __shared__ float warp_sums[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_sums[warp_id] = acc;
    __syncthreads();
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        acc = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (lane_id == 0) output[row] = acc;
    }
}

// ============================================================================
// FP8 MatVec (for Ada Lovelace / Blackwell GPUs)
// ============================================================================

__global__ void matvec_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ weights,  // FP8 weights [rows, cols]
    const float*         __restrict__ input,     // FP32 input [cols]
    float*               __restrict__ output,    // FP32 output [rows]
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared_input[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        shared_input[i] = input[i];
    }
    __syncthreads();

    float acc = 0.0f;
    const __nv_fp8_e4m3* row_w = weights + (size_t)row * cols;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float w = (float)row_w[col];
        acc += w * shared_input[col];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    __shared__ float warp_sums[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_sums[warp_id] = acc;
    __syncthreads();
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        acc = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (lane_id == 0) output[row] = acc;
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void dequant_matvec_int4(
    const void* weights, const void* scales, const void* biases,
    const float* input, float* output,
    int rows, int cols, int group_size, cudaStream_t stream
) {
    int block_size = 256;
    int shared_mem = cols * sizeof(float);

    dequant_matvec_int4_kernel<<<rows, block_size, shared_mem, stream>>>(
        (const uint32_t*)weights, (const half*)scales, (const half*)biases,
        input, output, rows, cols, group_size
    );
}

void dequant_matvec_int2(
    const void* weights, const void* scales, const void* biases,
    const float* input, float* output,
    int rows, int cols, int group_size, cudaStream_t stream
) {
    int block_size = 256;
    int shared_mem = cols * sizeof(float);

    dequant_matvec_int2_kernel<<<rows, block_size, shared_mem, stream>>>(
        (const uint32_t*)weights, (const half*)scales, (const half*)biases,
        input, output, rows, cols, group_size
    );
}

} // namespace cuda
} // namespace titan
