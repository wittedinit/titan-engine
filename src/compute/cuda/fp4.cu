#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace titan {
namespace cuda {

// ============================================================================
// FP4 (E2M1) Dequantization + MatVec for Blackwell GPUs (sm_100+)
//
// Blackwell (RTX 5090) has native FP4 Tensor Cores that provide ~2x throughput
// over FP8. FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit.
//
// Values: {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}
// + subnormals and special values
//
// Weight layout: 8 FP4 values packed per uint32 (4 bits each)
// Each group of `group_size` values shares one FP16 scale factor.
// ============================================================================

// FP4 E2M1 to FP32 lookup table
__constant__ float fp4_lut[16] = {
     0.0f,   0.5f,   1.0f,   1.5f,   2.0f,   3.0f,   4.0f,   6.0f,
    -0.0f,  -0.5f,  -1.0f,  -1.5f,  -2.0f,  -3.0f,  -4.0f,  -6.0f
};

__global__ void dequant_matvec_fp4_kernel(
    const uint32_t* __restrict__ weights,    // Packed FP4 [rows, cols/8]
    const half*     __restrict__ scales,     // Per-group scales [rows, cols/group_size]
    const float*    __restrict__ input,      // Input vector [cols]
    float*          __restrict__ output,     // Output vector [rows]
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
    int packed_per_row = cols / 8;
    int groups_per_row = cols / group_size;

    const uint32_t* rw = weights + (size_t)row * packed_per_row;
    const half* rs = scales + (size_t)row * groups_per_row;

    for (int p = threadIdx.x; p < packed_per_row; p += blockDim.x) {
        uint32_t packed = rw[p];
        int base_col = p * 8;
        int group = base_col / group_size;
        float scale = __half2float(rs[group]);

        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int col = base_col + n;
            if (col >= cols) break;
            uint32_t nibble = (packed >> (n * 4)) & 0xF;
            float dequant = fp4_lut[nibble] * scale;
            acc += dequant * shared_input[col];
        }
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
        int nw = (blockDim.x + 31) / 32;
        acc = (lane_id < nw) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (lane_id == 0) output[row] = acc;
    }
}

// ============================================================================
// FP4 Quantization (FP32 → FP4 E2M1 with per-group scaling)
// ============================================================================

__global__ void quantize_fp4_kernel(
    const float* __restrict__ input,     // [numel]
    uint32_t*    __restrict__ output,    // [numel/8]
    half*        __restrict__ scales,    // [numel/group_size]
    int numel, int group_size
) {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = numel / group_size;
    if (group_idx >= num_groups) return;

    int base = group_idx * group_size;

    // Find max absolute value in group for scale computation
    float amax = 0.0f;
    for (int i = 0; i < group_size; i++) {
        float val = fabsf(input[base + i]);
        if (val > amax) amax = val;
    }

    // Scale = amax / 6.0 (max FP4 magnitude)
    float scale = amax / 6.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv_scale = 1.0f / scale;
    scales[group_idx] = __float2half(scale);

    // Quantize each value to nearest FP4
    for (int i = 0; i < group_size; i += 8) {
        uint32_t packed = 0;
        for (int n = 0; n < 8 && (i + n) < group_size; n++) {
            float val = input[base + i + n] * inv_scale;

            // Find nearest FP4 value
            // Positive: 0, 0.5, 1, 1.5, 2, 3, 4, 6
            uint32_t nibble;
            float abs_val = fabsf(val);

            if (abs_val < 0.25f)      nibble = 0;  // 0
            else if (abs_val < 0.75f)  nibble = 1;  // 0.5
            else if (abs_val < 1.25f)  nibble = 2;  // 1.0
            else if (abs_val < 1.75f)  nibble = 3;  // 1.5
            else if (abs_val < 2.5f)   nibble = 4;  // 2.0
            else if (abs_val < 3.5f)   nibble = 5;  // 3.0
            else if (abs_val < 5.0f)   nibble = 6;  // 4.0
            else                       nibble = 7;  // 6.0

            if (val < 0) nibble |= 8; // Set sign bit

            packed |= (nibble << (n * 4));
        }
        output[(base + i) / 8] = packed;
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void dequant_matvec_fp4(
    const void* weights, const void* scales,
    const float* input, float* output,
    int rows, int cols, int group_size, cudaStream_t stream
) {
    int block_size = 256;
    int shared_mem = cols * sizeof(float);
    dequant_matvec_fp4_kernel<<<rows, block_size, shared_mem, stream>>>(
        (const uint32_t*)weights, (const half*)scales,
        input, output, rows, cols, group_size
    );
}

void quantize_fp4(
    const float* input, void* output, void* scales,
    int numel, int group_size, cudaStream_t stream
) {
    int num_groups = numel / group_size;
    int threads = 256;
    int blocks = (num_groups + threads - 1) / threads;
    quantize_fp4_kernel<<<blocks, threads, 0, stream>>>(
        input, (uint32_t*)output, (half*)scales,
        numel, group_size
    );
}

} // namespace cuda
} // namespace titan
