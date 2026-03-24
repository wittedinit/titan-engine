#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace titan {
namespace cuda {

// ============================================================================
// SwiGLU Activation — Fused gate * up * sigmoid
//
// SwiGLU(x) = (x * sigmoid(gate)) * up
// gate_out and up_out are results of two separate linear projections.
// This kernel fuses the activation to avoid extra memory reads/writes.
// ============================================================================

__global__ void swiglu_kernel(
    float*       __restrict__ output,      // [dim]
    const float* __restrict__ gate_out,    // [dim] — gate projection output
    const float* __restrict__ up_out,      // [dim] — up projection output
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float gate = gate_out[i];
    float up = up_out[i];

    // SiLU(gate) = gate * sigmoid(gate)
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    output[i] = (gate * sigmoid_gate) * up;
}

// ============================================================================
// GELU Activation
// ============================================================================

__global__ void gelu_kernel(
    float*       __restrict__ output,
    const float* __restrict__ input,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float x = input[i];
    // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3); // sqrt(2/pi) ≈ 0.7978845608
    output[i] = 0.5f * x * (1.0f + tanhf(inner));
}

// ============================================================================
// Fused Add + RMSNorm (residual connection + normalization)
//
// Combines: residual = residual + hidden
//           output = rmsnorm(residual, weight)
// Saves one global memory pass.
// ============================================================================

__global__ void fused_add_rmsnorm_kernel(
    float*       __restrict__ output,      // [dim] — normalized result
    float*       __restrict__ residual,    // [dim] — residual (modified in-place)
    const float* __restrict__ hidden,      // [dim] — value to add
    const float* __restrict__ weight,      // [dim] — normalization weight
    int dim, float eps
) {
    extern __shared__ float shared[];

    // Step 1: residual += hidden, and compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = residual[i] + hidden[i];
        residual[i] = val;  // Update residual in-place
        sum_sq += val * val;
    }

    // Reduce sum_sq
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        sum_sq = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
        if (lane_id == 0) shared[0] = rsqrtf(sum_sq / dim + eps);
    }
    __syncthreads();
    float inv_rms = shared[0];

    // Step 2: Normalize
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[i] = residual[i] * inv_rms * weight[i];
    }
}

// ============================================================================
// Fused MoE Expert Combine + Residual + Norm
//
// Combines multiple expert outputs with routing weights, adds residual,
// and applies RMSNorm. This is the critical post-expert fusion kernel
// (inspired by flash-moe's CMD3 shader).
// ============================================================================

__global__ void fused_moe_combine_norm_kernel(
    float*       __restrict__ output,           // [dim] — final normalized output
    float*       __restrict__ residual,         // [dim] — residual (modified in-place)
    const float* __restrict__ expert_outputs,   // [num_active, dim]
    const float* __restrict__ routing_weights,  // [num_active]
    const float* __restrict__ shared_expert,    // [dim] or nullptr
    float        shared_weight,                 // Weight for shared expert (if present)
    const float* __restrict__ norm_weight,      // [dim]
    int dim, int num_active, float eps
) {
    extern __shared__ float shared[];

    // Step 1: Weighted sum of expert outputs + shared expert + residual
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float combined = 0.0f;

        // Sum active expert outputs with routing weights
        for (int e = 0; e < num_active; e++) {
            combined += expert_outputs[e * dim + i] * routing_weights[e];
        }

        // Add shared expert if present
        if (shared_expert) {
            combined += shared_expert[i] * shared_weight;
        }

        // Residual connection
        float val = residual[i] + combined;
        residual[i] = val;
        sum_sq += val * val;
    }

    // Reduce and normalize (same pattern as rmsnorm)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        sum_sq = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
        if (lane_id == 0) shared[0] = rsqrtf(sum_sq / dim + eps);
    }
    __syncthreads();
    float inv_rms = shared[0];

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[i] = residual[i] * inv_rms * norm_weight[i];
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void swiglu(float* output, const float* gate, const float* up,
            int dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(output, gate, up, dim);
}

void gelu(float* output, const float* input, int dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(output, input, dim);
}

void fused_add_rmsnorm(float* output, float* residual, const float* hidden,
                        const float* weight, int dim, float eps,
                        cudaStream_t stream) {
    int threads = (dim < 1024) ? dim : 1024;
    int shared_mem = 32 * sizeof(float);
    fused_add_rmsnorm_kernel<<<1, threads, shared_mem, stream>>>(
        output, residual, hidden, weight, dim, eps
    );
}

void fused_moe_combine_norm(
    float* output, float* residual,
    const float* expert_outputs, const float* routing_weights,
    const float* shared_expert, float shared_weight,
    const float* norm_weight,
    int dim, int num_active, float eps, cudaStream_t stream
) {
    int threads = (dim < 1024) ? dim : 1024;
    int shared_mem = 32 * sizeof(float);
    fused_moe_combine_norm_kernel<<<1, threads, shared_mem, stream>>>(
        output, residual, expert_outputs, routing_weights,
        shared_expert, shared_weight, norm_weight,
        dim, num_active, eps
    );
}

} // namespace cuda
} // namespace titan
