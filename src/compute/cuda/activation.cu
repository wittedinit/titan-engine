#include "cuda_check.h"
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
    CUDA_CHECK_LAUNCH();
}

void gelu(float* output, const float* input, int dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(output, input, dim);
    CUDA_CHECK_LAUNCH();
}

void fused_add_rmsnorm(float* output, float* residual, const float* hidden,
                        const float* weight, int dim, float eps,
                        cudaStream_t stream) {
    int threads = (dim < 1024) ? dim : 1024;
    int shared_mem = 32 * sizeof(float);
    fused_add_rmsnorm_kernel<<<1, threads, shared_mem, stream>>>(
        output, residual, hidden, weight, dim, eps
    );
    CUDA_CHECK_LAUNCH();
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
    CUDA_CHECK_LAUNCH();
}

// ============================================================================
// MLA Helpers — Multi-head Latent Attention (DeepSeek V3 / Kimi K2 style)
// ============================================================================

// Split interleaved kv_b_proj output [n_heads*(nope_hd+v_hd)] into
// separate k_nope [n_heads*nope_hd] and v [n_heads*v_hd] buffers.
// kv_expanded layout: [head0_k_nope(nope_hd) | head0_v(v_hd) | head1_k_nope | ...]
__global__ void mla_deinterleave_kv_kernel(
    const float* __restrict__ kv_expanded,  // [n_heads * (nope_hd + v_hd)]
    float*       __restrict__ k_nope,       // [n_heads * nope_hd]
    float*       __restrict__ v,            // [n_heads * v_hd]
    int n_heads, int nope_hd, int v_hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = nope_hd + v_hd;
    int total = n_heads * stride;
    if (idx >= total) return;

    int head = idx / stride;
    int off  = idx % stride;

    if (off < nope_hd) {
        k_nope[head * nope_hd + off] = kv_expanded[idx];
    } else {
        v[head * v_hd + (off - nope_hd)] = kv_expanded[idx];
    }
}

// Assemble full K [n_heads*(nope_hd+rope_hd)] from k_nope and a shared k_rope vector.
// k_rope is the same for all heads (broadcast).
__global__ void mla_assemble_k_kernel(
    const float* __restrict__ k_nope,    // [n_heads * nope_hd]
    const float* __restrict__ k_rope_dec,// [rope_hd] — shared across all heads
    float*       __restrict__ k_out,     // [n_heads * (nope_hd + rope_hd)]
    int n_heads, int nope_hd, int rope_hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_dim = nope_hd + rope_hd;
    int total = n_heads * head_dim;
    if (idx >= total) return;

    int head = idx / head_dim;
    int off  = idx % head_dim;

    if (off < nope_hd) {
        k_out[idx] = k_nope[head * nope_hd + off];
    } else {
        k_out[idx] = k_rope_dec[off - nope_hd];
    }
}

// Extract the nope portion of MLA Q for attention (drops rope_hd dims per head).
// Attention only needs the full Q for the RoPE dot product; for the nope-only
// dot product we'd normally need to split per head. This helper keeps the
// first nope_hd dims of each head and discards rope_hd dims.
__global__ void mla_extract_q_nope_kernel(
    const float* __restrict__ q_full,    // [n_heads * (nope_hd + rope_hd)]
    float*       __restrict__ q_nope,    // [n_heads * nope_hd]
    int n_heads, int nope_hd, int rope_hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * nope_hd) return;
    int head = idx / nope_hd;
    int off  = idx % nope_hd;
    q_nope[idx] = q_full[head * (nope_hd + rope_hd) + off];
}

void mla_deinterleave_kv(
    const float* kv_expanded, float* k_nope, float* v,
    int n_heads, int nope_hd, int v_hd, cudaStream_t stream
) {
    int total = n_heads * (nope_hd + v_hd);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mla_deinterleave_kv_kernel<<<blocks, threads, 0, stream>>>(
        kv_expanded, k_nope, v, n_heads, nope_hd, v_hd
    );
    CUDA_CHECK_LAUNCH();
}

void mla_assemble_k(
    const float* k_nope, const float* k_rope_dec, float* k_out,
    int n_heads, int nope_hd, int rope_hd, cudaStream_t stream
) {
    int total = n_heads * (nope_hd + rope_hd);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mla_assemble_k_kernel<<<blocks, threads, 0, stream>>>(
        k_nope, k_rope_dec, k_out, n_heads, nope_hd, rope_hd
    );
    CUDA_CHECK_LAUNCH();
}

void mla_extract_q_nope(
    const float* q_full, float* q_nope,
    int n_heads, int nope_hd, int rope_hd, cudaStream_t stream
) {
    int total = n_heads * nope_hd;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mla_extract_q_nope_kernel<<<blocks, threads, 0, stream>>>(
        q_full, q_nope, n_heads, nope_hd, rope_hd
    );
    CUDA_CHECK_LAUNCH();
}

} // namespace cuda
} // namespace titan
