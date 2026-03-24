#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

namespace titan {
namespace cuda {

// ============================================================================
// MoE Routing Gate: hidden_state @ gate_weight -> logits -> softmax -> topK
// ============================================================================

// Step 1: Gate projection (hidden_state × gate_weight^T → expert logits)
__global__ void moe_gate_kernel(
    const float* __restrict__ hidden,       // [hidden_dim]
    const float* __restrict__ gate_weight,  // [num_experts, hidden_dim]
    float*       __restrict__ logits,       // [num_experts]
    int hidden_dim, int num_experts
) {
    int expert = blockIdx.x;
    if (expert >= num_experts) return;

    extern __shared__ float shared_hidden[];
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        shared_hidden[i] = hidden[i];
    }
    __syncthreads();

    float dot = 0.0f;
    const float* w = gate_weight + (size_t)expert * hidden_dim;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        dot += w[i] * shared_hidden[i];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
    }
    __shared__ float warp_sums[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) warp_sums[warp_id] = dot;
    __syncthreads();
    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        dot = (lane_id < nw) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        }
        if (lane_id == 0) logits[expert] = dot;
    }
}

// Step 2: Softmax + Top-K selection
// This runs on a single thread block since num_experts is typically small (8-512)
__global__ void moe_topk_kernel(
    const float* __restrict__ logits,       // [num_experts]
    float*       __restrict__ weights,      // [k] — output routing weights (softmax'd)
    int*         __restrict__ indices,      // [k] — output expert indices
    int num_experts, int k
) {
    extern __shared__ float shared[];
    float* slogits = shared;           // [num_experts]
    float* sweights = shared + num_experts;  // [k]

    // Load logits
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        slogits[i] = logits[i];
    }
    __syncthreads();

    if (threadIdx.x > 0) return; // Single thread for small K selection

    // Softmax: find max
    float max_val = -FLT_MAX;
    for (int i = 0; i < num_experts; i++) {
        max_val = fmaxf(max_val, slogits[i]);
    }

    // Exp and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < num_experts; i++) {
        slogits[i] = expf(slogits[i] - max_val);
        sum_exp += slogits[i];
    }

    // Normalize
    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    for (int i = 0; i < num_experts; i++) {
        slogits[i] *= inv_sum;
    }

    // Top-K selection (simple selection sort for small K)
    for (int j = 0; j < k; j++) {
        float best_val = -1.0f;
        int best_idx = -1;
        for (int i = 0; i < num_experts; i++) {
            if (slogits[i] > best_val) {
                best_val = slogits[i];
                best_idx = i;
            }
        }
        weights[j] = best_val;
        indices[j] = best_idx;
        slogits[best_idx] = -2.0f; // Mark as used
    }

    // Renormalize selected weights to sum to 1
    float sel_sum = 0.0f;
    for (int j = 0; j < k; j++) sel_sum += weights[j];
    float inv_sel = 1.0f / (sel_sum + 1e-8f);
    for (int j = 0; j < k; j++) weights[j] *= inv_sel;
}

// ============================================================================
// Expert Forward: SwiGLU MLP (gate + up + down projections)
//
// This kernel handles a single expert. Multiple experts are launched
// as separate kernel instances on different streams for parallelism.
// For INT4 experts, use dequant_matvec_int4 instead.
// ============================================================================

__global__ void expert_forward_fp32_kernel(
    const float* __restrict__ input,       // [hidden_dim]
    const float* __restrict__ gate_proj,   // [inter_dim, hidden_dim]
    const float* __restrict__ up_proj,     // [inter_dim, hidden_dim]
    const float* __restrict__ down_proj,   // [hidden_dim, inter_dim]
    float*       __restrict__ output,      // [hidden_dim]
    float*       __restrict__ scratch,     // [inter_dim * 2] — for gate and up outputs
    int hidden_dim, int inter_dim
) {
    // This is a reference implementation.
    // In practice, use cuBLAS or the dequant kernels for quantized weights.

    extern __shared__ float shared_input[];
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        shared_input[i] = input[i];
    }
    __syncthreads();

    float* gate_out = scratch;
    float* up_out = scratch + inter_dim;

    // Gate projection: gate_out = gate_proj @ input
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < inter_dim;
         row += gridDim.x * blockDim.x) {
        float dot = 0.0f;
        for (int col = 0; col < hidden_dim; col++) {
            dot += gate_proj[row * hidden_dim + col] * shared_input[col];
        }
        gate_out[row] = dot;
    }
    __syncthreads();

    // Up projection: up_out = up_proj @ input
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < inter_dim;
         row += gridDim.x * blockDim.x) {
        float dot = 0.0f;
        for (int col = 0; col < hidden_dim; col++) {
            dot += up_proj[row * hidden_dim + col] * shared_input[col];
        }
        up_out[row] = dot;
    }
    __syncthreads();

    // SwiGLU activation: act_out = SiLU(gate_out) * up_out
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inter_dim;
         i += gridDim.x * blockDim.x) {
        float g = gate_out[i];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        gate_out[i] = (g * sigmoid_g) * up_out[i]; // Reuse gate_out as scratch
    }
    __syncthreads();

    // Down projection: output = down_proj @ act_out
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < hidden_dim;
         row += gridDim.x * blockDim.x) {
        float dot = 0.0f;
        for (int col = 0; col < inter_dim; col++) {
            dot += down_proj[row * inter_dim + col] * gate_out[col];
        }
        output[row] = dot;
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void moe_gate(const float* hidden, const float* gate_weight, float* logits,
              int hidden_dim, int num_experts, cudaStream_t stream) {
    int threads = 256;
    int shared_mem = hidden_dim * sizeof(float);
    moe_gate_kernel<<<num_experts, threads, shared_mem, stream>>>(
        hidden, gate_weight, logits, hidden_dim, num_experts
    );
}

void moe_topk(const float* logits, float* weights, int* indices,
              int num_experts, int k, cudaStream_t stream) {
    int shared_mem = (num_experts + k) * sizeof(float);
    moe_topk_kernel<<<1, 32, shared_mem, stream>>>(
        logits, weights, indices, num_experts, k
    );
}

} // namespace cuda
} // namespace titan
