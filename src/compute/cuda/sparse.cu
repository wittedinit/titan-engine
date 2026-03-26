#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace titan {
namespace cuda {

// ============================================================================
// Activation Sparsity — Skip near-zero neurons for dense model speedup
//
// PowerInfer insight: In most dense LLMs, 80-90% of neurons in FFN layers
// activate near-zero for any given token. By profiling which neurons are
// "hot" (frequently activated) vs "cold" (rarely activated), we can:
//
// 1. Keep hot neurons on GPU (always computed)
// 2. Skip cold neurons entirely (or compute on CPU only when needed)
// 3. Achieve 2-3x speedup without any model modification
//
// This is NOT MoE — the model structure is unchanged. We just skip work.
// ============================================================================

// ---------------------------------------------------------------------------
// Activation Magnitude Profiler
//
// During a profiling pass, record the magnitude of each neuron's activation
// across N sample tokens. Neurons with consistently low magnitude are "cold".
// ---------------------------------------------------------------------------

__global__ void profile_activation_magnitudes_kernel(
    const float* __restrict__ activations,  // [batch_tokens, inter_dim]
    float*       __restrict__ magnitude_sum, // [inter_dim] — accumulated |activation|
    int*         __restrict__ nonzero_count, // [inter_dim] — count of |act| > threshold
    int inter_dim, int batch_tokens, float threshold
) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= inter_dim) return;

    float sum = 0.0f;
    int count = 0;

    for (int t = 0; t < batch_tokens; t++) {
        float val = fabsf(activations[t * inter_dim + neuron]);
        sum += val;
        if (val > threshold) count++;
    }

    atomicAdd(&magnitude_sum[neuron], sum);
    atomicAdd(&nonzero_count[neuron], count);
}

// ---------------------------------------------------------------------------
// Sparse Activation Predictor
//
// Given the current hidden state, predict which neurons will be "hot" using
// a lightweight linear projection (hidden_dim → inter_dim sigmoid scores).
// Only compute neurons with predicted score > threshold.
// ---------------------------------------------------------------------------

__global__ void predict_active_neurons_kernel(
    const float* __restrict__ hidden,           // [hidden_dim]
    const float* __restrict__ predictor_weight, // [inter_dim, hidden_dim]
    const float* __restrict__ predictor_bias,   // [inter_dim]
    int*         __restrict__ active_indices,    // [max_active] — output indices
    int*         __restrict__ num_active,        // [1] — output count
    int hidden_dim, int inter_dim, int max_active, float threshold
) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= inter_dim) return;

    // Compute predictor score: sigmoid(w @ hidden + b)
    float dot = 0.0f;
    const float* w = predictor_weight + (size_t)neuron * hidden_dim;
    for (int i = 0; i < hidden_dim; i++) {
        dot += w[i] * hidden[i];
    }
    dot += predictor_bias[neuron];
    float score = 1.0f / (1.0f + expf(-dot));

    // If predicted active, add to output list
    if (score > threshold) {
        int idx = atomicAdd(num_active, 1);
        if (idx < max_active) {
            active_indices[idx] = neuron;
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse MatVec — Only compute rows corresponding to active neurons
//
// Instead of computing output = W @ input for all rows,
// only compute rows in active_indices. For 80% sparsity, this is ~5x faster.
// ---------------------------------------------------------------------------

__global__ void sparse_matvec_fp32_kernel(
    const float* __restrict__ weight,          // [total_rows, cols]
    const float* __restrict__ input,           // [cols]
    float*       __restrict__ output,          // [total_rows] — sparse, only active filled
    const int*   __restrict__ active_indices,  // [num_active]
    int num_active, int cols
) {
    int idx = blockIdx.x;
    if (idx >= num_active) return;

    int row = active_indices[idx];

    extern __shared__ float shared_input[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        shared_input[i] = input[i];
    }
    __syncthreads();

    float acc = 0.0f;
    const float* w = weight + (size_t)row * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        acc += w[col] * shared_input[col];
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

// Sparse INT4 dequant matvec — same idea but with quantized weights
__global__ void sparse_dequant_matvec_int4_kernel(
    const uint32_t* __restrict__ weights,
    const half*     __restrict__ scales,
    const half*     __restrict__ biases,
    const float*    __restrict__ input,
    float*          __restrict__ output,
    const int*      __restrict__ active_indices,
    int num_active, int cols, int group_size
) {
    int idx = blockIdx.x;
    if (idx >= num_active) return;

    int row = active_indices[idx];
    int packed_per_row = cols / 8;
    int groups_per_row = cols / group_size;

    extern __shared__ float shared_input[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        shared_input[i] = input[i];
    }
    __syncthreads();

    float acc = 0.0f;
    const uint32_t* rw = weights + (size_t)row * packed_per_row;
    const half* rs = scales + (size_t)row * groups_per_row;
    const half* rb = biases + (size_t)row * groups_per_row;

    for (int p = threadIdx.x; p < packed_per_row; p += blockDim.x) {
        uint32_t packed = rw[p];
        int base_col = p * 8;
        int group = base_col / group_size;
        float scale = __half2float(rs[group]);
        float bias = __half2float(rb[group]);

        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int col = base_col + n;
            if (col >= cols) break;
            uint32_t nibble = (packed >> (n * 4)) & 0xF;
            float x = shared_input[col];
            float sx = scale * x;
            float bx = bias * x;
            acc = fmaf((float)nibble, sx, acc + bx);
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

// ---------------------------------------------------------------------------
// Sparse SwiGLU — Only compute activation for active neurons
// ---------------------------------------------------------------------------

__global__ void sparse_swiglu_kernel(
    float*       __restrict__ output,
    const float* __restrict__ gate_out,
    const float* __restrict__ up_out,
    const int*   __restrict__ active_indices,
    int num_active
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active) return;

    int neuron = active_indices[idx];
    float gate = gate_out[neuron];
    float up = up_out[neuron];
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    output[neuron] = (gate * sigmoid_gate) * up;
}

// ---------------------------------------------------------------------------
// Scatter-Gather for sparse down projection
//
// After sparse SwiGLU, only active neurons have values.
// The down projection needs to gather only those active rows from down_proj.
// ---------------------------------------------------------------------------

__global__ void sparse_down_proj_kernel(
    const float* __restrict__ sparse_activation, // [inter_dim] — sparse, only active filled
    const float* __restrict__ down_weight,       // [hidden_dim, inter_dim]
    float*       __restrict__ output,            // [hidden_dim]
    const int*   __restrict__ active_indices,
    int num_active, int hidden_dim, int inter_dim
) {
    int row = blockIdx.x;
    if (row >= hidden_dim) return;

    float acc = 0.0f;
    const float* w = down_weight + (size_t)row * inter_dim;

    // Only accumulate from active neurons (gather)
    for (int i = threadIdx.x; i < num_active; i += blockDim.x) {
        int col = active_indices[i];
        acc += w[col] * sparse_activation[col];
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
// Launch Wrappers
// ============================================================================

void profile_activations(
    const float* activations, float* magnitude_sum, int* nonzero_count,
    int inter_dim, int batch_tokens, float threshold, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (inter_dim + threads - 1) / threads;
    profile_activation_magnitudes_kernel<<<blocks, threads, 0, stream>>>(
        activations, magnitude_sum, nonzero_count,
        inter_dim, batch_tokens, threshold
    );
    CUDA_CHECK_LAUNCH();
}

void predict_active_neurons(
    const float* hidden, const float* pred_weight, const float* pred_bias,
    int* active_indices, int* num_active,
    int hidden_dim, int inter_dim, int max_active, float threshold,
    cudaStream_t stream
) {
    // Reset counter
    cudaMemsetAsync(num_active, 0, sizeof(int), stream);

    int threads = 256;
    int blocks = (inter_dim + threads - 1) / threads;
    predict_active_neurons_kernel<<<blocks, threads, 0, stream>>>(
        hidden, pred_weight, pred_bias, active_indices, num_active,
        hidden_dim, inter_dim, max_active, threshold
    );
    CUDA_CHECK_LAUNCH();
}

void sparse_matvec(
    const float* weight, const float* input, float* output,
    const int* active_indices, int num_active, int cols,
    cudaStream_t stream
) {
    int threads = 256;
    int shared_mem = cols * sizeof(float);
    sparse_matvec_fp32_kernel<<<num_active, threads, shared_mem, stream>>>(
        weight, input, output, active_indices, num_active, cols
    );
    CUDA_CHECK_LAUNCH();
}

void sparse_dequant_matvec_int4(
    const void* weights, const void* scales, const void* biases,
    const float* input, float* output,
    const int* active_indices, int num_active, int cols, int group_size,
    cudaStream_t stream
) {
    int threads = 256;
    int shared_mem = cols * sizeof(float);
    sparse_dequant_matvec_int4_kernel<<<num_active, threads, shared_mem, stream>>>(
        (const uint32_t*)weights, (const half*)scales, (const half*)biases,
        input, output, active_indices, num_active, cols, group_size
    );
    CUDA_CHECK_LAUNCH();
}

void sparse_swiglu(
    float* output, const float* gate, const float* up,
    const int* active_indices, int num_active, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_active + threads - 1) / threads;
    sparse_swiglu_kernel<<<blocks, threads, 0, stream>>>(
        output, gate, up, active_indices, num_active
    );
    CUDA_CHECK_LAUNCH();
}

void sparse_down_proj(
    const float* sparse_activation, const float* down_weight, float* output,
    const int* active_indices, int num_active,
    int hidden_dim, int inter_dim, cudaStream_t stream
) {
    int threads = 256;
    sparse_down_proj_kernel<<<hidden_dim, threads, 32 * sizeof(float), stream>>>(
        sparse_activation, down_weight, output,
        active_indices, num_active, hidden_dim, inter_dim
    );
    CUDA_CHECK_LAUNCH();
}

} // namespace cuda
} // namespace titan
