#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cfloat>

namespace titan {
namespace cuda {

// ============================================================================
// GPU-Accelerated Sampling
//
// For large vocabularies (248K+ tokens), computing argmax/top-k on GPU
// is faster than copying logits to CPU.
// ============================================================================

// Temperature scaling + softmax
__global__ void apply_temperature_softmax_kernel(
    float*       __restrict__ probs,
    const float* __restrict__ logits,
    int vocab_size, float temperature, float inv_temperature
) {
    extern __shared__ float shared[];

    // Pass 1: Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float scaled = logits[i] * inv_temperature;
        probs[i] = scaled;
        max_val = fmaxf(max_val, scaled);
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) shared[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        max_val = (lane_id < nw) ? shared[lane_id] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
        }
        if (lane_id == 0) shared[0] = max_val;
    }
    __syncthreads();
    max_val = shared[0];

    // Pass 2: Exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = expf(probs[i] - max_val);
        probs[i] = val;
        sum += val;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane_id == 0) shared[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        sum = (lane_id < nw) ? shared[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) shared[0] = sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / (shared[0] + 1e-8f);

    // Pass 3: Normalize
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        probs[i] *= inv_sum;
    }
}

// Top-K filtering: zero out all probabilities outside top-K
__global__ void topk_filter_kernel(
    float* __restrict__ probs,
    int vocab_size, int k
) {
    // For large vocab + small k, this uses a partial sort approach
    // Single block implementation for simplicity
    if (blockIdx.x > 0) return;

    extern __shared__ float shared[];

    // Find the k-th largest value (threshold)
    // Use iterative approach: find max, mark it, repeat K times
    // For production: use radix select or bitonic sort
    float threshold = 0.0f;

    // Simple approach: find approximate threshold
    // Scan for rough quantile
    float max_val = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        max_val = fmaxf(max_val, probs[i]);
    }
    // Reduce to find global max
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }

    // Binary search for threshold that keeps approximately K elements
    // (Simplified — in production, use a proper top-k algorithm)
    // For now, just do the filtering on CPU after softmax
}

// Top-P (nucleus) sampling
__global__ void topp_sample_kernel(
    const float* __restrict__ probs,
    int*         __restrict__ output_token,
    int vocab_size, float top_p, float random_val
) {
    // Single thread — this is fast enough for single-token sampling
    if (threadIdx.x > 0 || blockIdx.x > 0) return;

    // Cumulative sum until we reach top_p threshold
    // Note: probs should be sorted for proper nucleus sampling
    // For simplicity, we scan in order and accept once cumsum > random_val * top_p_mass
    float cumsum = 0.0f;
    float target = random_val; // random_val in [0, 1)

    // Renormalize within top_p mass
    float total_p = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        total_p += probs[i];
        if (total_p >= top_p) break;
    }

    float rescaled_target = target * fminf(total_p, top_p);
    cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= rescaled_target) {
            *output_token = i;
            return;
        }
    }
    *output_token = vocab_size - 1; // Fallback
}

// Greedy (argmax) sampling
__global__ void argmax_kernel(
    const float* __restrict__ logits,
    int*         __restrict__ output_token,
    int vocab_size
) {
    extern __shared__ float shared[];
    float* svals = shared;
    int*   sidxs = (int*)(shared + 32);

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        svals[warp_id] = max_val;
        sidxs[warp_id] = max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        max_val = (lane_id < nw) ? svals[lane_id] : -FLT_MAX;
        max_idx = (lane_id < nw) ? sidxs[lane_id] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }
        if (lane_id == 0) *output_token = max_idx;
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void sample_token(
    const float* logits, int* output_token,
    int vocab_size, float temperature, float top_p, int top_k,
    uint64_t seed, cudaStream_t stream
) {
    if (temperature < 0.01f) {
        // Greedy
        int shared_mem = 64 * sizeof(float);
        argmax_kernel<<<1, 1024, shared_mem, stream>>>(logits, output_token, vocab_size);
    } else {
        // Temperature + softmax
        float* probs;
        cudaMalloc(&probs, vocab_size * sizeof(float));

        int shared_mem = 32 * sizeof(float);
        apply_temperature_softmax_kernel<<<1, 1024, shared_mem, stream>>>(
            probs, logits, vocab_size, temperature, 1.0f / temperature
        );

        // Generate random number
        float random_val;
        curandGenerator_t gen;
        // Simplified: in production, use a persistent generator
        // For now, use a simple hash-based random
        random_val = (float)((seed * 6364136223846793005ULL + 1442695040888963407ULL) & 0xFFFFFF) / (float)0xFFFFFF;

        // Top-P sampling
        topp_sample_kernel<<<1, 1, 0, stream>>>(probs, output_token, vocab_size, top_p, random_val);

        cudaFree(probs);
    }
}

} // namespace cuda
} // namespace titan
