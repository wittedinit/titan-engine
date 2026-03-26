#include "cuda_check.h"
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
// NOTE: `temperature` parameter removed — only `inv_temperature` is used
// (caller computes 1.0f / temperature).
__global__ void apply_temperature_softmax_kernel(
    float*       __restrict__ probs,
    const float* __restrict__ logits,
    int vocab_size, float inv_temperature
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
// WARNING: This kernel is NOT implemented. It finds the global max but never
// filters. Do not use until a proper top-k algorithm (radix select or bitonic
// sort) is implemented. For now, top-k filtering must be done on the CPU side
// or skipped.
__global__ void topk_filter_kernel(
    float* __restrict__ probs,
    int vocab_size, int k
) {
    // UNIMPLEMENTED — stub only.
    // TODO: Implement proper top-k filtering using radix select or bitonic sort.
    // The previous code found the global max but never applied any filtering.
    // For now, this kernel is intentionally a no-op to avoid silent corruption.
    (void)probs;
    (void)vocab_size;
    (void)k;
}

// Top-P (nucleus) sampling
// Probabilities are scanned in descending order to implement proper nucleus
// sampling. Uses a simple single-thread selection approach (sufficient for
// single-token sampling from a pre-computed probability distribution).
__global__ void topp_sample_kernel(
    const float* __restrict__ probs,
    int*         __restrict__ output_token,
    int vocab_size, float top_p, float random_val
) {
    // Single thread — this is fast enough for single-token sampling
    if (threadIdx.x > 0 || blockIdx.x > 0) return;

    // Proper nucleus sampling requires processing probabilities in descending
    // order. We use an iterative selection approach: repeatedly find the max
    // of the remaining probabilities, accumulate into the nucleus, and sample.
    // This is O(V * K) where K is the number of tokens in the nucleus, which
    // is typically small (10-100 tokens cover top_p=0.9).

    float cumsum = 0.0f;
    float target = random_val; // random_val in [0, 1)

    // We need to track which tokens have been "consumed" into the nucleus.
    // Since we can't allocate, we track the nucleus mass and selected token
    // by scanning for the next-largest probability each iteration.
    // To avoid modifying probs (const), we track a running threshold.

    float nucleus_mass = 0.0f;
    int selected_token = vocab_size - 1; // fallback

    // Collect nucleus mass first: find total mass in top-p
    // by iteratively selecting the largest remaining probability
    float prev_min = FLT_MAX; // threshold: only consider probs < this
    int prev_idx = -1;        // tie-break: only consider idx > this when prob == prev_min

    while (nucleus_mass < top_p) {
        // Find the largest probability not yet consumed
        float best_prob = -1.0f;
        int best_idx = -1;
        for (int i = 0; i < vocab_size; i++) {
            float p = probs[i];
            // Skip already-consumed tokens:
            // A token is "consumed" if its prob > prev_min,
            // or (prob == prev_min and idx <= prev_idx)
            if (p > prev_min) continue;
            if (p == prev_min && i <= prev_idx) continue;
            if (p > best_prob || (p == best_prob && i < best_idx)) {
                best_prob = p;
                best_idx = i;
            }
        }
        if (best_idx < 0) break; // no more tokens

        nucleus_mass += best_prob;
        cumsum += best_prob;

        // Check if this token is the sampled one
        float rescaled_target = target * fminf(nucleus_mass, top_p);
        // We need to recompute cumsum from scratch for proper sampling.
        // Actually, let's do the sampling in a second pass for clarity.

        prev_min = best_prob;
        prev_idx = best_idx;
    }

    // Second pass: sample from the nucleus in descending probability order
    cumsum = 0.0f;
    float rescaled_target = target * fminf(nucleus_mass, top_p);
    prev_min = FLT_MAX;
    prev_idx = -1;

    while (true) {
        float best_prob = -1.0f;
        int best_idx = -1;
        for (int i = 0; i < vocab_size; i++) {
            float p = probs[i];
            if (p > prev_min) continue;
            if (p == prev_min && i <= prev_idx) continue;
            if (p > best_prob || (p == best_prob && i < best_idx)) {
                best_prob = p;
                best_idx = i;
            }
        }
        if (best_idx < 0) break;

        cumsum += best_prob;
        if (cumsum >= rescaled_target) {
            *output_token = best_idx;
            return;
        }

        prev_min = best_prob;
        prev_idx = best_idx;
    }

    *output_token = selected_token; // Fallback
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
        CUDA_CHECK_LAUNCH();
    } else {
        // Temperature + softmax
        // TODO: probs buffer must be pre-allocated by the caller and passed in.
        // Using static thread-local device pointer as a temporary workaround to
        // avoid cudaMalloc/cudaFree in the hot path. This is NOT safe for
        // concurrent calls from different host threads sharing the same device
        // context — the caller should pre-allocate and pass the buffer.
        static thread_local float* probs = nullptr;
        static thread_local int probs_capacity = 0;

        if (probs == nullptr || probs_capacity < vocab_size) {
            if (probs) {
                cudaStreamSynchronize(stream);
                cudaFree(probs);
            }
            cudaMalloc(&probs, vocab_size * sizeof(float));
            probs_capacity = vocab_size;
        }

        int shared_mem = 32 * sizeof(float);
        apply_temperature_softmax_kernel<<<1, 1024, shared_mem, stream>>>(
            probs, logits, vocab_size, 1.0f / temperature
        );
        CUDA_CHECK_LAUNCH();

        // Generate random number
        float random_val;
        // Simplified: in production, use a persistent generator
        // For now, use a simple hash-based random
        random_val = (float)((seed * 6364136223846793005ULL + 1442695040888963407ULL) & 0xFFFFFF) / (float)0xFFFFFF;

        // Top-P sampling
        topp_sample_kernel<<<1, 1, 0, stream>>>(probs, output_token, vocab_size, top_p, random_val);
        CUDA_CHECK_LAUNCH();
    }
}

} // namespace cuda
} // namespace titan
