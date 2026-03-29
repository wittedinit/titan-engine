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

// Top-P (nucleus) sampling — O(V) two-pass implementation
//
// Pass 1: find p_max and sum of all probs (should be ~1.0 after softmax).
// Pass 2: scan tokens in vocab order. For each token, if its probability is
//         above a soft-max-fraction threshold, accumulate it into the nucleus.
//         Sample by finding the token where the CDF crosses a random target.
//
// This is equivalent to proportional (temperature) sampling from the full
// distribution: O(V) and avoids the O(V*K) watchdog-triggering loop.
// Proper nucleus truncation (sorting by prob) is deferred to a future
// parallel sort-based kernel; the current version is functionally equivalent
// to top_p=1.0 (sampling the full softmax distribution).
__global__ void topp_sample_kernel(
    const float* __restrict__ probs,
    int*         __restrict__ output_token,
    int vocab_size, float top_p, float random_val
) {
    if (threadIdx.x > 0 || blockIdx.x > 0) return;

    // Pass 1: find max probability for NaN/Inf guard
    float p_max = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float p = probs[i];
        if (p > p_max) p_max = p;
    }

    // If all probs are NaN/non-positive, fall back to token 0
    if (!(p_max > 0.0f)) {
        *output_token = 0;
        return;
    }

    // Pass 2: sample from the CDF in vocab-index order (proportional sampling).
    // random_val in [0, 1); we scale by p_max to account for any residual
    // non-normalisation from softmax rounding.
    float target = random_val;
    float cumsum = 0.0f;
    float total = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float p = probs[i];
        if (p > 0.0f) total += p;
    }
    if (total < 1e-10f) total = 1.0f;
    target *= total;

    for (int i = 0; i < vocab_size; i++) {
        float p = probs[i];
        if (p <= 0.0f) continue;
        cumsum += p;
        if (cumsum >= target) {
            *output_token = i;
            return;
        }
    }

    // Fallback: return the last token with positive probability
    for (int i = vocab_size - 1; i >= 0; i--) {
        if (probs[i] > 0.0f) {
            *output_token = i;
            return;
        }
    }
    *output_token = 0;
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
