#include "core/types.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace titan {
namespace cuda {

// ============================================================================
// RoPE (Rotary Position Embedding) — fused with Q/K projection output
// ============================================================================

__global__ void rope_kernel(
    float* __restrict__ q,          // [num_heads, head_dim]
    float* __restrict__ k,          // [num_kv_heads, head_dim]
    int num_heads, int num_kv_heads, int head_dim,
    int position, float theta_base, float rope_scaling
) {
    int head = blockIdx.x;
    int d = threadIdx.x * 2; // Process pairs of dimensions
    if (d >= head_dim) return;

    float freq = 1.0f / powf(theta_base, (float)d / head_dim);
    freq *= rope_scaling;
    float angle = position * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Apply to Q heads
    if (head < num_heads) {
        float* qh = q + head * head_dim;
        float q0 = qh[d];
        float q1 = qh[d + 1];
        qh[d]     = q0 * cos_val - q1 * sin_val;
        qh[d + 1] = q0 * sin_val + q1 * cos_val;
    }

    // Apply to K heads (fewer heads with GQA)
    if (head < num_kv_heads) {
        float* kh = k + head * head_dim;
        float k0 = kh[d];
        float k1 = kh[d + 1];
        kh[d]     = k0 * cos_val - k1 * sin_val;
        kh[d + 1] = k0 * sin_val + k1 * cos_val;
    }
}

// ============================================================================
// Scaled Dot-Product Attention (single query, for autoregressive decoding)
//
// For batch/prefill, we should use Flash Attention 2.
// This kernel handles the common decode case: one query token attending to
// all previous KV cache entries.
// ============================================================================

__global__ void attention_decode_kernel(
    const float* __restrict__ q,        // [num_heads, head_dim]
    const float* __restrict__ k_cache,  // [seq_len, num_kv_heads, head_dim]
    const float* __restrict__ v_cache,  // [seq_len, num_kv_heads, head_dim]
    float*       __restrict__ output,   // [num_heads, head_dim]
    int num_heads, int num_kv_heads, int head_dim, int seq_len,
    float scale // 1/sqrt(head_dim)
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;

    // GQA: map query head to KV head
    int kv_head = head / (num_heads / num_kv_heads);

    extern __shared__ float shared[];
    float* scores = shared;                        // [seq_len]
    float* out_accum = shared + seq_len;           // [head_dim]
    // Reuse out_accum area for warp partials (needs max 32 floats, head_dim >= 32)
    float* warp_buf = out_accum;                   // [num_warps] temp during reductions

    const float* qh = q + head * head_dim;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = (blockDim.x + 31) / 32;

    // Step 1: Compute attention scores (Q @ K^T)
    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        const float* kh = k_cache + (size_t)pos * num_kv_heads * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qh[d] * kh[d];
        }
        scores[pos] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax over scores
    // Find max for numerical stability — proper cross-warp reduction
    float max_val = -1e30f;
    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        max_val = fmaxf(max_val, scores[pos]);
    }
    // Intra-warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    // Cross-warp reduce: each warp's lane 0 writes partial to shared memory
    if (lane_id == 0) warp_buf[warp_id] = max_val;
    __syncthreads();
    // Warp 0 reduces across all warp partials
    if (warp_id == 0) {
        max_val = (lane_id < num_warps) ? warp_buf[lane_id] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
        }
        if (lane_id == 0) warp_buf[0] = max_val;
    }
    __syncthreads();
    max_val = warp_buf[0];

    // Exp and sum — proper cross-warp reduction
    float sum_exp = 0.0f;
    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        scores[pos] = expf(scores[pos] - max_val);
        sum_exp += scores[pos];
    }
    // Intra-warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    // Cross-warp reduce sum
    if (lane_id == 0) warp_buf[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        sum_exp = (lane_id < num_warps) ? warp_buf[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        }
        if (lane_id == 0) warp_buf[0] = sum_exp;
    }
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / (warp_buf[0] + 1e-8f);
    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        scores[pos] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum of V (scores @ V)
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            const float* vh = v_cache + (size_t)pos * num_kv_heads * head_dim + kv_head * head_dim;
            acc += scores[pos] * vh[d];
        }
        output[head * head_dim + d] = acc;
    }
}

// ============================================================================
// MLA Attention Decode — K and V have different head dims
//
// Q: [num_heads, k_head_dim] — score computation uses k_head_dim
// K: [seq_len, num_kv_heads, k_head_dim] — includes RoPE dimensions
// V: [seq_len, num_kv_heads, v_head_dim] — nope dims only
// Output: [num_heads, v_head_dim]
// ============================================================================

__global__ void attention_decode_mla_kernel(
    const float* __restrict__ q,        // [num_heads, k_head_dim]
    const float* __restrict__ k_cache,  // [seq_len, num_kv_heads, k_head_dim]
    const float* __restrict__ v_cache,  // [seq_len, num_kv_heads, v_head_dim]
    float*       __restrict__ output,   // [num_heads, v_head_dim]
    int num_heads, int num_kv_heads,
    int k_head_dim, int v_head_dim, int seq_len,
    float scale
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;

    int kv_head = head / (num_heads / num_kv_heads);

    extern __shared__ float shared[];
    float* scores = shared;                            // [seq_len]
    float* warp_buf = shared + seq_len;                // [32] for reductions

    const float* qh = q + head * k_head_dim;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = (blockDim.x + 31) / 32;

    // Step 1: Compute attention scores Q @ K^T using k_head_dim
    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        const float* kh = k_cache + (size_t)pos * num_kv_heads * k_head_dim + kv_head * k_head_dim;
        float dot = 0.0f;
        for (int d = 0; d < k_head_dim; d++) {
            dot += qh[d] * kh[d];
        }
        scores[pos] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax
    float max_val = -1e30f;
    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        max_val = fmaxf(max_val, scores[pos]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    if (lane_id == 0) warp_buf[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        max_val = (lane_id < num_warps) ? warp_buf[lane_id] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
        if (lane_id == 0) warp_buf[0] = max_val;
    }
    __syncthreads();
    max_val = warp_buf[0];

    float sum_exp = 0.0f;
    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        scores[pos] = expf(scores[pos] - max_val);
        sum_exp += scores[pos];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    if (lane_id == 0) warp_buf[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        sum_exp = (lane_id < num_warps) ? warp_buf[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        if (lane_id == 0) warp_buf[0] = sum_exp;
    }
    __syncthreads();

    float inv_sum = 1.0f / (warp_buf[0] + 1e-8f);
    for (int pos = threadIdx.x; pos < seq_len; pos += blockDim.x) {
        scores[pos] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum of V using v_head_dim (different from k_head_dim)
    for (int d = threadIdx.x; d < v_head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            const float* vh = v_cache + (size_t)pos * num_kv_heads * v_head_dim + kv_head * v_head_dim;
            acc += scores[pos] * vh[d];
        }
        output[head * v_head_dim + d] = acc;
    }
}

// ============================================================================
// RoPE for MLA — only applies to the last rope_hd dims of each head
// ============================================================================

__global__ void rope_mla_kernel(
    float* __restrict__ q,          // [num_heads, nope_hd + rope_hd]
    float* __restrict__ k,          // [num_kv_heads, nope_hd + rope_hd]
    int num_heads, int num_kv_heads,
    int nope_hd, int rope_hd,
    int position, float theta_base
) {
    int head = blockIdx.x;
    int d = threadIdx.x * 2;  // Process pairs within rope portion
    if (d >= rope_hd) return;

    float freq = 1.0f / powf(theta_base, (float)d / rope_hd);
    float angle = position * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    int head_dim = nope_hd + rope_hd;

    // Apply to Q heads — rope portion starts at offset nope_hd
    if (head < num_heads) {
        float* qh = q + head * head_dim + nope_hd;
        float q0 = qh[d];
        float q1 = qh[d + 1];
        qh[d]     = q0 * cos_val - q1 * sin_val;
        qh[d + 1] = q0 * sin_val + q1 * cos_val;
    }

    // Apply to K heads — rope portion starts at offset nope_hd
    if (head < num_kv_heads) {
        float* kh = k + head * head_dim + nope_hd;
        float k0 = kh[d];
        float k1 = kh[d + 1];
        kh[d]     = k0 * cos_val - k1 * sin_val;
        kh[d + 1] = k0 * sin_val + k1 * cos_val;
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void apply_rope(
    float* q, float* k,
    int num_heads, int num_kv_heads, int head_dim,
    int position, float theta, float scaling,
    cudaStream_t stream
) {
    int max_heads = (num_heads > num_kv_heads) ? num_heads : num_kv_heads;
    int threads = head_dim / 2;
    rope_kernel<<<max_heads, threads, 0, stream>>>(
        q, k, num_heads, num_kv_heads, head_dim,
        position, theta, scaling
    );
    CUDA_CHECK_LAUNCH();
}

void attention_decode(
    const float* q, const float* k_cache, const float* v_cache,
    float* output,
    int num_heads, int num_kv_heads, int head_dim, int seq_len,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int shared_mem = (seq_len + head_dim) * sizeof(float);
    int threads = 256;

    attention_decode_kernel<<<num_heads, threads, shared_mem, stream>>>(
        q, k_cache, v_cache, output,
        num_heads, num_kv_heads, head_dim, seq_len, scale
    );
    CUDA_CHECK_LAUNCH();
}

void attention_decode_mla(
    const float* q, const float* k_cache, const float* v_cache,
    float* output,
    int num_heads, int num_kv_heads,
    int k_head_dim, int v_head_dim, int seq_len,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf((float)k_head_dim);
    // Shared memory: scores[seq_len] + warp_buf[32]
    int shared_mem = (seq_len + 32) * sizeof(float);
    int threads = 256;

    attention_decode_mla_kernel<<<num_heads, threads, shared_mem, stream>>>(
        q, k_cache, v_cache, output,
        num_heads, num_kv_heads, k_head_dim, v_head_dim, seq_len, scale
    );
    CUDA_CHECK_LAUNCH();
}

void apply_rope_mla(
    float* q, float* k,
    int num_heads, int num_kv_heads,
    int nope_hd, int rope_hd,
    int position, float theta,
    cudaStream_t stream
) {
    int max_heads = (num_heads > num_kv_heads) ? num_heads : num_kv_heads;
    int threads = rope_hd / 2;
    if (threads < 1) return;
    rope_mla_kernel<<<max_heads, threads, 0, stream>>>(
        q, k, num_heads, num_kv_heads,
        nope_hd, rope_hd, position, theta
    );
    CUDA_CHECK_LAUNCH();
}

} // namespace cuda
} // namespace titan
