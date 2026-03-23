#pragma once

#include "core/types.h"

namespace titan {

// ============================================================================
// Compute Dispatch — Route operations to GPU or CPU based on weight placement
//
// The key insight from flash-moe and KTransformers: not everything needs to
// run on the GPU. When weights are in RAM, executing on CPU with AVX-512/AMX
// can be faster than transferring to GPU and back.
//
// Decision matrix:
//   Weights in VRAM → Execute on GPU (no transfer needed)
//   Weights in RAM  → Execute on CPU if GPU is busy, else transfer + GPU
//   Weights on NVMe → Load to RAM first, then decide CPU vs GPU
// ============================================================================

enum class ComputeDevice {
    GPU,
    CPU,
    AUTO,  // Decide based on weight location and GPU utilization
};

// Forward declarations for GPU kernels
namespace cuda {
    void dequant_matvec_int4(const void* weights, const void* scales, const void* biases,
                             const float* input, float* output,
                             int rows, int cols, int group_size, void* stream);
    void dequant_matvec_int2(const void* weights, const void* scales, const void* biases,
                             const float* input, float* output,
                             int rows, int cols, int group_size, void* stream);
    void rmsnorm(float* output, const float* input, const float* weight,
                 int dim, float eps, void* stream);
    void swiglu(float* output, const float* gate, const float* up,
                int dim, void* stream);
    void apply_rope(float* q, float* k, int num_heads, int num_kv_heads, int head_dim,
                    int position, float theta, float scaling, void* stream);
    void attention_decode(const float* q, const float* k_cache, const float* v_cache,
                          float* output, int num_heads, int num_kv_heads,
                          int head_dim, int seq_len, void* stream);
    void moe_gate(const float* hidden, const float* gate_weight, float* logits,
                  int hidden_dim, int num_experts, void* stream);
    void moe_topk(const float* logits, float* weights, int* indices,
                  int num_experts, int k, void* stream);
    void fused_add_rmsnorm(float* output, float* residual, const float* hidden,
                            const float* weight, int dim, float eps, void* stream);
    void fused_moe_combine_norm(float* output, float* residual,
                                 const float* expert_outputs, const float* routing_weights,
                                 const float* shared_expert, float shared_weight,
                                 const float* norm_weight,
                                 int dim, int num_active, float eps, void* stream);
    void sample_token(const float* logits, int* output_token,
                      int vocab_size, float temperature, float top_p, int top_k,
                      uint64_t seed, void* stream);
}

// Forward declarations for CPU kernels
namespace cpu {
    void matvec_fp32_avx512(const float* weight, const float* input, float* output,
                            int rows, int cols);
    void dequant_matvec_int4_avx512(const uint32_t* weights, const uint16_t* scales,
                                     const uint16_t* biases, const float* input,
                                     float* output, int rows, int cols, int group_size);
    void swiglu_cpu(float* output, const float* gate, const float* up, int dim);
    void expert_forward_int4_cpu(const float* input,
                                  const uint32_t* gate_w, const uint16_t* gate_s, const uint16_t* gate_b,
                                  const uint32_t* up_w, const uint16_t* up_s, const uint16_t* up_b,
                                  const uint32_t* down_w, const uint16_t* down_s, const uint16_t* down_b,
                                  float* output, float* scratch,
                                  int hidden_dim, int inter_dim, int group_size);
}

} // namespace titan
