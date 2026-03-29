#pragma once

#include "core/types.h"
#include <cuda_runtime.h>
#include <cstdint>

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

    // --- dequant.cu ---
    void dequant_matvec_int4(const void* weights, const void* scales, const void* biases,
                             const float* input, float* output,
                             int rows, int cols, int group_size, cudaStream_t stream);
    void dequant_matvec_int2(const void* weights, const void* scales, const void* biases,
                             const float* input, float* output,
                             int rows, int cols, int group_size, cudaStream_t stream);

    // --- norm.cu ---
    void rmsnorm(float* output, const float* input, const float* weight,
                 int dim, float eps, cudaStream_t stream);
    void layernorm(float* output, const float* input,
                   const float* weight, const float* bias,
                   int dim, float eps, cudaStream_t stream);

    // --- activation.cu ---
    void swiglu(float* output, const float* gate, const float* up,
                int dim, cudaStream_t stream);
    void gelu(float* output, const float* input, int dim, cudaStream_t stream);
    void fused_add_rmsnorm(float* output, float* residual, const float* hidden,
                            const float* weight, int dim, float eps,
                            cudaStream_t stream);
    // --- MLA helpers (activation.cu) ---
    void mla_deinterleave_kv(const float* kv_expanded, float* k_nope, float* v,
                             int n_heads, int nope_hd, int v_hd, cudaStream_t stream);
    void mla_assemble_k(const float* k_nope, const float* k_rope_dec, float* k_out,
                        int n_heads, int nope_hd, int rope_hd, cudaStream_t stream);
    void mla_extract_q_nope(const float* q_full, float* q_nope,
                            int n_heads, int nope_hd, int rope_hd, cudaStream_t stream);

    void fused_moe_combine_norm(float* output, float* residual,
                                 const float* expert_outputs, const float* routing_weights,
                                 const float* shared_expert, float shared_weight,
                                 const float* norm_weight,
                                 int dim, int num_active, float eps, cudaStream_t stream);

    // --- attention.cu ---
    void apply_rope(float* q, float* k, int num_heads, int num_kv_heads, int head_dim,
                    int position, float theta, float scaling, cudaStream_t stream);
    void attention_decode(const float* q, const float* k_cache, const float* v_cache,
                          float* output, int num_heads, int num_kv_heads,
                          int head_dim, int seq_len, cudaStream_t stream);

    // --- moe.cu ---
    void moe_gate(const float* hidden, const float* gate_weight, float* logits,
                  int hidden_dim, int num_experts, cudaStream_t stream);
    void moe_topk(const float* logits, float* weights, int* indices,
                  int num_experts, int k, cudaStream_t stream);

    // --- sampling.cu ---
    void sample_token(const float* logits, int* output_token,
                      int vocab_size, float temperature, float top_p, int top_k,
                      uint64_t seed, cudaStream_t stream);

    // --- gemv.cu ---
    void init_cublas();
    void destroy_cublas();
    void gemv_fp32(const float* A, const float* x, float* y,
                   int rows, int cols, cudaStream_t stream);
    void gemv_fp32_batched(const float* A, const float* x, float* y,
                           int rows, int cols, int batch, cudaStream_t stream);
    void gemv_bf16_to_fp32(const void* A, const float* x, float* y,
                           int rows, int cols, cudaStream_t stream);
    void embed_token_bf16(float* out, const void* emb, int token_id, int dim,
                          cudaStream_t stream);
    void vector_add(float* y, const float* a, const float* b, int n, cudaStream_t stream);
    void vector_copy(float* dst, const float* src, int n, cudaStream_t stream);

    // --- fp4.cu ---
    void dequant_matvec_fp4(const void* weights, const void* scales,
                            const float* input, float* output,
                            int rows, int cols, int group_size, cudaStream_t stream);
    // NVFP4: U8 packed (2 FP4/byte), F8_E4M3 scales (group=16), F32 global scale
    void dequant_matvec_nvfp4(const void* weights, const void* scales, float global_scale,
                               const float* input, float* output,
                               int rows, int cols, cudaStream_t stream);
    void quantize_fp4(const float* input, void* output, void* scales,
                      int numel, int group_size, cudaStream_t stream);

    // --- sparse.cu ---
    void profile_activations(const float* activations, float* magnitude_sum, int* nonzero_count,
                             int inter_dim, int batch_tokens, float threshold, cudaStream_t stream);
    void predict_active_neurons(const float* hidden, const float* pred_weight, const float* pred_bias,
                                int* active_indices, int* num_active,
                                int hidden_dim, int inter_dim, int max_active, float threshold,
                                cudaStream_t stream);
    void sparse_matvec(const float* weight, const float* input, float* output,
                       const int* active_indices, int num_active, int cols,
                       cudaStream_t stream);
    void sparse_dequant_matvec_int4(const void* weights, const void* scales, const void* biases,
                                    const float* input, float* output,
                                    const int* active_indices, int num_active, int cols, int group_size,
                                    cudaStream_t stream);
    void sparse_swiglu(float* output, const float* gate, const float* up,
                       const int* active_indices, int num_active, cudaStream_t stream);
    void sparse_down_proj(const float* sparse_activation, const float* down_weight, float* output,
                          const int* active_indices, int num_active,
                          int hidden_dim, int inter_dim, cudaStream_t stream);
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
