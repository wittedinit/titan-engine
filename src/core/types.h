#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>

namespace titan {

// ============================================================================
// Quantization & Data Types
// ============================================================================

enum class DType : uint8_t {
    FP32   = 0,
    FP16   = 1,
    BF16   = 2,
    FP8_E4M3 = 3,  // NVIDIA FP8 format
    FP8_E5M2 = 4,
    FP4    = 5,     // Blackwell native FP4
    INT8   = 6,
    INT4   = 7,     // 4-bit affine quantized
    INT2   = 8,     // 2-bit extreme compression
    Q4_K   = 9,     // k-quants style (mixed precision within block)
    Q3_K   = 10,
    Q2_K   = 11,
    Q5_K   = 12,
    Q6_K   = 13,
    Q8_0   = 14,
};

// Bytes per element (for non-grouped types)
inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FP32:      return 4;
        case DType::FP16:      return 2;
        case DType::BF16:      return 2;
        case DType::FP8_E4M3:  return 1;
        case DType::FP8_E5M2:  return 1;
        case DType::INT8:      return 1;
        // Sub-byte types: return 0 (use group-based size calculation)
        default:               return 0;
    }
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::FP32:      return "fp32";
        case DType::FP16:      return "fp16";
        case DType::BF16:      return "bf16";
        case DType::FP8_E4M3:  return "fp8_e4m3";
        case DType::FP8_E5M2:  return "fp8_e5m2";
        case DType::FP4:       return "fp4";
        case DType::INT8:      return "int8";
        case DType::INT4:      return "int4";
        case DType::INT2:      return "int2";
        case DType::Q4_K:      return "q4_k";
        case DType::Q3_K:      return "q3_k";
        case DType::Q2_K:      return "q2_k";
        case DType::Q5_K:      return "q5_k";
        case DType::Q6_K:      return "q6_k";
        case DType::Q8_0:      return "q8_0";
    }
    return "unknown";
}

// ============================================================================
// Quantization Block Descriptors
// ============================================================================

// Affine quantization: val = (raw * scale) + bias
// group_size elements share one scale + one bias
struct QuantBlockAffine {
    DType       raw_type;      // INT4, INT2, etc.
    uint32_t    group_size;    // Typically 64 or 128
    DType       scale_type;    // FP16 or BF16
    DType       bias_type;     // FP16 or BF16 (or NONE for symmetric)
};

// ============================================================================
// Tensor Descriptor (lightweight, non-owning view)
// ============================================================================

enum class MemoryTier : uint8_t {
    VRAM  = 0,   // GPU device memory
    RAM   = 1,   // System RAM (pinned or pageable)
    NVME  = 2,   // On-disk (NVMe SSD)
    NONE  = 255, // Not yet placed
};

inline const char* tier_name(MemoryTier t) {
    switch (t) {
        case MemoryTier::VRAM: return "VRAM";
        case MemoryTier::RAM:  return "RAM";
        case MemoryTier::NVME: return "NVMe";
        default:               return "none";
    }
}

struct TensorDesc {
    std::string     name;
    DType           dtype       = DType::FP16;
    std::vector<int64_t> shape;            // e.g., [4096, 4096]
    size_t          byte_offset = 0;       // Offset in storage
    size_t          byte_size   = 0;       // Total bytes
    MemoryTier      tier        = MemoryTier::NONE;

    // For quantized tensors
    QuantBlockAffine quant_block = {};

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

// ============================================================================
// Tensor (owns or borrows data)
// ============================================================================

struct Tensor {
    TensorDesc      desc;
    void*           data    = nullptr;     // Pointer to data (CPU or GPU)
    bool            owns    = false;       // Whether we manage the memory

    Tensor() = default;
    ~Tensor() {
        // Caller is responsible for freeing based on tier
        // (cudaFree for VRAM, free/munmap for RAM, nothing for NVMe)
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& o) noexcept
        : desc(std::move(o.desc)), data(o.data), owns(o.owns) {
        o.data = nullptr;
        o.owns = false;
    }
    Tensor& operator=(Tensor&& o) noexcept {
        desc = std::move(o.desc);
        data = o.data;
        owns = o.owns;
        o.data = nullptr;
        o.owns = false;
        return *this;
    }
};

// ============================================================================
// Model Architecture Descriptors
// ============================================================================

enum class ModelType : uint8_t {
    DENSE      = 0,    // Standard transformer (Llama, Mistral, etc.)
    MOE        = 1,    // Full MoE (Mixtral, DeepSeek-V3, etc.)
    HYBRID_MOE = 2,    // Mixed dense + MoE layers (Qwen3.5-397B)
};

enum class AttentionType : uint8_t {
    FULL_ATTENTION     = 0,   // Standard scaled dot-product
    GATED_DELTA_NET    = 1,   // Linear attention with recurrence (Qwen3.5)
    SLIDING_WINDOW     = 2,   // Mistral-style sliding window
    GROUPED_QUERY      = 3,   // GQA (most modern models)
};

enum class ActivationType : uint8_t {
    SWIGLU  = 0,
    GELU    = 1,
    RELU    = 2,
    SILU    = 3,
};

struct LayerConfig {
    AttentionType   attn_type   = AttentionType::GROUPED_QUERY;
    bool            is_moe      = false;
    uint32_t        num_experts = 0;
    uint32_t        experts_per_tok = 0;
    uint32_t        num_shared_experts = 0;
};

struct ModelConfig {
    std::string     name;
    ModelType       model_type      = ModelType::DENSE;

    // Dimensions
    uint32_t        hidden_dim      = 0;
    uint32_t        num_layers      = 0;
    uint32_t        num_attn_heads  = 0;
    uint32_t        num_kv_heads    = 0;
    uint32_t        head_dim        = 0;
    uint32_t        intermediate_dim = 0;  // FFN intermediate size
    uint32_t        vocab_size      = 0;

    // MoE config (if applicable)
    uint32_t        num_experts         = 0;
    uint32_t        experts_per_tok     = 0;
    uint32_t        num_shared_experts  = 0;
    uint32_t        moe_intermediate_dim = 0;
    uint32_t        first_k_dense_replace = 0;  // First N layers use dense MLP
    uint32_t        moe_layer_freq      = 1;    // Every Nth layer is MoE (1 = all)

    // MLA (Multi-head Latent Attention) config — DeepSeek V3 / Kimi K2 style
    bool            has_mla         = false;
    uint32_t        kv_lora_rank    = 0;    // KV compression rank (0 = standard GQA)
    uint32_t        q_lora_rank     = 0;    // Q compression rank
    uint32_t        rope_head_dim   = 0;    // Decoupled RoPE head dim (MLA only)
    uint32_t        nope_head_dim   = 0;    // Non-RoPE head dim (MLA only)
    uint32_t        v_head_dim      = 0;    // Value head dim (MLA only, may differ from K)

    // Attention config
    float           rope_theta      = 10000.0f;
    float           rope_scaling    = 1.0f;
    uint32_t        max_position    = 131072;

    // Activation
    ActivationType  activation      = ActivationType::SWIGLU;

    // Per-layer overrides (empty = all layers same)
    std::vector<LayerConfig> layer_configs;

    // Derived
    size_t total_params() const;
    size_t active_params_per_token() const;
    size_t estimated_weight_bytes(DType quant) const;
};

// ============================================================================
// Runtime Configuration
// ============================================================================

struct RuntimeConfig {
    // Memory budget
    size_t      vram_budget_mb      = 0;    // 0 = auto-detect
    size_t      ram_budget_mb       = 0;    // 0 = auto-detect
    size_t      nvme_cache_mb       = 0;    // 0 = unlimited

    // Quantization
    DType       weight_dtype        = DType::INT4;
    DType       kv_cache_dtype      = DType::FP16;
    DType       compute_dtype       = DType::FP16;

    // Inference
    uint32_t    max_batch_size      = 1;
    uint32_t    max_context_len     = 8192;
    uint32_t    num_speculative     = 0;    // 0 = no speculative decoding

    // I/O
    uint32_t    io_threads          = 4;
    bool        use_io_uring        = true;
    bool        use_direct_io       = true;

    // Scheduling
    bool        enable_prefetch     = true;
    uint32_t    prefetch_layers     = 1;    // How many layers ahead to prefetch

    // Paths
    std::string model_path;
    std::string nvme_cache_path;
};

// ============================================================================
// Sampling Parameters
// ============================================================================

struct SamplingParams {
    float       temperature     = 0.7f;
    float       top_p           = 0.9f;
    uint32_t    top_k           = 40;
    float       repetition_penalty = 1.1f;
    uint32_t    max_tokens      = 2048;
    uint64_t    seed            = 0;    // 0 = random
};

} // namespace titan
