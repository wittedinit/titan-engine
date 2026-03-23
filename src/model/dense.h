#pragma once

#include "model/architecture.h"
#include "model/tokenizer.h"
#include "inference/kv_cache.h"
#include "core/config.h"

#include <cuda_runtime.h>

namespace titan {

// ============================================================================
// Dense Transformer Executor
//
// Implements the full forward pass for standard dense transformers:
// Llama, Mistral, Qwen, etc.
//
// Per-layer pipeline:
//   1. RMSNorm (attention norm)
//   2. Q/K/V projections (dequant matvec)
//   3. RoPE on Q and K
//   4. Update KV cache
//   5. Attention (Flash Attention decode or full)
//   6. O projection + residual add
//   7. RMSNorm (FFN norm)
//   8. Gate + Up projections (dequant matvec)
//   9. SwiGLU activation
//   10. Down projection
//   11. Residual add
// ============================================================================

class DenseExecutor : public ModelArchitecture {
public:
    DenseExecutor() = default;
    ~DenseExecutor() override;

    const ModelConfig& config() const override { return config_; }

    bool initialize(const std::string& model_path,
                   MemoryManager& memory,
                   const RuntimeConfig& runtime) override;

    void forward_layer(float* hidden, float* residual,
                       uint32_t layer_id, int position,
                       void* cuda_stream) override;

    void compute_logits(const float* hidden, float* logits,
                        void* cuda_stream) override;

    size_t attention_weight_bytes(uint32_t layer_id) const override;
    size_t ffn_weight_bytes(uint32_t layer_id) const override;
    size_t expert_weight_bytes(uint32_t, uint32_t) const override { return 0; }
    size_t kv_cache_bytes_per_token(uint32_t layer_id) const override;

    void update_kv_cache(uint32_t layer_id, int position,
                         const float* key, const float* value) override;

    // Embedding lookup (token_id → hidden state)
    void embed_token(int token_id, float* output, cudaStream_t stream = nullptr);

    // Access KV cache
    KVCache& kv_cache() { return kv_cache_; }

private:
    ModelConfig config_;
    RuntimeConfig runtime_;
    MemoryManager* memory_ = nullptr;
    KVCache kv_cache_;

    // Per-layer weights (pointers into VRAM/RAM)
    DType weight_format_ = DType::FP32;  // Format of loaded weights
    int quant_group_size_ = 64;          // Group size for quantized weights

    struct LayerWeights {
        // Attention norms
        float* attn_norm = nullptr;     // [hidden_dim]
        float* ffn_norm = nullptr;      // [hidden_dim]

        // Attention projections (may be quantized)
        void* q_proj = nullptr;         // [num_heads * head_dim, hidden_dim]
        void* q_proj_scales = nullptr;
        void* q_proj_biases = nullptr;
        void* k_proj = nullptr;         // [num_kv_heads * head_dim, hidden_dim]
        void* k_proj_scales = nullptr;
        void* k_proj_biases = nullptr;
        void* v_proj = nullptr;
        void* v_proj_scales = nullptr;
        void* v_proj_biases = nullptr;
        void* o_proj = nullptr;         // [hidden_dim, num_heads * head_dim]
        void* o_proj_scales = nullptr;
        void* o_proj_biases = nullptr;

        // FFN projections
        void* gate_proj = nullptr;      // [inter_dim, hidden_dim]
        void* gate_proj_scales = nullptr;
        void* gate_proj_biases = nullptr;
        void* up_proj = nullptr;        // [inter_dim, hidden_dim]
        void* up_proj_scales = nullptr;
        void* up_proj_biases = nullptr;
        void* down_proj = nullptr;      // [hidden_dim, inter_dim]
        void* down_proj_scales = nullptr;
        void* down_proj_biases = nullptr;
    };

    std::vector<LayerWeights> layer_weights_;

    // Embedding and LM head
    float* embedding_ = nullptr;        // [vocab_size, hidden_dim]
    void* lm_head_ = nullptr;           // [vocab_size, hidden_dim] (may be quantized)
    void* lm_head_scales_ = nullptr;
    void* lm_head_biases_ = nullptr;
    float* final_norm_ = nullptr;       // [hidden_dim]

    // Scratch buffers (on GPU)
    float* q_buf_ = nullptr;            // [num_heads * head_dim]
    float* k_buf_ = nullptr;            // [num_kv_heads * head_dim]
    float* v_buf_ = nullptr;
    float* attn_out_ = nullptr;         // [num_heads * head_dim]
    float* gate_buf_ = nullptr;         // [inter_dim]
    float* up_buf_ = nullptr;           // [inter_dim]
    float* down_buf_ = nullptr;         // [hidden_dim]
    float* norm_buf_ = nullptr;         // [hidden_dim]

    // Weight loading helpers
    bool load_weights(const std::string& model_path);
    void allocate_scratch_buffers();
    void free_scratch_buffers();
};

} // namespace titan
