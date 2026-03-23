#pragma once

#include "model/architecture.h"
#include "model/loader.h"
#include "model/tokenizer.h"
#include "inference/kv_cache.h"

namespace titan {

// ============================================================================
// MoE Transformer Executor
//
// Extends the dense executor with expert routing and 3-tier expert management.
// Pre-allocates all GPU scratch buffers at init to avoid per-token cudaMalloc.
// ============================================================================

class MoEExecutor : public ModelArchitecture {
public:
    MoEExecutor() = default;
    ~MoEExecutor() override;

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
    size_t expert_weight_bytes(uint32_t layer_id, uint32_t expert_id) const override;
    size_t kv_cache_bytes_per_token(uint32_t layer_id) const override;

    void update_kv_cache(uint32_t layer_id, int position,
                         const float* key, const float* value) override;

    void embed_token(int token_id, float* output, cudaStream_t stream = nullptr);
    KVCache& kv_cache() { return kv_cache_; }

private:
    ModelConfig config_;
    RuntimeConfig runtime_;
    MemoryManager* memory_ = nullptr;
    KVCache kv_cache_;

    // Attention weights (always VRAM)
    struct AttentionWeights {
        float* attn_norm = nullptr;
        float* ffn_norm = nullptr;
        void* q_proj = nullptr;
        void* k_proj = nullptr;
        void* v_proj = nullptr;
        void* o_proj = nullptr;
    };
    std::vector<AttentionWeights> attn_weights_;

    // MoE per-layer state
    struct MoELayerState {
        float* gate_weight = nullptr;   // [num_experts, hidden_dim]
        void* shared_gate_proj = nullptr;
        void* shared_up_proj = nullptr;
        void* shared_down_proj = nullptr;
    };
    std::vector<MoELayerState> moe_state_;

    std::string expert_dir_;
    size_t expert_bytes_ = 0;

    // Embedding + LM head
    float* embedding_ = nullptr;
    void* lm_head_ = nullptr;
    float* final_norm_ = nullptr;

    // Pre-allocated scratch buffers (no per-token mallocs!)
    float* q_buf_ = nullptr;
    float* k_buf_ = nullptr;
    float* v_buf_ = nullptr;
    float* attn_out_ = nullptr;
    float* norm_buf_ = nullptr;
    float* gate_logits_ = nullptr;
    float* routing_weights_ = nullptr;
    int*   routing_indices_ = nullptr;
    float* expert_outputs_ = nullptr;  // [max_k, hidden_dim]
    float* shared_out_ = nullptr;

    // Pre-allocated expert compute buffers (double-buffered for overlap)
    float* expert_weight_buf_[2] = {};  // Two buffers for double-buffering
    size_t expert_weight_buf_size_ = 0; // Size of each buffer in bytes
    float* expert_gate_out_ = nullptr;  // [moe_intermediate]
    float* expert_up_out_ = nullptr;    // [moe_intermediate]
    float* shared_gate_out_ = nullptr;  // [moe_intermediate]
    float* shared_up_out_ = nullptr;    // [moe_intermediate]

    void allocate_buffers();
    void free_buffers();
};

} // namespace titan
