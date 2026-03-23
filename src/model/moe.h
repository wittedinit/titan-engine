#pragma once

#include "model/architecture.h"
#include "model/loader.h"
#include "model/tokenizer.h"
#include "inference/kv_cache.h"

namespace titan {

// ============================================================================
// MoE Transformer Executor
//
// Extends the dense executor with expert routing and 3-tier expert management:
// - Attention weights always in VRAM (same as dense)
// - Shared experts always in VRAM
// - Routed experts: cached in RAM with LRU, loaded from NVMe on miss
// - Expert forward: on GPU if in VRAM, on CPU if in RAM
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

    // Attention weights (always in VRAM, same as dense)
    struct AttentionWeights {
        float* attn_norm = nullptr;
        float* ffn_norm = nullptr;
        void* q_proj = nullptr;
        void* k_proj = nullptr;
        void* v_proj = nullptr;
        void* o_proj = nullptr;
    };
    std::vector<AttentionWeights> attn_weights_;

    // MoE-specific per layer
    struct MoELayerState {
        float* gate_weight = nullptr;   // [num_experts, hidden_dim] routing gate
        // Shared expert weights (always in VRAM)
        void* shared_gate_proj = nullptr;
        void* shared_up_proj = nullptr;
        void* shared_down_proj = nullptr;
    };
    std::vector<MoELayerState> moe_state_;

    // Expert weights path (for loading from NVMe/disk)
    std::string expert_dir_;
    size_t expert_bytes_ = 0;  // Bytes per expert

    // Embedding + LM head
    float* embedding_ = nullptr;
    void* lm_head_ = nullptr;
    float* final_norm_ = nullptr;

    // Scratch buffers
    float* q_buf_ = nullptr;
    float* k_buf_ = nullptr;
    float* v_buf_ = nullptr;
    float* attn_out_ = nullptr;
    float* norm_buf_ = nullptr;
    float* gate_logits_ = nullptr;     // [num_experts]
    float* routing_weights_ = nullptr; // [experts_per_tok]
    int*   routing_indices_ = nullptr; // [experts_per_tok]
    float* expert_outputs_ = nullptr;  // [experts_per_tok, hidden_dim]
    float* shared_out_ = nullptr;      // [hidden_dim]

    void allocate_buffers();
    void free_buffers();
};

} // namespace titan
