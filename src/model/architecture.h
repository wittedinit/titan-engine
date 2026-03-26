#pragma once

#include "core/types.h"
#include "memory/memory_manager.h"
#include <cuda_runtime.h>

namespace titan {

// ============================================================================
// Abstract Model Architecture Interface
//
// Each model architecture (Dense, MoE, Hybrid) implements this interface.
// The inference engine calls these methods without knowing the specifics.
// ============================================================================

class ModelArchitecture {
public:
    virtual ~ModelArchitecture() = default;

    // Get model configuration
    virtual const ModelConfig& config() const = 0;

    // Initialize: load weights, set up buffers
    virtual bool initialize(const std::string& model_path,
                           MemoryManager& memory,
                           const RuntimeConfig& runtime) = 0;

    // Forward pass for a single layer
    // hidden: [hidden_dim] input hidden state (modified in-place or returned)
    // residual: [hidden_dim] residual stream
    // layer_id: which layer to execute
    // position: current sequence position (for RoPE)
    virtual void forward_layer(float* hidden, float* residual,
                               uint32_t layer_id, int position,
                               cudaStream_t cuda_stream) = 0;

    // Compute logits from final hidden state
    virtual void compute_logits(const float* hidden, float* logits,
                                cudaStream_t cuda_stream) = 0;

    // Get memory requirements for planning
    virtual size_t attention_weight_bytes(uint32_t layer_id) const = 0;
    virtual size_t ffn_weight_bytes(uint32_t layer_id) const = 0;
    virtual size_t expert_weight_bytes(uint32_t layer_id, uint32_t expert_id) const = 0;
    virtual size_t kv_cache_bytes_per_token(uint32_t layer_id) const = 0;

    // Update KV cache with new key/value for given position
    virtual void update_kv_cache(uint32_t layer_id, int position,
                                 const float* key, const float* value) = 0;

    // Embedding lookup: map token_id to hidden state vector on GPU
    virtual void embed_token(int token_id, float* output,
                             cudaStream_t stream = nullptr) = 0;
};

} // namespace titan
