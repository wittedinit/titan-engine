#pragma once

#include "core/types.h"
#include "model/architecture.h"
#include "model/tokenizer.h"
#include "inference/kv_cache.h"

#include <vector>
#include <memory>
#include <functional>

namespace titan {

// ============================================================================
// Speculative Decoding
//
// Three strategies supported:
//
// 1. Draft Model: A small model (e.g., 1B) generates N candidate tokens,
//    the target model verifies them in a single forward pass.
//    Speedup: ~2-3x (N accepted + 1 new per batch forward).
//
// 2. Self-Speculative: For MoE models, use fewer experts (top-1 instead of
//    top-K) as a cheap draft. Same model, different quality/speed tradeoff.
//    Speedup: ~1.5-2x (no separate model needed).
//
// 3. Medusa Heads: Parallel token prediction from intermediate hidden states
//    using lightweight prediction heads. Requires model-specific heads.
//    Speedup: ~2-3x with low memory overhead.
//
// Verification uses the standard rejection sampling algorithm:
// - Run target model on all draft tokens in one batch
// - Accept tokens where target agrees with draft
// - Reject first disagreement, sample from target at that position
// - Expected acceptance rate: 70-90% for well-matched draft models
// ============================================================================

enum class SpeculativeMethod {
    NONE,
    DRAFT_MODEL,
    SELF_SPECULATIVE,
    MEDUSA_HEADS,
};

struct SpeculativeConfig {
    SpeculativeMethod method = SpeculativeMethod::DRAFT_MODEL;
    int num_draft_tokens = 5;       // How many tokens to draft per step
    float acceptance_threshold = 0.0f; // Min probability ratio for acceptance (0 = standard)

    // Draft model config (for DRAFT_MODEL method)
    std::string draft_model_path;

    // Self-speculative config (for MoE models)
    int draft_experts_per_tok = 1;  // Use fewer experts for drafting

    // Medusa config
    int num_medusa_heads = 3;       // Number of parallel prediction heads
};

class SpeculativeDecoder {
public:
    SpeculativeDecoder() = default;
    ~SpeculativeDecoder();

    // Initialize with target model and speculative config
    bool initialize(ModelArchitecture* target_model,
                    MemoryManager& memory,
                    const RuntimeConfig& runtime,
                    const SpeculativeConfig& spec_config);

    // Generate tokens with speculative decoding
    // Returns the number of tokens generated (may be > 1 per call)
    using TokenCallback = std::function<void(int token_id, const std::string& text)>;

    int generate_step(
        float* hidden,          // Current hidden state (GPU)
        float* residual,        // Residual stream (GPU)
        float* logits,          // Logits buffer (GPU, [vocab_size])
        int position,           // Current sequence position
        const SamplingParams& sampling,
        const Tokenizer& tokenizer,
        TokenCallback on_token,
        cudaStream_t cuda_stream = nullptr
    );

    // Stats
    struct Stats {
        int total_draft_tokens = 0;
        int accepted_tokens = 0;
        int rejected_tokens = 0;
        int total_steps = 0;
        float acceptance_rate() const {
            return total_draft_tokens > 0
                ? (float)accepted_tokens / total_draft_tokens : 0;
        }
        float tokens_per_step() const {
            return total_steps > 0
                ? (float)(accepted_tokens + total_steps) / total_steps : 1;
        }
    };
    Stats stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

private:
    ModelArchitecture* target_ = nullptr;
    SpeculativeConfig config_;
    Stats stats_;

    // Draft model (separate small model)
    std::unique_ptr<ModelArchitecture> draft_model_;
    std::unique_ptr<KVCache> draft_kv_cache_;

    // Buffers for draft tokens
    std::vector<int> draft_tokens_;
    float* draft_hidden_ = nullptr;
    float* draft_residual_ = nullptr;
    float* draft_logits_ = nullptr;
    float* target_logits_batch_ = nullptr; // [num_draft+1, vocab_size]

    // Pre-allocated GPU buffer for sampling output (avoids host pointer to GPU kernel)
    int* d_sampled_token_ = nullptr;

    // Pre-allocated buffers for generate_step (avoids per-step cudaMalloc)
    float* verify_hidden_ = nullptr;     // [hidden_dim] copy for target verification
    float* verify_residual_ = nullptr;   // [hidden_dim] copy for target verification
    int* d_verify_token_ = nullptr;      // GPU int for sampling at rejection point

    // Verification
    int verify_and_accept(const float* target_logits_batch,
                          const std::vector<int>& draft_tokens,
                          int vocab_size,
                          const SamplingParams& sampling,
                          uint64_t& rng_state);

    // Draft model methods
    int draft_with_model(float* hidden, float* residual, int position,
                         const SamplingParams& sampling, cudaStream_t stream);
    int draft_self_speculative(float* hidden, float* residual, int position,
                               const SamplingParams& sampling, cudaStream_t stream);
};

} // namespace titan
