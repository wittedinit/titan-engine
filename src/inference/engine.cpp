#include "core/types.h"
#include "core/hardware.h"
#include "core/logging.h"
#include "memory/memory_manager.h"
#include "model/architecture.h"

#include <chrono>
#include <functional>
#include <vector>

namespace titan {

// ============================================================================
// Inference Engine — Main orchestrator
//
// Manages the token generation loop:
// 1. Encode input tokens
// 2. For each generated token:
//    a. Run all layers (attention + FFN/MoE)
//    b. Compute logits
//    c. Sample next token
//    d. Update KV cache
// 3. Stream output tokens
// ============================================================================

class InferenceEngine {
public:
    InferenceEngine(const RuntimeConfig& config)
        : config_(config) {}

    bool initialize() {
        // Detect hardware
        hw_ = detect_hardware();
        print_hardware_summary(hw_);

        // Initialize memory manager
        memory_ = std::make_unique<MemoryManager>(hw_, config_);

        // Generate execution plan
        // (model architecture must be loaded first)

        LOG_INFO("Inference engine initialized");
        return true;
    }

    void set_model(std::unique_ptr<ModelArchitecture> model) {
        model_ = std::move(model);

        // Generate execution plan
        plan_ = plan_execution(model_->config(), hw_, config_);

        LOG_INFO("Execution plan: VRAM=%.1f GB, RAM=%.1f GB, NVMe=%s",
                 plan_.vram_used / 1e9, plan_.ram_used / 1e9,
                 plan_.nvme_used > 0 ? "active" : "none");
    }

    // Generate tokens from a prompt
    using TokenCallback = std::function<void(int token_id, const std::string& text)>;

    void generate(const std::vector<int>& prompt_tokens,
                  const SamplingParams& sampling,
                  TokenCallback on_token) {
        if (!model_) {
            LOG_ERROR("No model loaded");
            return;
        }

        const auto& cfg = model_->config();
        int position = 0;

        // Allocate scratch buffers
        size_t hidden_bytes = cfg.hidden_dim * sizeof(float);
        size_t logits_bytes = cfg.vocab_size * sizeof(float);

        float* hidden = (float*)memory_->vram().allocate(hidden_bytes);
        float* residual = (float*)memory_->vram().allocate(hidden_bytes);
        float* logits = (float*)memory_->vram().allocate(logits_bytes);
        int* sampled_token_gpu = (int*)memory_->vram().allocate(sizeof(int));

        if (!hidden || !residual || !logits) {
            LOG_ERROR("Cannot allocate inference buffers");
            return;
        }

        auto t_start = std::chrono::steady_clock::now();
        int tokens_generated = 0;

        // Prefill: process prompt tokens
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            // TODO: Batch prefill for efficiency
            // For now, process one token at a time
            // Embedding lookup would go here
            position = i;

            for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
                model_->forward_layer(hidden, residual, layer, position, nullptr);
            }
        }

        auto t_prefill = std::chrono::steady_clock::now();
        double prefill_time = std::chrono::duration<double>(t_prefill - t_start).count();
        LOG_INFO("Prefill: %zu tokens in %.1f ms (%.0f tok/s)",
                 prompt_tokens.size(), prefill_time * 1000,
                 prompt_tokens.size() / prefill_time);

        // Decode: generate tokens autoregressively
        uint64_t rng_state = sampling.seed ? sampling.seed :
            std::chrono::steady_clock::now().time_since_epoch().count();

        for (uint32_t step = 0; step < sampling.max_tokens; step++) {
            auto t_tok_start = std::chrono::steady_clock::now();

            // Forward pass through all layers
            for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
                model_->forward_layer(hidden, residual, layer, position, nullptr);
            }

            // Compute logits
            model_->compute_logits(hidden, logits, nullptr);

            // Sample next token
            cuda::sample_token(logits, sampled_token_gpu, cfg.vocab_size,
                              sampling.temperature, sampling.top_p, sampling.top_k,
                              rng_state++, nullptr);

            // Copy token ID back to CPU
            int token_id;
            cudaMemcpy(&token_id, sampled_token_gpu, sizeof(int), cudaMemcpyDeviceToHost);

            // Check for EOS
            // TODO: Get EOS token ID from tokenizer
            if (token_id == 2) break; // Typical EOS

            tokens_generated++;
            position++;

            // Callback
            if (on_token) {
                on_token(token_id, ""); // Text decoding handled by caller
            }

            auto t_tok_end = std::chrono::steady_clock::now();
            double tok_time = std::chrono::duration<double>(t_tok_end - t_tok_start).count();

            if (tokens_generated % 10 == 0) {
                auto total_time = std::chrono::duration<double>(t_tok_end - t_prefill).count();
                LOG_DEBUG("Token %d: %.1f ms (avg: %.1f tok/s)",
                          tokens_generated, tok_time * 1000,
                          tokens_generated / total_time);
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        double total_time = std::chrono::duration<double>(t_end - t_prefill).count();
        LOG_INFO("Generated %d tokens in %.1f s (%.1f tok/s)",
                 tokens_generated, total_time,
                 tokens_generated / total_time);

        // Print memory stats
        memory_->print_usage();

        // Cleanup
        memory_->vram().free(hidden);
        memory_->vram().free(residual);
        memory_->vram().free(logits);
        memory_->vram().free(sampled_token_gpu);
    }

private:
    RuntimeConfig config_;
    HardwareProfile hw_;
    std::unique_ptr<MemoryManager> memory_;
    std::unique_ptr<ModelArchitecture> model_;
    ExecutionPlan plan_;
};

} // namespace titan
