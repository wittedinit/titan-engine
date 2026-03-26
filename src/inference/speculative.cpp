#include "inference/speculative.h"
#include "model/dense.h"
#include "compute/dispatch.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <random>

namespace titan {

SpeculativeDecoder::~SpeculativeDecoder() {
    if (draft_hidden_) cudaFree(draft_hidden_);
    if (draft_residual_) cudaFree(draft_residual_);
    if (draft_logits_) cudaFree(draft_logits_);
    if (target_logits_batch_) cudaFree(target_logits_batch_);
    if (d_sampled_token_) cudaFree(d_sampled_token_);
    if (verify_hidden_) cudaFree(verify_hidden_);
    if (verify_residual_) cudaFree(verify_residual_);
    if (d_verify_token_) cudaFree(d_verify_token_);
}

bool SpeculativeDecoder::initialize(ModelArchitecture* target,
                                     MemoryManager& memory,
                                     const RuntimeConfig& runtime,
                                     const SpeculativeConfig& spec_config) {
    target_ = target;
    config_ = spec_config;

    uint32_t vocab = target->config().vocab_size;
    uint32_t hd = target->config().hidden_dim;
    int N = config_.num_draft_tokens;

    // Allocate draft buffers
    cudaMalloc(&draft_hidden_, hd * sizeof(float));
    cudaMalloc(&draft_residual_, hd * sizeof(float));
    cudaMalloc(&draft_logits_, vocab * sizeof(float));
    cudaMalloc(&target_logits_batch_, (N + 1) * vocab * sizeof(float));
    cudaMalloc(&d_sampled_token_, sizeof(int));
    cudaMalloc(&verify_hidden_, hd * sizeof(float));
    cudaMalloc(&verify_residual_, hd * sizeof(float));
    cudaMalloc(&d_verify_token_, sizeof(int));

    draft_tokens_.reserve(N);

    if (config_.method == SpeculativeMethod::DRAFT_MODEL &&
        !config_.draft_model_path.empty()) {
        // Load the separate draft model
        auto draft = std::make_unique<DenseExecutor>();
        if (!draft->initialize(config_.draft_model_path, memory, runtime)) {
            LOG_ERROR("Failed to load draft model from %s",
                      config_.draft_model_path.c_str());
            return false;
        }
        draft_model_ = std::move(draft);

        draft_kv_cache_ = std::make_unique<KVCache>();
        draft_kv_cache_->initialize(
            draft_model_->config().num_layers,
            draft_model_->config().num_kv_heads,
            draft_model_->config().head_dim,
            runtime.max_context_len
        );

        LOG_INFO("Speculative decoding: draft model loaded (%.1fB params, N=%d)",
                 draft_model_->config().total_params() / 1e9, N);
    } else if (config_.method == SpeculativeMethod::SELF_SPECULATIVE) {
        LOG_INFO("Speculative decoding: self-speculative (K=%d draft experts, N=%d)",
                 config_.draft_experts_per_tok, N);
    } else {
        LOG_INFO("Speculative decoding: N=%d draft tokens", N);
    }

    return true;
}

// ============================================================================
// Draft Token Generation
// ============================================================================

int SpeculativeDecoder::draft_with_model(float* hidden, float* residual,
                                          int position,
                                          const SamplingParams& sampling,
                                          cudaStream_t stream) {
    if (!draft_model_) return 0;

    draft_tokens_.clear();
    int N = config_.num_draft_tokens;
    uint32_t vocab = draft_model_->config().vocab_size;
    uint32_t hd = draft_model_->config().hidden_dim;

    // Copy current state to draft buffers
    cudaMemcpy(draft_hidden_, hidden, hd * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(draft_residual_, residual, hd * sizeof(float), cudaMemcpyDeviceToDevice);

    uint64_t rng = position * 6364136223846793005ULL + 1;

    for (int i = 0; i < N; i++) {
        // Forward pass through draft model
        for (uint32_t layer = 0; layer < draft_model_->config().num_layers; layer++) {
            draft_model_->forward_layer(draft_hidden_, draft_residual_,
                                         layer, position + i, stream);
        }

        // Get logits and sample
        draft_model_->compute_logits(draft_hidden_, draft_logits_, stream);

        cuda::sample_token(draft_logits_, d_sampled_token_, vocab,
                          sampling.temperature, sampling.top_p, sampling.top_k,
                          rng++, stream);

        // Copy token to CPU
        int h_token;
        cudaMemcpy(&h_token, d_sampled_token_, sizeof(int), cudaMemcpyDeviceToHost);
        draft_tokens_.push_back(h_token);

        // Embed for next step (uses virtual method on ModelArchitecture)
        if (i < N - 1) {
            draft_model_->embed_token(h_token, draft_hidden_);
            cudaMemcpy(draft_residual_, draft_hidden_, hd * sizeof(float),
                        cudaMemcpyDeviceToDevice);
        }
    }

    stats_.total_draft_tokens += N;
    return N;
}

int SpeculativeDecoder::draft_self_speculative(float* hidden, float* residual,
                                                int position,
                                                const SamplingParams& sampling,
                                                cudaStream_t stream) {
    // Self-speculative: run the same model with fewer experts
    // This is a simplified version — full implementation would modify
    // the MoE executor's experts_per_tok on the fly
    draft_tokens_.clear();
    int N = config_.num_draft_tokens;
    uint32_t vocab = target_->config().vocab_size;
    uint32_t hd = target_->config().hidden_dim;

    cudaMemcpy(draft_hidden_, hidden, hd * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(draft_residual_, residual, hd * sizeof(float), cudaMemcpyDeviceToDevice);

    uint64_t rng = position * 6364136223846793005ULL + 1;

    for (int i = 0; i < N; i++) {
        // Use target model but with reduced expert count
        // (In production, the MoE executor would have a set_draft_mode() method)
        for (uint32_t layer = 0; layer < target_->config().num_layers; layer++) {
            target_->forward_layer(draft_hidden_, draft_residual_,
                                    layer, position + i, stream);
        }

        target_->compute_logits(draft_hidden_, draft_logits_, stream);

        cuda::sample_token(draft_logits_, d_sampled_token_, vocab,
                          sampling.temperature, sampling.top_p, sampling.top_k,
                          rng++, stream);

        int h_token;
        cudaMemcpy(&h_token, d_sampled_token_, sizeof(int), cudaMemcpyDeviceToHost);
        draft_tokens_.push_back(h_token);
    }

    stats_.total_draft_tokens += N;
    return N;
}

// ============================================================================
// Verification: Run target model on all draft tokens, accept/reject
// ============================================================================

int SpeculativeDecoder::verify_and_accept(const float* target_logits_batch,
                                           const std::vector<int>& draft_tokens,
                                           int vocab_size,
                                           const SamplingParams& sampling,
                                           uint64_t& rng_state) {
    // Standard speculative decoding rejection sampling:
    // For each draft token t_i:
    //   p_target = softmax(target_logits[i])[t_i]
    //   p_draft  = softmax(draft_logits[i])[t_i]
    //   accept with probability min(1, p_target / p_draft)

    // Simplified: accept if target's top-1 matches draft token
    // (This is a conservative approximation — full implementation
    //  would use the proper probability ratio)
    int accepted = 0;

    for (size_t i = 0; i < draft_tokens.size(); i++) {
        const float* logits = target_logits_batch + i * vocab_size;

        // Find argmax of target logits (on CPU)
        // In production, this should be done on GPU
        std::vector<float> h_logits(vocab_size);
        cudaMemcpy(h_logits.data(), logits, vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost);

        int target_token = 0;
        float max_val = h_logits[0];
        for (int j = 1; j < vocab_size; j++) {
            if (h_logits[j] > max_val) {
                max_val = h_logits[j];
                target_token = j;
            }
        }

        if (target_token == draft_tokens[i]) {
            accepted++;
        } else {
            // First rejection — stop here
            break;
        }
    }

    stats_.accepted_tokens += accepted;
    stats_.rejected_tokens += (int)draft_tokens.size() - accepted;
    return accepted;
}

// ============================================================================
// Main Generation Step
// ============================================================================

int SpeculativeDecoder::generate_step(
    float* hidden, float* residual, float* logits,
    int position, const SamplingParams& sampling,
    const Tokenizer& tokenizer, TokenCallback on_token,
    cudaStream_t cuda_stream
) {
    stats_.total_steps++;
    int N = config_.num_draft_tokens;
    uint32_t vocab = target_->config().vocab_size;

    // Step 1: Generate N draft tokens
    int num_drafted = 0;
    switch (config_.method) {
        case SpeculativeMethod::DRAFT_MODEL:
            num_drafted = draft_with_model(hidden, residual, position,
                                           sampling, cuda_stream);
            break;
        case SpeculativeMethod::SELF_SPECULATIVE:
            num_drafted = draft_self_speculative(hidden, residual, position,
                                                  sampling, cuda_stream);
            break;
        default:
            // No speculative decoding — generate one token normally
            return 0;
    }

    if (num_drafted == 0) return 0;

    // Step 2: Run target model on all draft positions in batch
    // (In production, this would be a batched forward pass)
    // For now, run sequentially and collect logits
    float* h_copy = verify_hidden_;
    float* r_copy = verify_residual_;
    uint32_t hd = target_->config().hidden_dim;
    cudaMemcpy(h_copy, hidden, hd * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(r_copy, residual, hd * sizeof(float), cudaMemcpyDeviceToDevice);

    for (int i = 0; i < num_drafted; i++) {
        if (i > 0) {
            target_->embed_token(draft_tokens_[i - 1], h_copy);
            cudaMemcpy(r_copy, h_copy, hd * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        for (uint32_t layer = 0; layer < target_->config().num_layers; layer++) {
            target_->forward_layer(h_copy, r_copy, layer, position + i, cuda_stream);
        }

        target_->compute_logits(h_copy, target_logits_batch_ + i * vocab, cuda_stream);
    }

    // Step 3: Verify and accept
    uint64_t rng = position;
    int accepted = verify_and_accept(target_logits_batch_, draft_tokens_,
                                      vocab, sampling, rng);

    // Step 4: Output accepted tokens
    int output_count = 0;
    for (int i = 0; i < accepted; i++) {
        if (on_token) {
            std::string text = tokenizer.decode(draft_tokens_[i]);
            on_token(draft_tokens_[i], text);
        }
        output_count++;
    }

    // Step 5: Sample one new token from target at the rejection point
    int new_token = -1;
    if (accepted < num_drafted) {
        // Sample from target logits at the rejected position
        cuda::sample_token(target_logits_batch_ + accepted * vocab, d_verify_token_,
                          vocab, sampling.temperature, sampling.top_p,
                          sampling.top_k, rng, (cudaStream_t)cuda_stream);
        cudaMemcpy(&new_token, d_verify_token_, sizeof(int), cudaMemcpyDeviceToHost);

        if (on_token) {
            std::string text = tokenizer.decode(new_token);
            on_token(new_token, text);
        }
        output_count++;
    }

    // Update hidden/residual to reflect accepted state
    // (The target model's KV cache already has the correct entries)
    if (output_count > 0) {
        // Bug #6 fix: After rejection, use the newly sampled token (new_token),
        // not the rejected draft token (draft_tokens_[accepted]).
        int last_token = (accepted < num_drafted)
            ? new_token                 // Use the resampled token, not the rejected draft
            : draft_tokens_.back();
        target_->embed_token(last_token, hidden);
        cudaMemcpy(residual, hidden, hd * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    LOG_DEBUG("Speculative step: drafted=%d accepted=%d output=%d (rate=%.0f%%)",
              num_drafted, accepted, output_count,
              stats_.acceptance_rate() * 100);

    return output_count;
}

} // namespace titan
