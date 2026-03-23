#include "inference/engine.h"
#include "model/dense.h"
#include "model/moe.h"
#include "compute/dispatch.h"
#include "core/logging.h"
#include "core/config.h"

#include <cuda_runtime.h>
#include <chrono>

namespace titan {

bool InferenceEngine::initialize(const RuntimeConfig& config) {
    config_ = config;

    // Detect hardware
    hw_ = detect_hardware();
    print_hardware_summary(hw_);

    // Initialize memory manager
    memory_ = std::make_unique<MemoryManager>(hw_, config_);

    initialized_ = true;
    LOG_INFO("Inference engine initialized");
    return true;
}

bool InferenceEngine::load_model(const std::string& model_path) {
    if (!initialized_) {
        LOG_ERROR("Engine not initialized — call initialize() first");
        return false;
    }

    // Load tokenizer
    if (!tokenizer_.load(model_path)) {
        LOG_ERROR("Failed to load tokenizer from %s", model_path.c_str());
        return false;
    }
    LOG_INFO("Tokenizer loaded: %d tokens, BOS=%d EOS=%d",
             tokenizer_.vocab_size(), tokenizer_.bos_id(), tokenizer_.eos_id());

    // Load model config to determine type
    ModelConfig model_config = load_model_config(model_path + "/config.json");

    // Create appropriate executor based on model type
    if (model_config.model_type == ModelType::MOE ||
        model_config.model_type == ModelType::HYBRID_MOE) {
        LOG_INFO("Detected MoE model — using MoE executor");
        auto executor = std::make_unique<MoEExecutor>();
        if (!executor->initialize(model_path, *memory_, config_)) {
            LOG_ERROR("Failed to initialize MoE executor");
            return false;
        }
        model_ = std::move(executor);
    } else {
        LOG_INFO("Using dense executor");
        auto executor = std::make_unique<DenseExecutor>();
        if (!executor->initialize(model_path, *memory_, config_)) {
            LOG_ERROR("Failed to initialize dense executor");
            return false;
        }
        model_ = std::move(executor);
    }

    // Generate execution plan
    plan_ = plan_execution(model_->config(), hw_, config_);

    LOG_INFO("Model loaded: %s (%.1fB params, %.1fB active/token)",
             model_->config().name.c_str(),
             model_->config().total_params() / 1e9,
             model_->config().active_params_per_token() / 1e9);
    LOG_INFO("Execution plan: VRAM=%.1f GB, RAM=%.1f GB",
             plan_.vram_used / 1e9, plan_.ram_used / 1e9);

    return true;
}

void InferenceEngine::generate(const std::string& prompt,
                                const SamplingParams& sampling,
                                TokenCallback on_token) {
    if (!model_) {
        LOG_ERROR("No model loaded");
        return;
    }

    const auto& cfg = model_->config();

    // Tokenize prompt
    auto prompt_tokens = tokenizer_.encode(prompt, true);
    LOG_INFO("Prompt: %zu tokens", prompt_tokens.size());

    // Allocate buffers on GPU
    float* hidden = nullptr;
    float* residual = nullptr;
    float* logits = nullptr;
    int* sampled_token_gpu = nullptr;

    size_t hidden_bytes = cfg.hidden_dim * sizeof(float);
    size_t logits_bytes = cfg.vocab_size * sizeof(float);

    cudaMalloc(&hidden, hidden_bytes);
    cudaMalloc(&residual, hidden_bytes);
    cudaMalloc(&logits, logits_bytes);
    cudaMalloc(&sampled_token_gpu, sizeof(int));

    if (!hidden || !residual || !logits || !sampled_token_gpu) {
        LOG_ERROR("Failed to allocate inference buffers");
        return;
    }

    cudaMemset(residual, 0, hidden_bytes);

    auto t_start = std::chrono::steady_clock::now();

    // Get executor for embedding access (works for both Dense and MoE)
    auto* dense = dynamic_cast<DenseExecutor*>(model_.get());
    auto* moe = dynamic_cast<MoEExecutor*>(model_.get());

    auto do_embed = [&](int token_id, float* buf) {
        if (dense) dense->embed_token(token_id, buf);
        else if (moe) moe->embed_token(token_id, buf);
    };

    // --- Prefill Phase: Process prompt tokens ---
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        // Embedding lookup
        do_embed(prompt_tokens[i], hidden);

        // Copy to residual stream
        cudaMemcpy(residual, hidden, hidden_bytes, cudaMemcpyDeviceToDevice);

        // Forward through all layers
        for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
            model_->forward_layer(hidden, residual, layer, (int)i, nullptr);
        }
    }

    auto t_prefill = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();

    if (!prompt_tokens.empty()) {
        LOG_INFO("Prefill: %zu tokens in %.0f ms (%.0f tok/s)",
                 prompt_tokens.size(), prefill_ms,
                 prompt_tokens.size() / (prefill_ms / 1000.0));
    }

    // --- Decode Phase: Generate tokens autoregressively ---
    int position = (int)prompt_tokens.size();
    uint64_t rng_state = sampling.seed;
    if (rng_state == 0) {
        rng_state = std::chrono::steady_clock::now().time_since_epoch().count();
    }

    int tokens_generated = 0;
    auto t_decode_start = std::chrono::steady_clock::now();

    for (uint32_t step = 0; step < sampling.max_tokens; step++) {
        auto t_tok = std::chrono::steady_clock::now();

        // Compute logits from final hidden state
        model_->compute_logits(hidden, logits, nullptr);

        // Sample next token
        cuda::sample_token(logits, sampled_token_gpu, cfg.vocab_size,
                          sampling.temperature, sampling.top_p, sampling.top_k,
                          rng_state++, nullptr);

        // Copy token ID to CPU
        int token_id;
        cudaMemcpy(&token_id, sampled_token_gpu, sizeof(int), cudaMemcpyDeviceToHost);

        // Check for EOS
        if (token_id == tokenizer_.eos_id()) {
            LOG_DEBUG("EOS token generated at step %d", step);
            break;
        }

        tokens_generated++;

        // Decode and callback
        std::string text = tokenizer_.decode(token_id);
        if (on_token) {
            on_token(token_id, text);
        }

        // Embed the new token for next step
        do_embed(token_id, hidden);
        cudaMemcpy(residual, hidden, hidden_bytes, cudaMemcpyDeviceToDevice);

        // Forward through all layers
        for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
            model_->forward_layer(hidden, residual, layer, position, nullptr);
        }
        position++;

        // Periodic stats
        if (tokens_generated % 20 == 0 && tokens_generated > 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_decode_start).count();
            LOG_DEBUG("  [%d tokens, %.1f tok/s]", tokens_generated,
                      tokens_generated / elapsed);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double decode_s = std::chrono::duration<double>(t_end - t_decode_start).count();

    LOG_INFO("Generated %d tokens in %.1f s (%.1f tok/s)",
             tokens_generated, decode_s,
             tokens_generated > 0 ? tokens_generated / decode_s : 0);

    // Cleanup
    cudaFree(hidden);
    cudaFree(residual);
    cudaFree(logits);
    cudaFree(sampled_token_gpu);
}

void InferenceEngine::print_stats() const {
    if (memory_) memory_->print_usage();
}

} // namespace titan
