#include "inference/engine.h"
#include "model/dense.h"
#include "model/moe.h"
#include "model/gguf_loader.h"
#include "compute/dispatch.h"
#include "core/logging.h"
#include "core/config.h"

#include <sys/stat.h>

#include <cuda_runtime.h>
#include <chrono>

namespace titan {

InferenceEngine::~InferenceEngine() {
    if (hidden_) cudaFree(hidden_);
    if (residual_) cudaFree(residual_);
    if (logits_) cudaFree(logits_);
    if (sampled_token_gpu_) cudaFree(sampled_token_gpu_);
}

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

    // Detect model format: GGUF (single file) vs HuggingFace (directory)
    bool is_gguf = false;
    {
        struct stat st;
        if (stat(model_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            // Single file — check if GGUF
            if (model_path.size() > 5 &&
                model_path.substr(model_path.size() - 5) == ".gguf") {
                is_gguf = true;
            }
        }
    }

    ModelConfig model_config;

    if (is_gguf) {
        LOG_INFO("Detected GGUF format: %s", model_path.c_str());
        GGUFLoader gguf;
        if (!gguf.load(model_path)) {
            LOG_ERROR("Failed to load GGUF file");
            return false;
        }
        model_config = gguf.to_model_config();
        LOG_INFO("GGUF model: %s (%uL, h=%u, %zu tensors)",
                 model_config.name.c_str(), model_config.num_layers,
                 model_config.hidden_dim, gguf.tensor_names().size());

        // GGUF tokenizer is embedded — use BOS=1 EOS=2 defaults for now
        // Full GGUF tokenizer extraction would parse the vocab from metadata
        LOG_WARN("GGUF tokenizer not yet extracted — using defaults (BOS=1, EOS=2)");
    } else {
        // HuggingFace directory format
        if (!tokenizer_.load(model_path)) {
            LOG_WARN("Tokenizer not found in %s — generation will work but text I/O limited",
                     model_path.c_str());
        } else {
            LOG_INFO("Tokenizer loaded: %d tokens, BOS=%d EOS=%d",
                     tokenizer_.vocab_size(), tokenizer_.bos_id(), tokenizer_.eos_id());
        }
        model_config = load_model_config(model_path + "/config.json");
    }

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

    // Pre-allocate inference buffers on GPU (never allocate during generation)
    const auto& mc = model_->config();
    size_t hidden_bytes = mc.hidden_dim * sizeof(float);
    size_t logits_bytes = mc.vocab_size * sizeof(float);

    cudaMalloc(&hidden_, hidden_bytes);
    cudaMalloc(&residual_, hidden_bytes);
    cudaMalloc(&logits_, logits_bytes);
    cudaMalloc(&sampled_token_gpu_, sizeof(int));

    if (!hidden_ || !residual_ || !logits_ || !sampled_token_gpu_) {
        LOG_ERROR("Failed to allocate pre-allocated inference buffers");
        return false;
    }

    // KV cache is initialized inside each executor's initialize() method,
    // with the correct head_dim for the model type (e.g., nope_head_dim for MLA).
    // DenseExecutor's KV cache is the exception — initialize it here since
    // DenseExecutor::initialize() doesn't do it internally.
    auto* dense = dynamic_cast<DenseExecutor*>(model_.get());
    if (dense) {
        dense->kv_cache().initialize(
            mc.num_layers, mc.num_kv_heads, mc.head_dim,
            config_.max_context_len);
    }
    // MoEExecutor already initialized its KV cache (with correct nope_head_dim for MLA)

    LOG_INFO("Model loaded: %s (%.1fB params, %.1fB active/token)",
             mc.name.c_str(),
             mc.total_params() / 1e9,
             mc.active_params_per_token() / 1e9);
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

    // Use pre-allocated GPU buffers (allocated once during load_model)
    float* hidden = hidden_;
    float* residual = residual_;
    float* logits = logits_;
    int* sampled_token_gpu = sampled_token_gpu_;

    size_t hidden_bytes = cfg.hidden_dim * sizeof(float);

    cudaMemset(residual, 0, hidden_bytes);

    auto t_start = std::chrono::steady_clock::now();

    // --- Prefill Phase: Process prompt tokens ---
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        // Embedding lookup (uses virtual method on ModelArchitecture)
        model_->embed_token(prompt_tokens[i], hidden);

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

        // Sync and check for CUDA errors before sampling (first token only)
        if (step == 0) {
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                LOG_ERROR("CUDA error before sampling step 0: %s", cudaGetErrorString(err));
            }
            // Log first few logit values for diagnostics
            float logit_sample[4] = {};
            cudaMemcpy(logit_sample, logits, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            LOG_DEBUG("logits[0..3] = %.4f %.4f %.4f %.4f",
                      logit_sample[0], logit_sample[1], logit_sample[2], logit_sample[3]);
        }

        // Sample next token
        cuda::sample_token(logits, sampled_token_gpu, cfg.vocab_size,
                          sampling.temperature, sampling.top_p, sampling.top_k,
                          rng_state++, nullptr);

        // Sync and check for CUDA errors after sampling (first token only)
        if (step == 0) {
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                LOG_ERROR("CUDA error after sampling step 0: %s", cudaGetErrorString(err));
            }
        }

        // Copy token ID to CPU
        int token_id = -1;
        cudaError_t memcpy_err = cudaMemcpy(&token_id, sampled_token_gpu, sizeof(int), cudaMemcpyDeviceToHost);
        if (memcpy_err != cudaSuccess) {
            LOG_ERROR("cudaMemcpy failed for token_id at step %u: %s", step, cudaGetErrorString(memcpy_err));
            break;
        }

        // Check for EOS
        if (token_id == tokenizer_.eos_id()) {
            LOG_DEBUG("EOS token generated at step %d", step);
            break;
        }

        tokens_generated++;

        // Decode and callback
        std::string text = tokenizer_.decode(token_id);
        LOG_DEBUG("tok[%d] id=%d text='%s'", tokens_generated, token_id, text.c_str());
        if (on_token) {
            on_token(token_id, text);
        }

        // Embed the new token for next step (uses virtual method)
        model_->embed_token(token_id, hidden);
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
}

void InferenceEngine::print_stats() const {
    if (memory_) memory_->print_usage();
}

} // namespace titan
