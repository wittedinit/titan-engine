#pragma once

#include "core/types.h"
#include "core/hardware.h"
#include "memory/memory_manager.h"
#include "model/architecture.h"
#include "model/tokenizer.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace titan {

// ============================================================================
// Inference Engine — Main entry point for text generation
// ============================================================================

class InferenceEngine {
public:
    InferenceEngine() = default;
    ~InferenceEngine() = default;

    // Initialize: detect hardware, load model, set up memory
    bool initialize(const RuntimeConfig& config);

    // Load a model from a HuggingFace directory
    bool load_model(const std::string& model_path);

    // Generate text from a prompt
    using TokenCallback = std::function<void(int token_id, const std::string& text)>;
    void generate(const std::string& prompt,
                  const SamplingParams& sampling,
                  TokenCallback on_token);

    // Access components
    const HardwareProfile& hardware() const { return hw_; }
    const ModelConfig& model_config() const { return model_->config(); }
    Tokenizer& tokenizer() { return tokenizer_; }

    // Print stats
    void print_stats() const;

private:
    RuntimeConfig config_;
    HardwareProfile hw_;
    std::unique_ptr<MemoryManager> memory_;
    std::unique_ptr<ModelArchitecture> model_;
    Tokenizer tokenizer_;
    ExecutionPlan plan_;
    bool initialized_ = false;
};

} // namespace titan
