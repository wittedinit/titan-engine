#pragma once

#include "core/types.h"
#include <vector>
#include <string>
#include <unordered_set>

namespace titan {

// ============================================================================
// Activation Sparsity Profile
//
// Records which neurons in each FFN layer are "hot" (frequently activated)
// vs "cold" (rarely activated above threshold). Used at inference time to
// skip cold neurons for 2-5x speedup on dense models.
// ============================================================================

struct NeuronProfile {
    uint32_t    layer_id;
    uint32_t    inter_dim;
    float       threshold;          // Activation magnitude threshold

    // Per-neuron statistics (from profiling)
    std::vector<float>    mean_magnitude;   // [inter_dim]
    std::vector<float>    activation_rate;  // [inter_dim] — fraction of tokens where |act| > threshold
    std::vector<bool>     is_hot;           // [inter_dim] — true if activation_rate > hot_threshold

    // Derived indices for fast lookup
    std::vector<int>      hot_indices;      // Sorted indices of hot neurons
    std::vector<int>      cold_indices;     // Sorted indices of cold neurons

    float sparsity() const {
        return 1.0f - (float)hot_indices.size() / inter_dim;
    }
};

struct SparsityProfile {
    std::string     model_name;
    uint32_t        num_layers = 0;
    float           activation_threshold = 0.01f;  // |activation| < this = "zero"
    float           hot_threshold = 0.3f;          // Neurons active > 30% of time = "hot"
    uint32_t        num_calibration_tokens = 0;

    std::vector<NeuronProfile> layers;

    // Summary stats
    float avg_sparsity() const {
        if (layers.empty()) return 0;
        float sum = 0;
        for (const auto& l : layers) sum += l.sparsity();
        return sum / layers.size();
    }

    float estimated_speedup() const {
        float s = avg_sparsity();
        // Speedup = 1 / (1 - sparsity * efficiency)
        // efficiency ~0.85 (overhead of sparse indexing)
        return 1.0f / (1.0f - s * 0.85f);
    }

    // Save/load profile to disk (avoid re-profiling)
    bool save(const std::string& path) const;
    bool load(const std::string& path);
};

// ============================================================================
// Activation Sparsity Profiler
//
// Run calibration data through the model, recording neuron activation
// magnitudes. Produces a SparsityProfile.
// ============================================================================

class SparsityProfiler {
public:
    struct Config {
        float   activation_threshold = 0.01f;
        float   hot_threshold = 0.3f;
        int     num_calibration_tokens = 1024;
        int     calibration_batch_size = 32;
    };

    SparsityProfiler(const Config& config) : config_(config) {}

    // Profile a model (requires running forward passes on calibration data)
    // calibration_tokens: list of token IDs for calibration
    // forward_fn: function that runs one layer and returns FFN activations
    // Returns the sparsity profile
    SparsityProfile profile(
        const ModelConfig& model_config,
        const std::vector<int>& calibration_tokens
    );

    // Quick estimate without full profiling (uses heuristics based on model architecture)
    SparsityProfile estimate(const ModelConfig& model_config);

private:
    Config config_;
};

// ============================================================================
// Activation Predictor
//
// Lightweight MLP that predicts which neurons will be active given the
// current hidden state. Trained on profiling data.
//
// Architecture: hidden_dim → inter_dim (single linear + sigmoid)
// Size: hidden_dim * inter_dim * 4 bytes ≈ 67MB for 4096×4096
// (small compared to the FFN weights it replaces)
// ============================================================================

struct ActivationPredictor {
    uint32_t    layer_id;
    uint32_t    hidden_dim;
    uint32_t    inter_dim;

    // Predictor weights (on GPU)
    float*      weight = nullptr;   // [inter_dim, hidden_dim]
    float*      bias = nullptr;     // [inter_dim]

    // Prediction threshold (calibrated per-layer)
    float       threshold = 0.5f;

    // Runtime buffers (on GPU)
    int*        active_indices = nullptr;  // [max_active]
    int*        num_active = nullptr;      // [1]
    int         max_active = 0;

    // Accuracy from validation (how often prediction matches actual activation)
    float       precision = 0.0f;   // Of predicted active, how many actually active
    float       recall = 0.0f;      // Of actually active, how many predicted

    bool initialized() const { return weight != nullptr; }
};

// ============================================================================
// Sparse FFN Executor
//
// Replaces the standard dense FFN forward pass with a sparse version:
// 1. Predict active neurons (cheap linear projection)
// 2. Compute gate/up projections only for active neurons
// 3. SwiGLU only on active neurons
// 4. Down projection gathers from active neurons only
//
// For 80% sparsity: ~3.5x speedup on FFN (which is 60-70% of total compute)
// Overall model speedup: ~2-2.5x
// ============================================================================

class SparseFfnExecutor {
public:
    SparseFfnExecutor() = default;

    // Initialize from a sparsity profile
    bool initialize(const SparsityProfile& profile, int gpu_id = 0);

    // Run sparse FFN forward pass for one layer
    // hidden: [hidden_dim] input (on GPU)
    // output: [hidden_dim] output (on GPU)
    void forward(
        float* output,
        const float* hidden,
        const void* gate_proj,     // Quantized gate projection weights
        const void* up_proj,       // Quantized up projection weights
        const void* down_proj,     // Quantized down projection weights
        uint32_t layer_id,
        DType weight_dtype,
        int group_size,
        void* cuda_stream
    );

    // Get sparsity stats for the last forward pass
    struct Stats {
        int     total_neurons;
        int     active_neurons;
        float   sparsity;
        float   time_ms;
    };
    Stats last_stats() const { return last_stats_; }

private:
    std::vector<ActivationPredictor> predictors_;
    SparsityProfile profile_;
    Stats last_stats_ = {};

    // Scratch buffers (on GPU)
    float* gate_out_ = nullptr;
    float* up_out_ = nullptr;
    float* act_out_ = nullptr;
    size_t scratch_size_ = 0;
};

} // namespace titan
