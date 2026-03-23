#include "model/sparsity.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <numeric>

namespace titan {

// ============================================================================
// Sparsity Profile Save/Load
// ============================================================================

bool SparsityProfile::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f.good()) return false;

    // Header
    uint32_t magic = 0x54535052; // "TSPR" — Titan Sparsity Profile
    uint32_t version = 1;
    f.write((char*)&magic, 4);
    f.write((char*)&version, 4);
    f.write((char*)&num_layers, 4);
    f.write((char*)&activation_threshold, 4);
    f.write((char*)&hot_threshold, 4);
    f.write((char*)&num_calibration_tokens, 4);

    uint32_t name_len = model_name.size();
    f.write((char*)&name_len, 4);
    f.write(model_name.data(), name_len);

    // Per-layer data
    for (const auto& layer : layers) {
        f.write((char*)&layer.layer_id, 4);
        f.write((char*)&layer.inter_dim, 4);
        f.write((char*)&layer.threshold, 4);

        f.write((char*)layer.mean_magnitude.data(),
                layer.inter_dim * sizeof(float));
        f.write((char*)layer.activation_rate.data(),
                layer.inter_dim * sizeof(float));

        uint32_t num_hot = layer.hot_indices.size();
        f.write((char*)&num_hot, 4);
        f.write((char*)layer.hot_indices.data(), num_hot * sizeof(int));
    }

    LOG_INFO("Saved sparsity profile to %s (%u layers, %.0f%% avg sparsity)",
             path.c_str(), num_layers, avg_sparsity() * 100);
    return true;
}

bool SparsityProfile::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    uint32_t magic, version;
    f.read((char*)&magic, 4);
    f.read((char*)&version, 4);

    if (magic != 0x54535052 || version != 1) {
        LOG_ERROR("Invalid sparsity profile: %s", path.c_str());
        return false;
    }

    f.read((char*)&num_layers, 4);
    f.read((char*)&activation_threshold, 4);
    f.read((char*)&hot_threshold, 4);
    f.read((char*)&num_calibration_tokens, 4);

    uint32_t name_len;
    f.read((char*)&name_len, 4);
    model_name.resize(name_len);
    f.read(model_name.data(), name_len);

    layers.resize(num_layers);
    for (uint32_t l = 0; l < num_layers; l++) {
        auto& layer = layers[l];
        f.read((char*)&layer.layer_id, 4);
        f.read((char*)&layer.inter_dim, 4);
        f.read((char*)&layer.threshold, 4);

        layer.mean_magnitude.resize(layer.inter_dim);
        layer.activation_rate.resize(layer.inter_dim);
        f.read((char*)layer.mean_magnitude.data(), layer.inter_dim * sizeof(float));
        f.read((char*)layer.activation_rate.data(), layer.inter_dim * sizeof(float));

        uint32_t num_hot;
        f.read((char*)&num_hot, 4);
        layer.hot_indices.resize(num_hot);
        f.read((char*)layer.hot_indices.data(), num_hot * sizeof(int));

        // Rebuild cold indices and is_hot
        layer.is_hot.resize(layer.inter_dim, false);
        for (int idx : layer.hot_indices) {
            layer.is_hot[idx] = true;
        }
        for (uint32_t i = 0; i < layer.inter_dim; i++) {
            if (!layer.is_hot[i]) layer.cold_indices.push_back(i);
        }
    }

    LOG_INFO("Loaded sparsity profile: %s (%u layers, %.0f%% avg sparsity)",
             model_name.c_str(), num_layers, avg_sparsity() * 100);
    return true;
}

// ============================================================================
// Quick Sparsity Estimation (heuristic, no forward passes needed)
// ============================================================================

SparsityProfile SparsityProfiler::estimate(const ModelConfig& model_config) {
    SparsityProfile profile;
    profile.model_name = model_config.name;
    profile.num_layers = model_config.num_layers;
    profile.activation_threshold = config_.activation_threshold;
    profile.hot_threshold = config_.hot_threshold;

    // Heuristic sparsity estimates based on architecture
    // These are empirical observations from PowerInfer and related work:
    //
    // - SwiGLU models (Llama, Mistral): ~85-92% sparsity in FFN
    // - GELU models (GPT-2/3 style): ~70-80% sparsity
    // - ReLU models (older): ~90-95% sparsity (ReLU is naturally sparse)
    // - MoE models: already sparse by design, less benefit from neuron-level sparsity

    float estimated_sparsity;
    switch (model_config.activation) {
        case ActivationType::SWIGLU:
        case ActivationType::SILU:
            estimated_sparsity = 0.88f; // SwiGLU: ~88% neurons near-zero
            break;
        case ActivationType::GELU:
            estimated_sparsity = 0.75f;
            break;
        case ActivationType::RELU:
            estimated_sparsity = 0.92f; // ReLU is naturally sparse
            break;
        default:
            estimated_sparsity = 0.80f;
    }

    uint32_t inter_dim = model_config.intermediate_dim;

    for (uint32_t l = 0; l < model_config.num_layers; l++) {
        NeuronProfile np;
        np.layer_id = l;
        np.inter_dim = inter_dim;
        np.threshold = config_.activation_threshold;

        // Estimate: first and last layers tend to be less sparse
        float layer_sparsity = estimated_sparsity;
        if (l < 2 || l >= model_config.num_layers - 2) {
            layer_sparsity *= 0.9f; // 10% less sparse at boundaries
        }

        uint32_t num_hot = (uint32_t)(inter_dim * (1.0f - layer_sparsity));

        // Generate placeholder hot indices (uniform distribution)
        // Real profiling would identify actual hot neurons
        np.is_hot.resize(inter_dim, false);
        np.mean_magnitude.resize(inter_dim, 0.0f);
        np.activation_rate.resize(inter_dim, 0.0f);

        // Mark the first `num_hot` neurons as hot (placeholder)
        for (uint32_t i = 0; i < num_hot && i < inter_dim; i++) {
            np.is_hot[i] = true;
            np.hot_indices.push_back(i);
            np.mean_magnitude[i] = 1.0f;
            np.activation_rate[i] = 0.5f;
        }
        for (uint32_t i = num_hot; i < inter_dim; i++) {
            np.cold_indices.push_back(i);
        }

        profile.layers.push_back(np);
    }

    LOG_INFO("Estimated sparsity profile for %s: %.0f%% avg (est. %.1fx speedup)",
             model_config.name.c_str(),
             profile.avg_sparsity() * 100,
             profile.estimated_speedup());
    LOG_WARN("This is a heuristic estimate — run full profiling for accuracy");

    return profile;
}

// ============================================================================
// Sparse FFN Executor
// ============================================================================

bool SparseFfnExecutor::initialize(const SparsityProfile& profile, int gpu_id) {
    profile_ = profile;
    cudaSetDevice(gpu_id);

    if (profile.layers.empty()) {
        LOG_ERROR("Empty sparsity profile");
        return false;
    }

    // Find max intermediate dimension
    uint32_t max_inter = 0;
    for (const auto& l : profile.layers) {
        max_inter = std::max(max_inter, l.inter_dim);
    }

    // Allocate scratch buffers
    scratch_size_ = max_inter * sizeof(float);
    cudaMalloc(&gate_out_, scratch_size_);
    cudaMalloc(&up_out_, scratch_size_);
    cudaMalloc(&act_out_, scratch_size_);

    // Zero them out
    cudaMemset(gate_out_, 0, scratch_size_);
    cudaMemset(up_out_, 0, scratch_size_);
    cudaMemset(act_out_, 0, scratch_size_);

    LOG_INFO("Sparse FFN executor initialized: %u layers, %.0f%% avg sparsity",
             (uint32_t)profile.layers.size(), profile.avg_sparsity() * 100);

    return true;
}

void SparseFfnExecutor::forward(
    float* output,
    const float* hidden,
    const void* gate_proj,
    const void* up_proj,
    const void* down_proj,
    uint32_t layer_id,
    DType weight_dtype,
    int group_size,
    void* cuda_stream
) {
    if (layer_id >= profile_.layers.size()) {
        LOG_ERROR("Layer %u out of range for sparsity profile", layer_id);
        return;
    }

    const auto& np = profile_.layers[layer_id];
    cudaStream_t stream = (cudaStream_t)cuda_stream;

    // If predictor is available, use dynamic prediction
    // Otherwise, use static hot neuron set from profile
    int num_active = np.hot_indices.size();
    int* d_active_indices = nullptr;

    // Upload hot indices to GPU
    cudaMalloc(&d_active_indices, num_active * sizeof(int));
    cudaMemcpyAsync(d_active_indices, np.hot_indices.data(),
                     num_active * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Zero scratch buffers (only active neurons will be written)
    cudaMemsetAsync(gate_out_, 0, scratch_size_, stream);
    cudaMemsetAsync(up_out_, 0, scratch_size_, stream);

    // Sparse gate and up projections
    // Only compute rows for active neurons
    namespace cuda = titan::cuda;

    // TODO: Call sparse_dequant_matvec_int4 or sparse_matvec based on weight_dtype
    // For now, this is the connection point — the CUDA kernels are implemented in sparse.cu

    // Record stats
    last_stats_.total_neurons = np.inter_dim;
    last_stats_.active_neurons = num_active;
    last_stats_.sparsity = np.sparsity();

    cudaFree(d_active_indices);
}

} // namespace titan
