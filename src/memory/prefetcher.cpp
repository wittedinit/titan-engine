#include "memory/memory_manager.h"
#include "core/logging.h"

#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>

namespace titan {

// The Prefetcher is integrated into MemoryManager.
// This file provides the predictive expert prefetching logic.
//
// Strategy:
// 1. After routing for layer N completes, we know which experts are needed.
// 2. While GPU computes layer N's expert forward, we prefetch layer N+1's
//    predicted experts from NVMe→RAM.
// 3. Prediction uses a simple temporal heuristic: experts selected for layer N
//    are likely also selected for layer N+1 (~25% hit rate for same-layer,
//    higher for adjacent layers with similar routing patterns).
// 4. On discrete GPU systems, NVMe reads don't compete with GPU compute,
//    so prefetching is free during GPU execution.

// Expert access frequency tracker for smarter prefetching
struct ExpertFrequencyTracker {
    // Per-layer expert access counts
    std::vector<std::vector<uint32_t>> access_counts; // [layer][expert_id]
    uint32_t total_tokens = 0;

    void init(uint32_t num_layers, uint32_t num_experts) {
        access_counts.resize(num_layers);
        for (auto& layer : access_counts) {
            layer.resize(num_experts, 0);
        }
    }

    void record(uint32_t layer, uint32_t expert_id) {
        if (layer < access_counts.size() && expert_id < access_counts[layer].size()) {
            access_counts[layer][expert_id]++;
        }
        total_tokens++;
    }

    // Get top-N most frequently accessed experts for a layer
    std::vector<uint32_t> top_experts(uint32_t layer, uint32_t n) const {
        if (layer >= access_counts.size()) return {};

        const auto& counts = access_counts[layer];
        std::vector<std::pair<uint32_t, uint32_t>> sorted; // (count, expert_id)
        for (uint32_t i = 0; i < counts.size(); i++) {
            sorted.push_back({counts[i], i});
        }
        std::sort(sorted.rbegin(), sorted.rend());

        std::vector<uint32_t> result;
        for (uint32_t i = 0; i < std::min(n, (uint32_t)sorted.size()); i++) {
            result.push_back(sorted[i].second);
        }
        return result;
    }

    // Predict experts for next token based on frequency
    // Returns experts sorted by likelihood of selection
    std::vector<uint32_t> predict(uint32_t layer, uint32_t n,
                                   const std::vector<uint32_t>& current_experts) const {
        // Combine frequency-based prediction with temporal locality
        std::vector<uint32_t> candidates;

        // First: experts that were just used (temporal locality)
        for (auto e : current_experts) {
            candidates.push_back(e);
        }

        // Then: most frequent experts for this layer
        auto freq = top_experts(layer, n * 2);
        for (auto e : freq) {
            // Avoid duplicates
            bool already = false;
            for (auto c : candidates) {
                if (c == e) { already = true; break; }
            }
            if (!already) candidates.push_back(e);
            if (candidates.size() >= n * 2) break;
        }

        return candidates;
    }
};

// This is used by the inference engine to drive prefetching.
// The actual prefetch calls go through MemoryManager::prefetch_expert().

// In the future, this could be extended with:
// - ML-based routing prediction (train small MLP on hidden state → expert routing)
// - Cross-layer correlation (some experts co-activate across layers)
// - Adaptive prefetch depth (prefetch more aggressively when cache hit rate is low)

} // namespace titan
