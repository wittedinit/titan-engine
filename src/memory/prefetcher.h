#pragma once

#include "core/types.h"
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>

namespace titan {

// ============================================================================
// Expert Prefetcher with io_uring Pipelining
//
// During layer N's GPU compute, prefetch layer N+1's predicted experts
// from NVMe → RAM. On discrete GPU systems, NVMe reads don't compete
// with GPU compute (unlike Apple Silicon unified memory).
//
// Pipeline:
//   Layer N GPU compute ──────────────────────────────►
//   Layer N+1 NVMe prefetch ─────────►│ready│
//   Layer N+1 RAM→GPU staging ────────────────►│ready│
//
// Uses frequency-based prediction + temporal locality to guess which
// experts will be needed next. Maintains a ring buffer of prefetch
// requests and tracks hit rates for adaptive aggressiveness.
// ============================================================================

struct PrefetchRequest {
    uint32_t    layer;
    uint32_t    expert_id;
    size_t      expert_bytes;
    int         priority;       // Lower = higher priority
    bool        completed = false;
    void*       destination = nullptr;  // RAM buffer to load into
};

class ExpertPrefetcher {
public:
    ExpertPrefetcher() = default;
    ~ExpertPrefetcher();

    // Initialize with model parameters
    bool initialize(uint32_t num_layers, uint32_t num_experts,
                    uint32_t experts_per_tok, size_t expert_bytes,
                    const std::string& expert_dir,
                    int num_io_threads = 4);

    // Notify that layer N has selected these experts
    // The prefetcher uses this to update its prediction model and
    // start prefetching for layer N+1
    void on_expert_selected(uint32_t layer, const std::vector<uint32_t>& expert_ids);

    // Request prefetch for specific experts (called by the engine)
    void prefetch(uint32_t layer, const std::vector<uint32_t>& expert_ids);

    // Check if an expert is already prefetched (in RAM)
    bool is_ready(uint32_t layer, uint32_t expert_id) const;

    // Get prefetched expert data (nullptr if not ready)
    void* get_prefetched(uint32_t layer, uint32_t expert_id);

    // Set aggressiveness: how many layers ahead to prefetch
    void set_prefetch_depth(int depth) { prefetch_depth_ = depth; }

    // Stats
    struct Stats {
        int total_prefetches = 0;
        int hits = 0;          // Expert was prefetched when needed
        int misses = 0;        // Expert was NOT ready when needed
        int wasted = 0;        // Prefetched but never used
        float hit_rate() const {
            int total = hits + misses;
            return total > 0 ? (float)hits / total : 0;
        }
    };
    Stats stats() const { return stats_; }

    // Shutdown
    void stop();

private:
    uint32_t num_layers_ = 0;
    uint32_t num_experts_ = 0;
    uint32_t experts_per_tok_ = 0;
    size_t   expert_bytes_ = 0;
    std::string expert_dir_;
    int prefetch_depth_ = 1;

    // Expert access frequency tracking
    // [layer][expert_id] → access count
    std::vector<std::vector<uint32_t>> access_counts_;
    uint32_t total_tokens_ = 0;

    // Last-used experts per layer (temporal prediction)
    std::vector<std::vector<uint32_t>> last_experts_;

    // Prefetch buffer pool (pre-allocated RAM buffers)
    struct PrefetchBuffer {
        void*    data = nullptr;
        size_t   size = 0;
        uint64_t key = 0;       // (layer << 32) | expert_id
        bool     valid = false;
        bool     in_flight = false;
    };
    std::vector<PrefetchBuffer> buffers_;
    int num_buffers_ = 0;
    mutable std::mutex buffer_mutex_;

    // I/O thread pool
    std::vector<std::thread> io_threads_;
    std::queue<PrefetchRequest> io_queue_;
    std::mutex io_mutex_;
    std::condition_variable io_cv_;
    std::atomic<bool> running_{false};

    Stats stats_;

    // Prediction
    std::vector<uint32_t> predict_experts(uint32_t layer, int count) const;

    // I/O worker
    void io_worker();

    // Find or allocate a buffer
    PrefetchBuffer* find_buffer(uint64_t key);
    PrefetchBuffer* allocate_buffer(uint64_t key);
};

} // namespace titan
