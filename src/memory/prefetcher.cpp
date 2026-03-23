#include "memory/prefetcher.h"
#include "core/logging.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace titan {

ExpertPrefetcher::~ExpertPrefetcher() {
    stop();
    for (auto& buf : buffers_) {
        if (buf.data) free(buf.data);
    }
}

bool ExpertPrefetcher::initialize(uint32_t num_layers, uint32_t num_experts,
                                   uint32_t experts_per_tok, size_t expert_bytes,
                                   const std::string& expert_dir,
                                   int num_io_threads) {
    num_layers_ = num_layers;
    num_experts_ = num_experts;
    experts_per_tok_ = experts_per_tok;
    expert_bytes_ = expert_bytes;
    expert_dir_ = expert_dir;

    // Initialize access tracking
    access_counts_.resize(num_layers, std::vector<uint32_t>(num_experts, 0));
    last_experts_.resize(num_layers);

    // Pre-allocate prefetch buffers
    // Budget: enough for prefetch_depth layers × K experts × double-buffer
    num_buffers_ = prefetch_depth_ * experts_per_tok * 2 + 4;
    buffers_.resize(num_buffers_);
    for (int i = 0; i < num_buffers_; i++) {
        buffers_[i].data = aligned_alloc(4096, expert_bytes); // 4KB aligned for O_DIRECT
        buffers_[i].size = expert_bytes;
        if (!buffers_[i].data) {
            LOG_ERROR("Failed to allocate prefetch buffer %d (%zu bytes)", i, expert_bytes);
            return false;
        }
    }

    // Start I/O threads
    running_ = true;
    for (int i = 0; i < num_io_threads; i++) {
        io_threads_.emplace_back(&ExpertPrefetcher::io_worker, this);
    }

    LOG_INFO("Expert prefetcher: %d buffers (%.1f MB each), %d I/O threads, depth=%d",
             num_buffers_, expert_bytes / 1e6, num_io_threads, prefetch_depth_);
    return true;
}

void ExpertPrefetcher::stop() {
    running_ = false;
    io_cv_.notify_all();
    for (auto& t : io_threads_) {
        if (t.joinable()) t.join();
    }
    io_threads_.clear();
}

// ============================================================================
// Prediction: which experts will layer N+1 need?
// ============================================================================

std::vector<uint32_t> ExpertPrefetcher::predict_experts(uint32_t layer, int count) const {
    if (layer >= num_layers_) return {};

    std::vector<std::pair<float, uint32_t>> scored; // (score, expert_id)

    for (uint32_t e = 0; e < num_experts_; e++) {
        float score = 0;

        // Frequency-based score
        if (total_tokens_ > 0) {
            score += (float)access_counts_[layer][e] / total_tokens_;
        }

        // Temporal locality: boost experts used in previous layer
        if (layer > 0 && !last_experts_[layer - 1].empty()) {
            for (auto prev_e : last_experts_[layer - 1]) {
                if (prev_e == e) {
                    score += 0.5f; // Strong temporal boost
                    break;
                }
            }
        }

        // Same-layer temporal: boost experts used last time this layer ran
        if (!last_experts_[layer].empty()) {
            for (auto last_e : last_experts_[layer]) {
                if (last_e == e) {
                    score += 0.3f;
                    break;
                }
            }
        }

        scored.push_back({score, e});
    }

    // Sort by score descending
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<uint32_t> result;
    for (int i = 0; i < count && i < (int)scored.size(); i++) {
        result.push_back(scored[i].second);
    }
    return result;
}

// ============================================================================
// Expert selection notification
// ============================================================================

void ExpertPrefetcher::on_expert_selected(uint32_t layer,
                                           const std::vector<uint32_t>& expert_ids) {
    // Update tracking
    for (auto e : expert_ids) {
        if (e < num_experts_) {
            access_counts_[layer][e]++;
        }
    }
    last_experts_[layer] = expert_ids;
    total_tokens_++;

    // Predict and prefetch for next layers
    for (int d = 1; d <= prefetch_depth_; d++) {
        uint32_t next_layer = layer + d;
        if (next_layer >= num_layers_) break;

        // Predict top experts_per_tok * 2 (over-predict for safety)
        auto predicted = predict_experts(next_layer, experts_per_tok_ * 2);
        prefetch(next_layer, predicted);
    }
}

// ============================================================================
// Prefetch management
// ============================================================================

void ExpertPrefetcher::prefetch(uint32_t layer,
                                 const std::vector<uint32_t>& expert_ids) {
    for (auto e : expert_ids) {
        uint64_t key = ((uint64_t)layer << 32) | e;

        std::lock_guard<std::mutex> lock(buffer_mutex_);

        // Already prefetched?
        auto* buf = find_buffer(key);
        if (buf && (buf->valid || buf->in_flight)) continue;

        // Allocate buffer and queue I/O
        buf = allocate_buffer(key);
        if (!buf) continue;

        buf->in_flight = true;

        PrefetchRequest req;
        req.layer = layer;
        req.expert_id = e;
        req.expert_bytes = expert_bytes_;
        req.destination = buf->data;

        {
            std::lock_guard<std::mutex> io_lock(io_mutex_);
            io_queue_.push(req);
        }
        io_cv_.notify_one();
        stats_.total_prefetches++;
    }
}

bool ExpertPrefetcher::is_ready(uint32_t layer, uint32_t expert_id) const {
    uint64_t key = ((uint64_t)layer << 32) | expert_id;
    for (const auto& buf : buffers_) {
        if (buf.key == key && buf.valid) return true;
    }
    return false;
}

void* ExpertPrefetcher::get_prefetched(uint32_t layer, uint32_t expert_id) {
    uint64_t key = ((uint64_t)layer << 32) | expert_id;
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    for (auto& buf : buffers_) {
        if (buf.key == key && buf.valid) {
            stats_.hits++;
            return buf.data;
        }
    }
    stats_.misses++;
    return nullptr;
}

ExpertPrefetcher::PrefetchBuffer* ExpertPrefetcher::find_buffer(uint64_t key) {
    for (auto& buf : buffers_) {
        if (buf.key == key) return &buf;
    }
    return nullptr;
}

ExpertPrefetcher::PrefetchBuffer* ExpertPrefetcher::allocate_buffer(uint64_t key) {
    // Find a free (invalid, not in-flight) buffer
    for (auto& buf : buffers_) {
        if (!buf.valid && !buf.in_flight) {
            buf.key = key;
            buf.valid = false;
            return &buf;
        }
    }

    // Evict oldest valid buffer (LRU approximation)
    for (auto& buf : buffers_) {
        if (buf.valid && !buf.in_flight) {
            stats_.wasted++;
            buf.key = key;
            buf.valid = false;
            return &buf;
        }
    }

    return nullptr; // All buffers in flight
}

// ============================================================================
// I/O Worker Thread
// ============================================================================

void ExpertPrefetcher::io_worker() {
    while (running_) {
        PrefetchRequest req;
        {
            std::unique_lock<std::mutex> lock(io_mutex_);
            io_cv_.wait(lock, [this] { return !io_queue_.empty() || !running_; });
            if (!running_ && io_queue_.empty()) return;
            req = io_queue_.front();
            io_queue_.pop();
        }

        // Read expert data from NVMe
        char path[512];
        snprintf(path, sizeof(path), "%s/experts/layer_%02u.bin",
                 expert_dir_.c_str(), req.layer);

        int flags = O_RDONLY;
#ifdef O_DIRECT
        // O_DIRECT for bypassing page cache (predictable latency)
        // Requires 4KB-aligned buffer and offset
        if (((uintptr_t)req.destination & 4095) == 0) {
            flags |= O_DIRECT;
        }
#endif

        int fd = open(path, flags);
        if (fd < 0) {
            // Fallback without O_DIRECT
            fd = open(path, O_RDONLY);
        }

        if (fd >= 0) {
            off_t offset = (off_t)req.expert_id * req.expert_bytes;
            ssize_t total = 0;
            char* buf = (char*)req.destination;

            while ((size_t)total < req.expert_bytes) {
                ssize_t n = pread(fd, buf + total, req.expert_bytes - total,
                                  offset + total);
                if (n <= 0) {
                    if (n < 0 && errno == EINTR) continue;
                    break;
                }
                total += n;
            }
            close(fd);

            // Mark buffer as valid
            if ((size_t)total == req.expert_bytes) {
                uint64_t key = ((uint64_t)req.layer << 32) | req.expert_id;
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                for (auto& b : buffers_) {
                    if (b.key == key && b.in_flight) {
                        b.valid = true;
                        b.in_flight = false;
                        break;
                    }
                }
            }
        }
    }
}

} // namespace titan
