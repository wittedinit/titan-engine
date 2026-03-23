#pragma once

#include "core/types.h"
#include "model/architecture.h"
#include "model/tokenizer.h"
#include "inference/kv_cache.h"

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <atomic>
#include <chrono>

namespace titan {

// ============================================================================
// Continuous Batching for Multi-User Serving
//
// Allows multiple concurrent requests to share GPU compute by:
// 1. Maintaining per-request KV caches
// 2. Batching multiple requests' tokens into a single forward pass
// 3. Dynamic scheduling: add/remove requests between iterations
// 4. Priority queue for fairness (round-robin or shortest-first)
//
// This is the vLLM/SGLang approach adapted for Titan's 3-tier memory.
// ============================================================================

enum class RequestState {
    QUEUED,         // Waiting to be scheduled
    PREFILLING,     // Processing prompt tokens
    DECODING,       // Generating tokens
    FINISHED,       // Done (EOS or max tokens)
    CANCELLED,      // Cancelled by user
};

struct InferenceRequest {
    int id;
    std::string prompt;
    SamplingParams sampling;
    RequestState state = RequestState::QUEUED;

    // Tokenized prompt
    std::vector<int> prompt_tokens;
    int prefill_pos = 0;        // How many prompt tokens processed

    // Generated tokens
    std::vector<int> output_tokens;
    int max_output_tokens = 2048;

    // KV cache slot
    int kv_slot = -1;           // Index into the batched KV cache

    // Timing
    std::chrono::steady_clock::time_point submit_time;
    std::chrono::steady_clock::time_point first_token_time;
    std::chrono::steady_clock::time_point finish_time;

    // Callback for streaming output
    using TokenCallback = std::function<void(int token_id, const std::string& text)>;
    using DoneCallback = std::function<void(const std::vector<int>& tokens)>;
    TokenCallback on_token;
    DoneCallback on_done;
};

struct BatchSchedulerConfig {
    int max_batch_size = 8;         // Max concurrent decoding requests
    int max_prefill_batch = 4;      // Max requests to prefill simultaneously
    int max_total_tokens = 32768;   // Max total tokens across all active requests
    int kv_cache_slots = 16;        // Number of independent KV cache slots
};

class BatchScheduler {
public:
    BatchScheduler() = default;
    ~BatchScheduler();

    bool initialize(ModelArchitecture* model,
                    const Tokenizer& tokenizer,
                    MemoryManager& memory,
                    const RuntimeConfig& runtime,
                    const BatchSchedulerConfig& config);

    // Submit a new request (thread-safe)
    int submit(InferenceRequest req);

    // Cancel a request (thread-safe)
    void cancel(int request_id);

    // Run the scheduler loop (blocking — call from dedicated thread)
    void run();

    // Stop the scheduler
    void stop();

    // Stats
    struct Stats {
        int active_requests = 0;
        int queued_requests = 0;
        int completed_requests = 0;
        int total_tokens_generated = 0;
        float avg_ttft_ms = 0;         // Time to first token
        float avg_tps = 0;             // Tokens per second per request
        float batch_utilization = 0;    // avg_batch_size / max_batch_size
    };
    Stats stats() const;

private:
    ModelArchitecture* model_ = nullptr;
    const Tokenizer* tokenizer_ = nullptr;
    MemoryManager* memory_ = nullptr;
    BatchSchedulerConfig config_;

    // Request queues
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<InferenceRequest> pending_;
    std::vector<InferenceRequest> active_;
    std::atomic<bool> running_{false};
    std::atomic<int> next_id_{0};

    // KV cache slots (one per active request)
    std::vector<std::unique_ptr<KVCache>> kv_slots_;
    std::vector<bool> slot_used_;

    // Batched computation buffers
    float* batch_hidden_ = nullptr;     // [max_batch, hidden_dim]
    float* batch_residual_ = nullptr;
    float* batch_logits_ = nullptr;     // [max_batch, vocab_size]

    // Scheduling
    int allocate_slot();
    void free_slot(int slot);
    void schedule_iteration();
    void prefill_requests(std::vector<InferenceRequest*>& reqs);
    void decode_step(std::vector<InferenceRequest*>& reqs);
};

} // namespace titan
