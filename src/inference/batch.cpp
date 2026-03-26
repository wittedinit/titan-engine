#include "inference/batch.h"
#include "model/dense.h"
#include "compute/dispatch.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace titan {

BatchScheduler::~BatchScheduler() {
    stop();
    if (batch_hidden_) cudaFree(batch_hidden_);
    if (batch_residual_) cudaFree(batch_residual_);
    if (batch_logits_) cudaFree(batch_logits_);
    if (batch_sampled_tokens_) cudaFree(batch_sampled_tokens_);
}

bool BatchScheduler::initialize(ModelArchitecture* model,
                                 const Tokenizer& tokenizer,
                                 MemoryManager& memory,
                                 const RuntimeConfig& runtime,
                                 const BatchSchedulerConfig& config) {
    model_ = model;
    tokenizer_ = &tokenizer;
    memory_ = &memory;
    config_ = config;

    uint32_t hd = model->config().hidden_dim;
    uint32_t vocab = model->config().vocab_size;
    int bs = config.max_batch_size;

    cudaMalloc(&batch_hidden_, bs * hd * sizeof(float));
    cudaMalloc(&batch_residual_, bs * hd * sizeof(float));
    cudaMalloc(&batch_logits_, bs * vocab * sizeof(float));
    cudaMalloc(&batch_sampled_tokens_, bs * sizeof(int));

    // Initialize KV cache slots
    kv_slots_.resize(config.kv_cache_slots);
    slot_used_.resize(config.kv_cache_slots, false);

    for (int i = 0; i < config.kv_cache_slots; i++) {
        kv_slots_[i] = std::make_unique<KVCache>();
        kv_slots_[i]->initialize(
            model->config().num_layers,
            model->config().num_kv_heads,
            model->config().head_dim,
            runtime.max_context_len / config.kv_cache_slots // Share context budget
        );
    }

    LOG_INFO("Batch scheduler initialized: max_batch=%d, kv_slots=%d",
             bs, config.kv_cache_slots);
    return true;
}

int BatchScheduler::submit(InferenceRequest req) {
    req.id = next_id_++;
    req.submit_time = std::chrono::steady_clock::now();
    req.state = RequestState::QUEUED;

    // Tokenize prompt
    if (tokenizer_) {
        req.prompt_tokens = tokenizer_->encode(req.prompt);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.push(std::move(req));
    }
    cv_.notify_one();

    return req.id;
}

void BatchScheduler::cancel(int request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& req : active_) {
        if (req.id == request_id) {
            req.state = RequestState::CANCELLED;
            break;
        }
    }
}

int BatchScheduler::allocate_slot() {
    for (int i = 0; i < (int)slot_used_.size(); i++) {
        if (!slot_used_[i]) {
            slot_used_[i] = true;
            kv_slots_[i]->clear();
            return i;
        }
    }
    return -1; // No slots available
}

void BatchScheduler::free_slot(int slot) {
    if (slot >= 0 && slot < (int)slot_used_.size()) {
        slot_used_[slot] = false;
    }
}

void BatchScheduler::schedule_iteration() {
    // Move pending requests to active (up to max_batch_size)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pending_.empty() && (int)active_.size() < config_.max_batch_size) {
            auto req = std::move(pending_.front());
            pending_.pop();

            int slot = allocate_slot();
            if (slot < 0) {
                // No KV slots available — put back
                pending_.push(std::move(req));
                break;
            }

            req.kv_slot = slot;
            req.state = RequestState::PREFILLING;
            active_.push_back(std::move(req));
        }
    }

    // Separate active requests into prefilling and decoding
    std::vector<InferenceRequest*> prefill_batch;
    std::vector<InferenceRequest*> decode_batch;

    for (auto& req : active_) {
        if (req.state == RequestState::CANCELLED || req.state == RequestState::FINISHED)
            continue;
        if (req.state == RequestState::PREFILLING) {
            prefill_batch.push_back(&req);
        } else if (req.state == RequestState::DECODING) {
            decode_batch.push_back(&req);
        }
    }

    // Process prefill requests (one token at a time for simplicity)
    if (!prefill_batch.empty()) {
        prefill_requests(prefill_batch);
    }

    // Process decoding requests
    if (!decode_batch.empty()) {
        decode_step(decode_batch);
    }

    // Clean up finished/cancelled requests
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_.erase(
            std::remove_if(active_.begin(), active_.end(),
                [this](InferenceRequest& req) {
                    if (req.state == RequestState::FINISHED ||
                        req.state == RequestState::CANCELLED) {
                        free_slot(req.kv_slot);
                        if (req.on_done) {
                            req.on_done(req.output_tokens);
                        }
                        return true;
                    }
                    return false;
                }),
            active_.end()
        );
    }
}

void BatchScheduler::prefill_requests(std::vector<InferenceRequest*>& reqs) {
    uint32_t hd = model_->config().hidden_dim;

    for (auto* req : reqs) {
        if (req->prefill_pos >= (int)req->prompt_tokens.size()) {
            req->state = RequestState::DECODING;
            req->first_token_time = std::chrono::steady_clock::now();
            continue;
        }

        // Process one prompt token
        int token = req->prompt_tokens[req->prefill_pos];
        float* hidden = batch_hidden_; // Use first slot for serial prefill
        float* residual = batch_residual_;

        model_->embed_token(token, hidden);
        cudaMemcpy(residual, hidden, hd * sizeof(float), cudaMemcpyDeviceToDevice);

        for (uint32_t layer = 0; layer < model_->config().num_layers; layer++) {
            model_->forward_layer(hidden, residual, layer, req->prefill_pos, nullptr);
        }

        req->prefill_pos++;

        if (req->prefill_pos >= (int)req->prompt_tokens.size()) {
            req->state = RequestState::DECODING;
            req->first_token_time = std::chrono::steady_clock::now();
        }
    }
}

void BatchScheduler::decode_step(std::vector<InferenceRequest*>& reqs) {
    uint32_t hd = model_->config().hidden_dim;
    uint32_t vocab = model_->config().vocab_size;

    // For each decoding request, generate one token
    // (True batched decode would run all in a single forward pass)
    for (int req_idx = 0; req_idx < (int)reqs.size(); req_idx++) {
        auto* req = reqs[req_idx];

        // Bug #8 fix: Index into batched buffers using req_idx, not always offset 0
        float* hidden = batch_hidden_ + req_idx * hd;
        float* residual = batch_residual_ + req_idx * hd;
        float* logits = batch_logits_ + req_idx * vocab;

        int position = (int)req->prompt_tokens.size() + (int)req->output_tokens.size();

        // Get logits
        model_->compute_logits(hidden, logits, nullptr);

        // Bug #9 fix: Use pre-allocated token output slot instead of per-step cudaMalloc
        int* d_token = batch_sampled_tokens_ + req_idx;
        uint64_t rng = position * 6364136223846793005ULL + req->id;
        cuda::sample_token(logits, d_token, vocab,
                          req->sampling.temperature, req->sampling.top_p,
                          req->sampling.top_k, rng, nullptr);

        int token;
        cudaMemcpy(&token, d_token, sizeof(int), cudaMemcpyDeviceToHost);

        // Check EOS
        if (tokenizer_ && token == tokenizer_->eos_id()) {
            req->state = RequestState::FINISHED;
            req->finish_time = std::chrono::steady_clock::now();
            continue;
        }

        // Check max tokens
        if ((int)req->output_tokens.size() >= req->max_output_tokens) {
            req->state = RequestState::FINISHED;
            req->finish_time = std::chrono::steady_clock::now();
            continue;
        }

        req->output_tokens.push_back(token);

        // Callback
        if (req->on_token && tokenizer_) {
            std::string text = tokenizer_->decode(token);
            req->on_token(token, text);
        }

        // Embed for next step (uses virtual method)
        model_->embed_token(token, hidden);
        cudaMemcpy(residual, hidden, hd * sizeof(float), cudaMemcpyDeviceToDevice);

        for (uint32_t layer = 0; layer < model_->config().num_layers; layer++) {
            model_->forward_layer(hidden, residual, layer, position, nullptr);
        }
    }
}

void BatchScheduler::run() {
    running_ = true;
    LOG_INFO("Batch scheduler running (max_batch=%d)", config_.max_batch_size);

    while (running_) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (pending_.empty() && active_.empty()) {
                cv_.wait_for(lock, std::chrono::milliseconds(10));
                continue;
            }
        }

        schedule_iteration();
    }
}

void BatchScheduler::stop() {
    running_ = false;
    cv_.notify_all();
}

BatchScheduler::Stats BatchScheduler::stats() const {
    Stats s;
    s.active_requests = active_.size();
    // Simplified stats
    return s;
}

} // namespace titan
