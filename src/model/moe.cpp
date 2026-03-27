#include "model/moe.h"
#include "model/loader.h"
#include "compute/dispatch.h"
#include "core/logging.h"
#include "core/config.h"

#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>

// All CUDA function declarations come from compute/dispatch.h (included above)

namespace titan {

MoEExecutor::~MoEExecutor() {
    free_buffers();

    // Free model weight allocations
    if (embedding_) { cudaFree(embedding_); embedding_ = nullptr; }
    // Only free lm_head_ if it's a separate allocation (not tied to embedding_)
    if (lm_head_ && lm_head_ != embedding_) { cudaFree(lm_head_); }
    lm_head_ = nullptr;
    if (final_norm_) { cudaFree(final_norm_); final_norm_ = nullptr; }

    for (auto& aw : attn_weights_) {
        if (aw.attn_norm) cudaFree(aw.attn_norm);
        if (aw.ffn_norm) cudaFree(aw.ffn_norm);
        if (aw.q_proj) cudaFree(aw.q_proj);
        if (aw.k_proj) cudaFree(aw.k_proj);
        if (aw.v_proj) cudaFree(aw.v_proj);
        if (aw.o_proj) cudaFree(aw.o_proj);
    }
    attn_weights_.clear();

    for (auto& ms : moe_state_) {
        if (ms.gate_weight) cudaFree(ms.gate_weight);
        if (ms.shared_gate_proj) cudaFree(ms.shared_gate_proj);
        if (ms.shared_up_proj) cudaFree(ms.shared_up_proj);
        if (ms.shared_down_proj) cudaFree(ms.shared_down_proj);
    }
    moe_state_.clear();

    cuda::destroy_cublas();
}

void MoEExecutor::allocate_buffers() {
    uint32_t qkv = config_.num_attn_heads * config_.head_dim;
    uint32_t kv = config_.num_kv_heads * config_.head_dim;
    uint32_t hd = config_.hidden_dim;
    uint32_t ne = config_.num_experts;
    uint32_t k = config_.experts_per_tok;
    uint32_t moe_inter = config_.moe_intermediate_dim > 0
                         ? config_.moe_intermediate_dim : config_.intermediate_dim;

    cudaMalloc(&q_buf_, qkv * sizeof(float));
    cudaMalloc(&k_buf_, kv * sizeof(float));
    cudaMalloc(&v_buf_, kv * sizeof(float));
    cudaMalloc(&attn_out_, qkv * sizeof(float));
    cudaMalloc(&norm_buf_, hd * sizeof(float));
    cudaMalloc(&gate_logits_, ne * sizeof(float));
    cudaMalloc(&routing_weights_, k * sizeof(float));
    cudaMalloc(&routing_indices_, k * sizeof(int));
    cudaMalloc(&expert_outputs_, k * hd * sizeof(float));
    cudaMalloc(&shared_out_, hd * sizeof(float));

    // Pre-allocate expert weight staging buffers
    // Each expert has 3 matrices: gate[moe_inter, hd] + up[moe_inter, hd] + down[hd, moe_inter]
    expert_weight_buf_size_ = 3 * (size_t)moe_inter * hd * sizeof(float);
    cudaMalloc(&expert_weight_buf_[0], expert_weight_buf_size_);
    cudaMalloc(&expert_weight_buf_[1], expert_weight_buf_size_);

    // Pre-allocate expert activation scratch
    cudaMalloc(&expert_gate_out_, moe_inter * sizeof(float));
    cudaMalloc(&expert_up_out_, moe_inter * sizeof(float));
    cudaMalloc(&shared_gate_out_, moe_inter * sizeof(float));
    cudaMalloc(&shared_up_out_, moe_inter * sizeof(float));

    LOG_INFO("MoE buffers pre-allocated: %.1f MB expert staging, %.1f MB scratch",
             expert_weight_buf_size_ * 2 / 1e6,
             (k * hd + ne + moe_inter * 4) * sizeof(float) / 1e6);
}

void MoEExecutor::free_buffers() {
    auto sf = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    sf(q_buf_); sf(k_buf_); sf(v_buf_); sf(attn_out_); sf(norm_buf_);
    sf(gate_logits_); sf(routing_weights_); sf(routing_indices_);
    sf(expert_outputs_); sf(shared_out_);
    sf(expert_weight_buf_[0]); sf(expert_weight_buf_[1]);
    sf(expert_gate_out_); sf(expert_up_out_);
    sf(shared_gate_out_); sf(shared_up_out_);
}

// ============================================================================
// Initialization
// ============================================================================

bool MoEExecutor::initialize(const std::string& model_path,
                              MemoryManager& memory,
                              const RuntimeConfig& runtime) {
    memory_ = &memory;
    runtime_ = runtime;

    config_ = load_model_config(model_path + "/config.json");
    if (config_.hidden_dim == 0) return false;
    if (config_.head_dim == 0 && config_.num_attn_heads > 0)
        config_.head_dim = config_.hidden_dim / config_.num_attn_heads;
    if (config_.moe_intermediate_dim == 0)
        config_.moe_intermediate_dim = config_.intermediate_dim;

    cuda::init_cublas();

    if (!kv_cache_.initialize(config_.num_layers, config_.num_kv_heads,
                               config_.head_dim, runtime.max_context_len))
        return false;

    ModelLoader loader;
    if (!loader.load(model_path)) return false;

    uint32_t hd = config_.hidden_dim;
    uint32_t qkv = config_.num_attn_heads * config_.head_dim;
    uint32_t kv = config_.num_kv_heads * config_.head_dim;
    uint32_t vocab = config_.vocab_size;
    uint32_t moe_inter = config_.moe_intermediate_dim;

    // Embedding — handle FP16/BF16 conversion to FP32
    cudaMalloc(&embedding_, vocab * hd * sizeof(float));
    cudaMemset(embedding_, 0, vocab * hd * sizeof(float));
    if (loader.has_tensor("model.embed_tokens.weight")) {
        auto emb_meta = loader.get_meta("model.embed_tokens.weight");
        bool emb_needs_convert = (emb_meta.dtype == DType::FP16 || emb_meta.dtype == DType::BF16);
        if (emb_needs_convert) {
            size_t num_elem = emb_meta.numel();
            size_t raw_bytes = emb_meta.byte_size();
            std::vector<uint16_t> fp16_data(num_elem);
            loader.read_tensor_cpu("model.embed_tokens.weight", fp16_data.data(), raw_bytes);
            std::vector<float> fp32_data(num_elem);
            for (size_t i = 0; i < num_elem; i++) {
                uint16_t h = fp16_data[i];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp_bits = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t f;
                if (exp_bits == 0) {
                    if (mant == 0) { f = sign << 31; }
                    else { exp_bits = 1; while (!(mant & 0x400)) { mant <<= 1; exp_bits--; } mant &= 0x3FF; f = (sign << 31) | ((exp_bits + 127 - 15) << 23) | (mant << 13); }
                } else if (exp_bits == 31) { f = (sign << 31) | 0x7F800000 | (mant << 13); }
                else { f = (sign << 31) | ((exp_bits + 127 - 15) << 23) | (mant << 13); }
                memcpy(&fp32_data[i], &f, 4);
            }
            cudaMemcpy(embedding_, fp32_data.data(), num_elem * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            loader.read_tensor_gpu("model.embed_tokens.weight", embedding_, emb_meta.byte_size());
        }
    }

    // Final norm
    cudaMalloc(&final_norm_, hd * sizeof(float));
    std::vector<float> ones(hd, 1.0f);
    cudaMemcpy(final_norm_, ones.data(), hd * sizeof(float), cudaMemcpyHostToDevice);
    if (loader.has_tensor("model.norm.weight")) {
        loader.read_tensor_gpu("model.norm.weight", final_norm_,
                                loader.get_meta("model.norm.weight").byte_size());
    }

    // LM head
    if (loader.has_tensor("lm_head.weight")) {
        cudaMalloc(&lm_head_, vocab * hd * sizeof(float));
        loader.read_tensor_gpu("lm_head.weight", lm_head_,
                                loader.get_meta("lm_head.weight").byte_size());
    } else {
        lm_head_ = embedding_;
    }

    // Per-layer
    attn_weights_.resize(config_.num_layers);
    moe_state_.resize(config_.num_layers);

    for (uint32_t l = 0; l < config_.num_layers; l++) {
        auto& aw = attn_weights_[l];
        auto& ms = moe_state_[l];
        std::string lp = "model.layers." + std::to_string(l);

        // Norms
        cudaMalloc(&aw.attn_norm, hd * sizeof(float));
        cudaMalloc(&aw.ffn_norm, hd * sizeof(float));
        cudaMemcpy(aw.attn_norm, ones.data(), hd * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(aw.ffn_norm, ones.data(), hd * sizeof(float), cudaMemcpyHostToDevice);

        auto try_load = [&](const std::string& name, void* dst, size_t fallback_size) {
            if (loader.has_tensor(name)) {
                loader.read_tensor_gpu(name, dst, loader.get_meta(name).byte_size());
            }
        };

        try_load(lp + ".input_layernorm.weight", aw.attn_norm, hd * sizeof(float));
        try_load(lp + ".post_attention_layernorm.weight", aw.ffn_norm, hd * sizeof(float));

        // Attention
        cudaMalloc(&aw.q_proj, qkv * hd * sizeof(float));
        cudaMalloc(&aw.k_proj, kv * hd * sizeof(float));
        cudaMalloc(&aw.v_proj, kv * hd * sizeof(float));
        cudaMalloc(&aw.o_proj, hd * qkv * sizeof(float));
        cudaMemset(aw.q_proj, 0, qkv * hd * sizeof(float));
        cudaMemset(aw.k_proj, 0, kv * hd * sizeof(float));
        cudaMemset(aw.v_proj, 0, kv * hd * sizeof(float));
        cudaMemset(aw.o_proj, 0, hd * qkv * sizeof(float));

        try_load(lp + ".self_attn.q_proj.weight", aw.q_proj, qkv * hd * sizeof(float));
        try_load(lp + ".self_attn.k_proj.weight", aw.k_proj, kv * hd * sizeof(float));
        try_load(lp + ".self_attn.v_proj.weight", aw.v_proj, kv * hd * sizeof(float));
        try_load(lp + ".self_attn.o_proj.weight", aw.o_proj, hd * qkv * sizeof(float));

        // Routing gate
        cudaMalloc(&ms.gate_weight, config_.num_experts * hd * sizeof(float));
        cudaMemset(ms.gate_weight, 0, config_.num_experts * hd * sizeof(float));
        try_load(lp + ".mlp.gate.weight", ms.gate_weight, config_.num_experts * hd * sizeof(float));

        // Shared expert
        if (config_.num_shared_experts > 0) {
            cudaMalloc(&ms.shared_gate_proj, moe_inter * hd * sizeof(float));
            cudaMalloc(&ms.shared_up_proj, moe_inter * hd * sizeof(float));
            cudaMalloc(&ms.shared_down_proj, hd * moe_inter * sizeof(float));
            cudaMemset(ms.shared_gate_proj, 0, moe_inter * hd * sizeof(float));
            cudaMemset(ms.shared_up_proj, 0, moe_inter * hd * sizeof(float));
            cudaMemset(ms.shared_down_proj, 0, hd * moe_inter * sizeof(float));
            try_load(lp + ".mlp.shared_expert.gate_proj.weight", ms.shared_gate_proj, 0);
            try_load(lp + ".mlp.shared_expert.up_proj.weight", ms.shared_up_proj, 0);
            try_load(lp + ".mlp.shared_expert.down_proj.weight", ms.shared_down_proj, 0);
        }

        if ((l + 1) % 10 == 0 || l == config_.num_layers - 1)
            LOG_INFO("Loaded layer %u/%u", l + 1, config_.num_layers);
    }

    expert_dir_ = model_path;
    expert_bytes_ = 3 * (size_t)moe_inter * hd * sizeof(float);

    allocate_buffers();

    LOG_INFO("MoE executor ready: %s (%uL, %u experts, K=%u, shared=%u)",
             config_.name.c_str(), config_.num_layers,
             config_.num_experts, config_.experts_per_tok,
             config_.num_shared_experts);
    return true;
}

void MoEExecutor::embed_token(int token_id, float* output, cudaStream_t stream) {
    if (!embedding_ || token_id < 0 || token_id >= (int)config_.vocab_size) return;
    cudaMemcpyAsync(output, embedding_ + (size_t)token_id * config_.hidden_dim,
                     config_.hidden_dim * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream ? stream : 0);
}

// ============================================================================
// Forward Pass — Pre-allocated buffers, no per-token malloc
// ============================================================================

void MoEExecutor::forward_layer(float* hidden, float* residual,
                                 uint32_t layer_id, int position,
                                 cudaStream_t cuda_stream) {
    if (layer_id >= config_.num_layers) return;

    cudaStream_t stream = cuda_stream;
    const auto& aw = attn_weights_[layer_id];
    const auto& ms = moe_state_[layer_id];
    uint32_t hd = config_.hidden_dim;
    uint32_t qkv = config_.num_attn_heads * config_.head_dim;
    uint32_t kv_dim = config_.num_kv_heads * config_.head_dim;
    uint32_t moe_inter = config_.moe_intermediate_dim;

    // ======= ATTENTION =======
    cuda::rmsnorm(norm_buf_, hidden, aw.attn_norm, hd, 1e-5f, stream);

    cuda::gemv_fp32((const float*)aw.q_proj, norm_buf_, q_buf_, qkv, hd, stream);
    cuda::gemv_fp32((const float*)aw.k_proj, norm_buf_, k_buf_, kv_dim, hd, stream);
    cuda::gemv_fp32((const float*)aw.v_proj, norm_buf_, v_buf_, kv_dim, hd, stream);

    cuda::apply_rope(q_buf_, k_buf_, config_.num_attn_heads, config_.num_kv_heads,
                     config_.head_dim, position, config_.rope_theta,
                     config_.rope_scaling, stream);

    kv_cache_.update(layer_id, position, k_buf_, v_buf_, stream);

    int seq_len = std::max(1, kv_cache_.seq_len());
    cuda::attention_decode(q_buf_, kv_cache_.k_cache(layer_id),
                           kv_cache_.v_cache(layer_id), attn_out_,
                           config_.num_attn_heads, config_.num_kv_heads,
                           config_.head_dim, seq_len, stream);

    // O proj + residual
    float* o_tmp = norm_buf_; // Reuse
    cuda::gemv_fp32((const float*)aw.o_proj, attn_out_, o_tmp, hd, qkv, stream);
    cuda::vector_add(residual, residual, o_tmp, hd, stream);

    // ======= MoE FFN =======
    cuda::rmsnorm(norm_buf_, residual, aw.ffn_norm, hd, 1e-5f, stream);

    // Routing
    cuda::moe_gate(norm_buf_, ms.gate_weight, gate_logits_, hd, config_.num_experts, stream);
    cuda::moe_topk(gate_logits_, routing_weights_, routing_indices_,
                   config_.num_experts, config_.experts_per_tok, stream);

    int k = config_.experts_per_tok;
    std::vector<int> h_indices(k);
    cudaMemcpy(h_indices.data(), routing_indices_, k * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemset(expert_outputs_, 0, k * hd * sizeof(float));

    // Execute experts using pre-allocated buffers (double-buffered)
    for (int e = 0; e < k; e++) {
        int expert_id = h_indices[e];
        float* expert_out = expert_outputs_ + e * hd;

        // Use alternating buffers for overlap potential
        float* weight_buf = expert_weight_buf_[e & 1];

        void* expert_data = memory_->get_expert(layer_id, expert_id, expert_bytes_);
        if (!expert_data) continue;

        // Copy expert weights from RAM → pre-allocated GPU buffer (single memcpy)
        cudaMemcpyAsync(weight_buf, expert_data, expert_bytes_,
                         cudaMemcpyHostToDevice, stream);

        // Parse weight layout within the buffer
        float* d_gate_w = weight_buf;
        float* d_up_w = weight_buf + (size_t)moe_inter * hd;
        float* d_down_w = d_up_w + (size_t)moe_inter * hd;

        // Expert forward: gate → up → SwiGLU → down
        cuda::gemv_fp32(d_gate_w, norm_buf_, expert_gate_out_, moe_inter, hd, stream);
        cuda::gemv_fp32(d_up_w, norm_buf_, expert_up_out_, moe_inter, hd, stream);
        cuda::swiglu(expert_gate_out_, expert_gate_out_, expert_up_out_, moe_inter, stream);
        cuda::gemv_fp32(d_down_w, expert_gate_out_, expert_out, hd, moe_inter, stream);
    }

    // Shared expert
    if (config_.num_shared_experts > 0 && ms.shared_gate_proj) {
        cuda::gemv_fp32((const float*)ms.shared_gate_proj, norm_buf_,
                        shared_gate_out_, moe_inter, hd, stream);
        cuda::gemv_fp32((const float*)ms.shared_up_proj, norm_buf_,
                        shared_up_out_, moe_inter, hd, stream);
        cuda::swiglu(shared_gate_out_, shared_gate_out_, shared_up_out_, moe_inter, stream);
        cuda::gemv_fp32((const float*)ms.shared_down_proj, shared_gate_out_,
                        shared_out_, hd, moe_inter, stream);
    }

    // Combine: weighted sum of experts + shared + residual → hidden
    // Use next layer's attn_norm, or final_norm_ for the last layer
    float* next_norm = (layer_id + 1 < config_.num_layers)
                       ? attn_weights_[layer_id + 1].attn_norm
                       : final_norm_;
    float shared_w = (config_.num_shared_experts > 0) ? 1.0f : 0.0f;
    cuda::fused_moe_combine_norm(
        hidden, residual, expert_outputs_, routing_weights_,
        (config_.num_shared_experts > 0) ? shared_out_ : nullptr, shared_w,
        next_norm, hd, k, 1e-5f, stream
    );
}

void MoEExecutor::compute_logits(const float* hidden, float* logits,
                                  cudaStream_t cuda_stream) {
    cudaStream_t stream = cuda_stream;
    cuda::rmsnorm(norm_buf_, hidden, final_norm_, config_.hidden_dim, 1e-5f, stream);
    cuda::gemv_fp32((const float*)lm_head_, norm_buf_, logits,
                    config_.vocab_size, config_.hidden_dim, stream);
}

size_t MoEExecutor::attention_weight_bytes(uint32_t) const {
    return (size_t)(config_.num_attn_heads + 2 * config_.num_kv_heads) *
           config_.head_dim * config_.hidden_dim * sizeof(float) +
           (size_t)config_.hidden_dim * config_.num_attn_heads * config_.head_dim * sizeof(float);
}

size_t MoEExecutor::ffn_weight_bytes(uint32_t) const {
    return (size_t)config_.num_experts * 3 * config_.hidden_dim *
           config_.moe_intermediate_dim * sizeof(float);
}

size_t MoEExecutor::expert_weight_bytes(uint32_t, uint32_t) const {
    return 3 * (size_t)config_.hidden_dim * config_.moe_intermediate_dim * sizeof(float);
}

size_t MoEExecutor::kv_cache_bytes_per_token(uint32_t) const {
    return 2 * (size_t)config_.num_kv_heads * config_.head_dim * sizeof(float);
}

void MoEExecutor::update_kv_cache(uint32_t layer_id, int position,
                                   const float* key, const float* value) {
    kv_cache_.update(layer_id, position, key, value);
}

} // namespace titan
