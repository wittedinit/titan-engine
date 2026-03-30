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

    if (memory_) {
        auto vf = [this](void* p) { if (p) memory_->vram_free(p); };

        // Free model weight allocations
        vf(embedding_); embedding_ = nullptr;
        if (lm_head_ && lm_head_ != embedding_) vf(lm_head_);
        lm_head_ = nullptr;
        vf(final_norm_); final_norm_ = nullptr;

        for (auto& aw : attn_weights_) {
            vf(aw.attn_norm);   vf(aw.ffn_norm);
            vf(aw.q_proj);      vf(aw.k_proj);    vf(aw.v_proj);    vf(aw.o_proj);
            vf(aw.q_a_proj);    vf(aw.q_b_proj);  vf(aw.q_a_norm);
            vf(aw.kv_a_proj);   vf(aw.kv_b_proj); vf(aw.kv_a_norm);
        }
        attn_weights_.clear();

        for (auto& ms : moe_state_) {
            vf(ms.gate_weight);
            vf(ms.shared_gate_proj);
            vf(ms.shared_up_proj);
            vf(ms.shared_down_proj);
            vf(ms.dense_ffn_buf);
        }
        moe_state_.clear();
    }

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

    auto va = [this](size_t sz) -> void* { return memory_->vram_alloc(sz); };
    q_buf_   = (float*)va(qkv * sizeof(float));
    k_buf_   = (float*)va(kv * sizeof(float));
    v_buf_   = (float*)va(kv * sizeof(float));
    attn_out_= (float*)va(qkv * sizeof(float));
    norm_buf_= (float*)va(hd * sizeof(float));
    // MLA scratch buffers (allocated even for non-MLA models — zero cost when unused)
    uint32_t q_lr  = config_.q_lora_rank  > 0 ? config_.q_lora_rank  : hd;
    uint32_t kv_lr = config_.kv_lora_rank > 0 ? config_.kv_lora_rank : hd;
    uint32_t rope_hd = config_.rope_head_dim > 0 ? config_.rope_head_dim : 64;
    uint32_t nope_hd = config_.nope_head_dim > 0 ? config_.nope_head_dim : (hd / config_.num_attn_heads);
    uint32_t v_hd    = config_.v_head_dim > 0 ? config_.v_head_dim : nope_hd;
    c_q_buf_    = (float*)va(q_lr  * sizeof(float));
    c_kv_buf_   = (float*)va((kv_lr + rope_hd) * sizeof(float));
    kv_expanded_= (float*)va(config_.num_kv_heads * (nope_hd + v_hd) * sizeof(float));
    k_nope_buf_ = (float*)va(config_.num_kv_heads * nope_hd * sizeof(float));
    v_mla_buf_  = (float*)va(config_.num_kv_heads * v_hd * sizeof(float));
    k_full_buf_ = (float*)va(config_.num_kv_heads * (nope_hd + rope_hd) * sizeof(float));
    q_nope_buf_ = (float*)va(config_.num_attn_heads * nope_hd * sizeof(float));
    gate_logits_     = (float*)va(ne * sizeof(float));
    routing_weights_ = (float*)va(k * sizeof(float));
    routing_indices_ = (int*)  va(k * sizeof(int));
    expert_outputs_  = (float*)va(k * hd * sizeof(float));
    shared_out_      = (float*)va(hd * sizeof(float));

    // Pre-allocate expert weight staging buffers.
    // Size depends on format: FP32 (3×moe_inter×hd×4) or NVFP4 (packed U8 + F8 scales).
    // NVFP4 sizes are set later in initialize() once has_nvfp4_ is known; fall back to FP32.
    if (has_nvfp4_ && nvfp4_gate_w_bytes_ > 0) {
        expert_weight_buf_size_ = nvfp4_gate_w_bytes_ + nvfp4_gate_s_bytes_ + sizeof(float)
                                + nvfp4_up_w_bytes_   + nvfp4_up_s_bytes_   + sizeof(float)
                                + nvfp4_down_w_bytes_ + nvfp4_down_s_bytes_ + sizeof(float);
    } else {
        expert_weight_buf_size_ = 3 * (size_t)moe_inter * hd * sizeof(float);
    }
    expert_weight_buf_[0] = (float*)va(expert_weight_buf_size_);
    expert_weight_buf_[1] = (float*)va(expert_weight_buf_size_);

    // Pre-allocate expert activation scratch — size to max of MoE inter and dense inter
    // so the same buffers can be reused for both MoE experts and dense FFN layers.
    uint32_t dense_inter = config_.intermediate_dim > 0 ? config_.intermediate_dim : moe_inter;
    uint32_t max_inter = std::max(moe_inter, dense_inter);
    expert_gate_out_ = (float*)va(max_inter * sizeof(float));
    expert_up_out_   = (float*)va(max_inter * sizeof(float));
    shared_gate_out_ = (float*)va(moe_inter * sizeof(float));
    shared_up_out_   = (float*)va(moe_inter * sizeof(float));

    LOG_INFO("MoE buffers pre-allocated: %.1f MB expert staging, %.1f MB scratch",
             expert_weight_buf_size_ * 2 / 1e6,
             (k * hd + ne + moe_inter * 4) * sizeof(float) / 1e6);
}

void MoEExecutor::free_buffers() {
    if (!memory_) return;
    auto sf = [this](auto*& p) { if (p) { memory_->vram_free((void*)p); p = nullptr; } };
    sf(q_buf_); sf(k_buf_); sf(v_buf_); sf(attn_out_); sf(norm_buf_);
    sf(gate_logits_); sf(routing_weights_); sf(routing_indices_);
    sf(expert_outputs_); sf(shared_out_);
    sf(expert_weight_buf_[0]); sf(expert_weight_buf_[1]);
    sf(expert_gate_out_); sf(expert_up_out_);
    sf(shared_gate_out_); sf(shared_up_out_);
    sf(c_q_buf_); sf(c_kv_buf_); sf(kv_expanded_);
    sf(k_nope_buf_); sf(v_mla_buf_); sf(k_full_buf_); sf(q_nope_buf_);
    if (nvfp4_load_buf_) { cudaFreeHost(nvfp4_load_buf_); nvfp4_load_buf_ = nullptr; }
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

    // For MLA (Multi-head Latent Attention), the KV cache stores K_nope and V
    // which both have dimension nope_head_dim (= v_head_dim for Kimi K2 models).
    // For standard GQA, use the full head_dim.
    uint32_t kv_cache_head_dim = config_.has_mla && config_.nope_head_dim > 0
                                  ? config_.nope_head_dim
                                  : config_.head_dim;
    // NOTE: KV cache is allocated AFTER model weights + scratch buffers so the
    // budget calculation reflects actual remaining VRAM.  (See alloc at end of init.)

    loader_ = std::make_unique<ModelLoader>();
    if (!loader_->load(model_path)) return false;
    ModelLoader& loader = *loader_;  // alias for the rest of init

    uint32_t hd = config_.hidden_dim;
    uint32_t qkv = config_.num_attn_heads * config_.head_dim;
    uint32_t kv = config_.num_kv_heads * config_.head_dim;
    uint32_t vocab = config_.vocab_size;
    uint32_t moe_inter = config_.moe_intermediate_dim;

    // Detect NVFP4 format: U8-packed experts mean shared expert can't use FP32 path
    bool has_nvfp4 = loader.has_tensor("model.layers.1.mlp.experts.0.gate_proj.weight") &&
                     loader.get_meta("model.layers.1.mlp.experts.0.gate_proj.weight").dtype == DType::INT8;
    has_nvfp4_ = has_nvfp4;

    if (has_nvfp4_) {
        // MoE expert layout: gate/up [moe_inter, hd/2] U8, scale [moe_inter, hd/16] F8
        //                    down    [hd, moe_inter/2] U8, scale [hd, moe_inter/16] F8
        nvfp4_gate_w_bytes_ = (size_t)moe_inter * (hd / 2);
        nvfp4_gate_s_bytes_ = (size_t)moe_inter * (hd / 16);
        nvfp4_up_w_bytes_   = nvfp4_gate_w_bytes_;
        nvfp4_up_s_bytes_   = nvfp4_gate_s_bytes_;
        nvfp4_down_w_bytes_ = (size_t)hd * (moe_inter / 2);
        nvfp4_down_s_bytes_ = (size_t)hd * (moe_inter / 16);
        expert_bytes_ = nvfp4_gate_w_bytes_ + nvfp4_gate_s_bytes_ + sizeof(float)
                      + nvfp4_up_w_bytes_   + nvfp4_up_s_bytes_   + sizeof(float)
                      + nvfp4_down_w_bytes_ + nvfp4_down_s_bytes_ + sizeof(float);

        // Dense FFN layout: gate/up [dense_inter, hd/2] U8, scale [dense_inter, hd/16] F8
        //                   down    [hd, dense_inter/2] U8, scale [hd, dense_inter/16] F8
        uint32_t dense_inter = config_.intermediate_dim > 0 ? config_.intermediate_dim : moe_inter;
        nvfp4_dense_gate_w_bytes_ = (size_t)dense_inter * (hd / 2);
        nvfp4_dense_gate_s_bytes_ = (size_t)dense_inter * (hd / 16);
        nvfp4_dense_up_w_bytes_   = nvfp4_dense_gate_w_bytes_;
        nvfp4_dense_up_s_bytes_   = nvfp4_dense_gate_s_bytes_;
        nvfp4_dense_down_w_bytes_ = (size_t)hd * (dense_inter / 2);
        nvfp4_dense_down_s_bytes_ = (size_t)hd * (dense_inter / 16);
        nvfp4_dense_ffn_bytes_    = nvfp4_dense_gate_w_bytes_ + nvfp4_dense_gate_s_bytes_ + sizeof(float)
                                  + nvfp4_dense_up_w_bytes_   + nvfp4_dense_up_s_bytes_   + sizeof(float)
                                  + nvfp4_dense_down_w_bytes_ + nvfp4_dense_down_s_bytes_ + sizeof(float);
        LOG_INFO("NVFP4 sizes: expert=%.1f MB, dense_ffn=%.1f MB",
                 expert_bytes_ / 1e6, nvfp4_dense_ffn_bytes_ / 1e6);

        // Pinned host buffer sized for whichever is larger (dense FFN or one expert)
        size_t staging_size = std::max(expert_bytes_, nvfp4_dense_ffn_bytes_);
        cudaMallocHost(&nvfp4_load_buf_, staging_size);
    } else {
        expert_bytes_ = 3 * (size_t)moe_inter * hd * sizeof(float);
    }

    // Embedding — store as BF16 if available (saves ~2.35 GB vs FP32 for vocab=163K, hidden=7168)
    if (loader.has_tensor("model.embed_tokens.weight")) {
        auto emb_meta = loader.get_meta("model.embed_tokens.weight");
        embedding_is_bf16_ = (emb_meta.dtype == DType::BF16 || emb_meta.dtype == DType::FP16);
        size_t emb_alloc = (size_t)vocab * hd * (embedding_is_bf16_ ? sizeof(uint16_t) : sizeof(float));
        embedding_ = memory_->vram_alloc(emb_alloc);
        cudaMemset(embedding_, 0, emb_alloc);
        loader.read_tensor_gpu("model.embed_tokens.weight", embedding_, emb_meta.byte_size());
    } else {
        embedding_is_bf16_ = false;
        embedding_ = memory_->vram_alloc((size_t)vocab * hd * sizeof(float));
        cudaMemset(embedding_, 0, (size_t)vocab * hd * sizeof(float));
    }

    // Final norm — always FP32 (tiny vector, used by rmsnorm kernel)
    final_norm_ = (float*)memory_->vram_alloc(hd * sizeof(float));
    std::vector<float> ones(hd, 1.0f);
    cudaMemcpy(final_norm_, ones.data(), hd * sizeof(float), cudaMemcpyHostToDevice);
    if (loader.has_tensor("model.norm.weight")) {
        // norm.weight is BF16 on disk; convert to FP32 for the rmsnorm kernel
        auto nm_meta = loader.get_meta("model.norm.weight");
        if (nm_meta.dtype == DType::BF16 || nm_meta.dtype == DType::FP16) {
            std::vector<uint16_t> raw(nm_meta.numel());
            loader.read_tensor_cpu("model.norm.weight", raw.data(), nm_meta.byte_size());
            std::vector<float> fp32(nm_meta.numel());
            for (size_t i = 0; i < (size_t)nm_meta.numel(); i++) {
                uint32_t v = (nm_meta.dtype == DType::BF16)
                             ? ((uint32_t)raw[i] << 16)
                             : [&]() -> uint32_t {
                                 uint16_t h=raw[i]; uint32_t s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF;
                                 if (!e) return (s<<31)|(m?(uint32_t)((127-15)<<23)|(m<<13):0u);
                                 if (e==31) return (s<<31)|0x7F800000|(m<<13);
                                 return (s<<31)|((e+127-15)<<23)|(m<<13);
                               }();
                memcpy(&fp32[i], &v, 4);
            }
            cudaMemcpy(final_norm_, fp32.data(), nm_meta.numel() * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            loader.read_tensor_gpu("model.norm.weight", final_norm_, nm_meta.byte_size());
        }
    }

    // LM head — same format as embedding (tied weights if no separate lm_head.weight)
    if (loader.has_tensor("lm_head.weight")) {
        auto lm_meta = loader.get_meta("lm_head.weight");
        size_t lm_alloc = (size_t)vocab * hd * (embedding_is_bf16_ ? sizeof(uint16_t) : sizeof(float));
        lm_head_ = memory_->vram_alloc(lm_alloc);
        loader.read_tensor_gpu("lm_head.weight", lm_head_, lm_meta.byte_size());
    } else {
        lm_head_ = embedding_; // Tied weights
    }

    // Per-layer
    attn_weights_.resize(config_.num_layers);
    moe_state_.resize(config_.num_layers);

    for (uint32_t l = 0; l < config_.num_layers; l++) {
        auto& aw = attn_weights_[l];
        auto& ms = moe_state_[l];
        std::string lp = "model.layers." + std::to_string(l);

        // Norms
        aw.attn_norm = (float*)memory_->vram_alloc(hd * sizeof(float));
        aw.ffn_norm  = (float*)memory_->vram_alloc(hd * sizeof(float));
        cudaMemcpy(aw.attn_norm, ones.data(), hd * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(aw.ffn_norm, ones.data(), hd * sizeof(float), cudaMemcpyHostToDevice);

        auto try_load = [&](const std::string& name, void* dst, size_t fallback_size) {
            if (loader.has_tensor(name)) {
                loader.read_tensor_gpu(name, dst, loader.get_meta(name).byte_size());
            }
        };

        try_load(lp + ".input_layernorm.weight", aw.attn_norm, hd * sizeof(float));
        try_load(lp + ".post_attention_layernorm.weight", aw.ffn_norm, hd * sizeof(float));

        // Attention — MLA models don't use standard q/k/v projections; skip those allocs
        // to avoid wasting ~400MB × 3 per layer (73 GB for a 61-layer model).
        if (config_.has_mla) {
            // o_proj for MLA: [hidden, n_heads * v_head_dim] = [7168, 8192] per layer
            // Stored in BF16 to save VRAM (117 MB → 58 MB per layer).
            uint32_t o_in_mla = config_.num_attn_heads * (config_.v_head_dim > 0
                                  ? config_.v_head_dim : (config_.nope_head_dim > 0
                                    ? config_.nope_head_dim : config_.head_dim));
            aw.o_proj = memory_->vram_alloc(hd * o_in_mla * sizeof(uint16_t));
            cudaMemset(aw.o_proj, 0, hd * o_in_mla * sizeof(uint16_t));
        } else {
            aw.q_proj = memory_->vram_alloc(qkv * hd * sizeof(float));
            aw.k_proj = memory_->vram_alloc(kv * hd * sizeof(float));
            aw.v_proj = memory_->vram_alloc(kv * hd * sizeof(float));
            aw.o_proj = memory_->vram_alloc(hd * qkv * sizeof(float));
            cudaMemset(aw.q_proj, 0, qkv * hd * sizeof(float));
            cudaMemset(aw.k_proj, 0, kv * hd * sizeof(float));
            cudaMemset(aw.v_proj, 0, kv * hd * sizeof(float));
            cudaMemset(aw.o_proj, 0, hd * qkv * sizeof(float));
        }

        if (config_.has_mla) {
            // MLA (Multi-head Latent Attention): load projection matrices in BF16 to save VRAM.
            // BF16 storage halves memory: 25 GB → 12.5 GB for 61-layer model.
            // Forward pass uses gemv_bf16_to_fp32 (cuBLAS GemmEx) for full-precision math.
            uint32_t q_lr  = config_.q_lora_rank  > 0 ? config_.q_lora_rank  : 1536;
            uint32_t kv_lr = config_.kv_lora_rank > 0 ? config_.kv_lora_rank :  512;
            uint32_t nope_hd = config_.nope_head_dim > 0 ? config_.nope_head_dim : 128;
            uint32_t v_hd  = config_.v_head_dim    > 0 ? config_.v_head_dim    : nope_hd;
            uint32_t rope_hd = config_.rope_head_dim > 0 ? config_.rope_head_dim : 64;
            uint32_t kv_a_rows = kv_lr + rope_hd;
            uint32_t kv_b_rows = config_.num_kv_heads * (nope_hd + v_hd);

            // Allocate BF16 projection matrices (2 bytes/elem instead of 4)
            aw.q_a_proj  = memory_->vram_alloc((size_t)q_lr * hd * sizeof(uint16_t));
            aw.q_b_proj  = memory_->vram_alloc((size_t)qkv  * q_lr * sizeof(uint16_t));
            aw.q_a_norm  = (float*)memory_->vram_alloc(q_lr * sizeof(float));
            aw.kv_a_proj = memory_->vram_alloc((size_t)kv_a_rows * hd * sizeof(uint16_t));
            aw.kv_b_proj = memory_->vram_alloc((size_t)kv_b_rows * kv_lr * sizeof(uint16_t));
            aw.kv_a_norm = (float*)memory_->vram_alloc(kv_lr * sizeof(float));
            cudaMemset(aw.q_a_proj,  0, (size_t)q_lr * hd * sizeof(uint16_t));
            cudaMemset(aw.q_b_proj,  0, (size_t)qkv  * q_lr * sizeof(uint16_t));
            // Norm weights stay FP32 (tiny vectors, used by rmsnorm kernel)
            std::fill(ones.begin(), ones.end(), 1.0f);
            if ((uint32_t)ones.size() >= q_lr)
                cudaMemcpy(aw.q_a_norm, ones.data(), q_lr * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(aw.kv_a_proj, 0, (size_t)kv_a_rows * hd * sizeof(uint16_t));
            cudaMemset(aw.kv_b_proj, 0, (size_t)kv_b_rows * kv_lr * sizeof(uint16_t));
            if ((uint32_t)ones.size() >= kv_lr)
                cudaMemcpy(aw.kv_a_norm, ones.data(), kv_lr * sizeof(float), cudaMemcpyHostToDevice);

            // Helper to load BF16/FP16 tensor from disk directly as BF16 into GPU buffer
            // (no FP32 expansion — kept BF16 to save VRAM)
            auto load_bf16_gpu = [&](const std::string& name, void* dst_bf16, size_t max_elems) {
                if (!loader.has_tensor(name)) return;
                auto meta = loader.get_meta(name);
                if ((size_t)meta.numel() > max_elems) return;
                if (meta.dtype != DType::BF16 && meta.dtype != DType::FP16) return;
                loader.read_tensor_gpu(name, dst_bf16, meta.byte_size());
            };

            // Helper to load norm weights as FP32 (BF16 on disk → FP32 on GPU)
            auto load_norm_fp32 = [&](const std::string& name, float* dst_fp32, size_t n_elems) {
                if (!loader.has_tensor(name)) return;
                auto meta = loader.get_meta(name);
                if ((size_t)meta.numel() > n_elems) return;
                std::vector<uint16_t> raw(meta.numel());
                loader.read_tensor_cpu(name, raw.data(), meta.byte_size());
                std::vector<float> fp32(meta.numel());
                bool is_bf16 = (meta.dtype == DType::BF16);
                for (size_t i = 0; i < (size_t)meta.numel(); i++) {
                    uint16_t h = raw[i];
                    uint32_t v = is_bf16 ? ((uint32_t)h << 16)
                                         : [&]() -> uint32_t {
                                             uint32_t s=(h>>15)&1, e=(h>>10)&0x1F, m=h&0x3FF;
                                             if (e==0) return (s<<31)|((m==0)?0u:(uint32_t)((127-15)<<23)|(m<<13));
                                             if (e==31) return (s<<31)|0x7F800000|(m<<13);
                                             return (s<<31)|((e+127-15)<<23)|(m<<13);
                                           }();
                    memcpy(&fp32[i], &v, 4);
                }
                cudaMemcpy(dst_fp32, fp32.data(), meta.numel() * sizeof(float), cudaMemcpyHostToDevice);
            };

            load_bf16_gpu(lp + ".self_attn.q_a_proj.weight",           aw.q_a_proj,  (size_t)q_lr * hd);
            load_bf16_gpu(lp + ".self_attn.q_b_proj.weight",           aw.q_b_proj,  (size_t)qkv * q_lr);
            load_norm_fp32(lp + ".self_attn.q_a_layernorm.weight",     aw.q_a_norm,  q_lr);
            load_bf16_gpu(lp + ".self_attn.kv_a_proj_with_mqa.weight", aw.kv_a_proj, (size_t)kv_a_rows * hd);
            load_bf16_gpu(lp + ".self_attn.kv_b_proj.weight",          aw.kv_b_proj, (size_t)kv_b_rows * kv_lr);
            load_norm_fp32(lp + ".self_attn.kv_a_layernorm.weight",    aw.kv_a_norm, kv_lr);
        } else {
            try_load(lp + ".self_attn.q_proj.weight", aw.q_proj, qkv * hd * sizeof(float));
            try_load(lp + ".self_attn.k_proj.weight", aw.k_proj, kv * hd * sizeof(float));
            try_load(lp + ".self_attn.v_proj.weight", aw.v_proj, kv * hd * sizeof(float));
        }
        // o_proj: [hidden, n_heads*v_hd] for MLA or [hidden, qkv] for standard
        size_t o_buf_bytes = config_.has_mla
            ? hd * (config_.num_attn_heads * (config_.v_head_dim > 0 ? config_.v_head_dim : config_.nope_head_dim)) * sizeof(float)
            : hd * qkv * sizeof(float);
        try_load(lp + ".self_attn.o_proj.weight", aw.o_proj, o_buf_bytes);

        // Dense layer detection: first_k_dense_replace layers use a standard FFN (no experts)
        bool is_dense_layer = (config_.first_k_dense_replace > 0 && l < config_.first_k_dense_replace);
        ms.is_dense = is_dense_layer;

        if (is_dense_layer) {
            // Dense FFN: load gate/up/down NVFP4 weights into GPU memory eagerly
            if (has_nvfp4 && nvfp4_dense_ffn_bytes_ > 0 && nvfp4_load_buf_) {
                uint32_t di = config_.intermediate_dim > 0 ? config_.intermediate_dim : moe_inter;
                std::string pfx = lp + ".mlp.";
                // Reuse load_nvfp4_expert logic inline with dense tensor names
                uint8_t* cpu = (uint8_t*)nvfp4_load_buf_;
                size_t off = 0;
                auto rdns = [&](const std::string& name, size_t sz) -> bool {
                    if (!loader.has_tensor(name)) { memset(cpu+off, 0, sz); return true; }
                    return loader.read_tensor_cpu(name, cpu+off, sz) == (ssize_t)sz;
                };
                auto rdg = [&](const std::string& name) -> bool {
                    if (!loader.has_tensor(name)) { float one=1.f; memcpy(cpu+off,&one,4); return true; }
                    auto meta = loader.get_meta(name);
                    if (meta.byte_size() == 2) {
                        uint16_t bf16=0; loader.read_tensor_cpu(name,&bf16,2);
                        uint32_t f32bits=(uint32_t)bf16<<16; memcpy(cpu+off,&f32bits,4); return true;
                    }
                    return loader.read_tensor_cpu(name,cpu+off,4)==4;
                };
                bool ok = rdns(pfx+"gate_proj.weight", nvfp4_dense_gate_w_bytes_); off+=nvfp4_dense_gate_w_bytes_;
                ok = ok && rdns(pfx+"gate_proj.weight_scale", nvfp4_dense_gate_s_bytes_); off+=nvfp4_dense_gate_s_bytes_;
                ok = ok && rdg(pfx+"gate_proj.weight_scale_2"); off+=sizeof(float);
                ok = ok && rdns(pfx+"up_proj.weight", nvfp4_dense_up_w_bytes_); off+=nvfp4_dense_up_w_bytes_;
                ok = ok && rdns(pfx+"up_proj.weight_scale", nvfp4_dense_up_s_bytes_); off+=nvfp4_dense_up_s_bytes_;
                ok = ok && rdg(pfx+"up_proj.weight_scale_2"); off+=sizeof(float);
                ok = ok && rdns(pfx+"down_proj.weight", nvfp4_dense_down_w_bytes_); off+=nvfp4_dense_down_w_bytes_;
                ok = ok && rdns(pfx+"down_proj.weight_scale", nvfp4_dense_down_s_bytes_); off+=nvfp4_dense_down_s_bytes_;
                ok = ok && rdg(pfx+"down_proj.weight_scale_2"); off+=sizeof(float);
                if (ok) {
                    // Read global scales from CPU staging before GPU transfer
                    size_t gate_g_off = nvfp4_dense_gate_w_bytes_ + nvfp4_dense_gate_s_bytes_;
                    size_t up_g_off   = gate_g_off + 4 + nvfp4_dense_up_w_bytes_ + nvfp4_dense_up_s_bytes_;
                    size_t down_g_off = up_g_off   + 4 + nvfp4_dense_down_w_bytes_ + nvfp4_dense_down_s_bytes_;
                    memcpy(&ms.dense_gate_g, cpu + gate_g_off, 4);
                    memcpy(&ms.dense_up_g,   cpu + up_g_off,   4);
                    memcpy(&ms.dense_down_g, cpu + down_g_off, 4);
                    ms.dense_ffn_buf = memory_->vram_alloc(nvfp4_dense_ffn_bytes_);
                    if (ms.dense_ffn_buf) {
                        cudaMemcpy(ms.dense_ffn_buf, nvfp4_load_buf_, nvfp4_dense_ffn_bytes_, cudaMemcpyHostToDevice);
                        LOG_INFO("Dense FFN layer %u: loaded %.1f MB (g=%.5f, u=%.5f, d=%.5f)",
                                 l, nvfp4_dense_ffn_bytes_/1e6,
                                 ms.dense_gate_g, ms.dense_up_g, ms.dense_down_g);
                    }
                }
            }
            // Dense layers have no routing gate — skip gate_weight alloc
            ms.gate_weight = nullptr;
        } else {
            // Routing gate
            ms.gate_weight = (float*)memory_->vram_alloc(config_.num_experts * hd * sizeof(float));
            cudaMemset(ms.gate_weight, 0, config_.num_experts * hd * sizeof(float));
            try_load(lp + ".mlp.gate.weight", ms.gate_weight, config_.num_experts * hd * sizeof(float));
        }

        // Shared expert — only allocate FP32 buffers for non-NVFP4 models.
        // NVFP4 shared experts (mlp.shared_experts.*) are U8 format and can't
        // be directly used with the FP32 gemv path. Leave nullptr to skip the
        // shared expert in forward_layer() (fused_moe_combine_norm handles nullptr).
        if (config_.num_shared_experts > 0 && !has_nvfp4) {
            ms.shared_gate_proj = memory_->vram_alloc(moe_inter * hd * sizeof(float));
            ms.shared_up_proj   = memory_->vram_alloc(moe_inter * hd * sizeof(float));
            ms.shared_down_proj = memory_->vram_alloc(hd * moe_inter * sizeof(float));
            cudaMemset(ms.shared_gate_proj, 0, moe_inter * hd * sizeof(float));
            cudaMemset(ms.shared_up_proj, 0, moe_inter * hd * sizeof(float));
            cudaMemset(ms.shared_down_proj, 0, hd * moe_inter * sizeof(float));
            // Try both singular and plural naming conventions
            if (!loader.has_tensor(lp + ".mlp.shared_expert.gate_proj.weight"))
                try_load(lp + ".mlp.shared_experts.gate_proj.weight", ms.shared_gate_proj, 0);
            else
                try_load(lp + ".mlp.shared_expert.gate_proj.weight", ms.shared_gate_proj, 0);
            if (!loader.has_tensor(lp + ".mlp.shared_expert.up_proj.weight"))
                try_load(lp + ".mlp.shared_experts.up_proj.weight", ms.shared_up_proj, 0);
            else
                try_load(lp + ".mlp.shared_expert.up_proj.weight", ms.shared_up_proj, 0);
            if (!loader.has_tensor(lp + ".mlp.shared_expert.down_proj.weight"))
                try_load(lp + ".mlp.shared_experts.down_proj.weight", ms.shared_down_proj, 0);
            else
                try_load(lp + ".mlp.shared_expert.down_proj.weight", ms.shared_down_proj, 0);
        }

        if ((l + 1) % 10 == 0 || l == config_.num_layers - 1)
            LOG_INFO("Loaded layer %u/%u", l + 1, config_.num_layers);
    }

    expert_dir_ = model_path;
    // expert_bytes_ already set above (NVFP4 or FP32 layout)

    allocate_buffers();

    // Allocate KV cache with whatever VRAM remains after model weights + scratch buffers.
    // Doing this last ensures the budget is accurate — on large models like Kimi K2.5,
    // fixed attention weights consume ~28 GB, leaving only ~1-2 GB for the KV cache.
    // With MLA's expanded KV representation (num_kv_heads × nope_head_dim per position),
    // each context position costs 2 × num_layers × num_kv_heads × nope_head_dim × 4 bytes.
    {
        size_t pool_free = memory_ ? memory_->vram_free_bytes() : 0;
        LOG_INFO("VRAM pool free for KV cache (after weights): %.1f MB", pool_free / 1e6);

        size_t kv_per_ctx = 2ULL * config_.num_layers * config_.num_kv_heads
                            * kv_cache_head_dim * sizeof(float);
        // Leave 256 MB headroom for inference scratch buffers in engine.cpp
        size_t kv_budget = (pool_free > 256ULL << 20) ? pool_free - (256ULL << 20) : 0;
        uint32_t max_ctx_by_pool = (kv_per_ctx > 0 && kv_budget > 0)
                                   ? (uint32_t)(kv_budget / kv_per_ctx)
                                   : 0u;
        uint32_t capped_ctx = std::min(runtime.max_context_len,
                                        std::max(128u, max_ctx_by_pool));
        if (capped_ctx < runtime.max_context_len) {
            LOG_WARN("KV cache: capping context from %u to %u (%.1f MB free after weights)",
                     runtime.max_context_len, capped_ctx, pool_free / 1e6);
        }

        size_t kv_bytes = (size_t)config_.num_layers * (size_t)capped_ctx
                          * config_.num_kv_heads * kv_cache_head_dim * sizeof(float);
        float* k_buf = (float*)memory_->vram_alloc(kv_bytes);
        float* v_buf = (float*)memory_->vram_alloc(kv_bytes);

        if (!k_buf || !v_buf) {
            LOG_ERROR("KV cache: failed to allocate %.1f MB from VRAM pool (pool_free=%.1f MB)",
                      kv_bytes * 2 / 1e6, pool_free / 1e6);
            return false;
        }

        LOG_INFO("KV cache: %u positions × %.1f MB/pos = %.1f MB total",
                 capped_ctx, kv_per_ctx / 1e6, kv_bytes * 2 / 1e6);

        if (!kv_cache_.initialize_external(config_.num_layers, config_.num_kv_heads,
                                            kv_cache_head_dim, capped_ctx, k_buf, v_buf))
            return false;
    }

    LOG_INFO("MoE executor ready: %s (%uL, %u experts, K=%u, shared=%u)",
             config_.name.c_str(), config_.num_layers,
             config_.num_experts, config_.experts_per_tok,
             config_.num_shared_experts);
    return true;
}

// ============================================================================
// NVFP4 On-Demand Expert Loading
//
// Reads gate/up/down weight + F8 scale + F32 global scale from the ModelLoader
// (safetensors on disk) directly into a pre-allocated GPU staging buffer.
// Layout in gpu_buf: [gate_w | gate_s | gate_g | up_w | up_s | up_g | down_w | down_s | down_g]
// where _w = U8 weights, _s = F8 scales, _g = F32 global (weight_scale_2)
// ============================================================================

bool MoEExecutor::load_nvfp4_expert(uint32_t layer_id, uint32_t expert_id,
                                     void* gpu_buf, cudaStream_t stream) {
    if (!loader_ || !nvfp4_load_buf_) return false;

    std::string base = "model.layers." + std::to_string(layer_id)
                     + ".mlp.experts." + std::to_string(expert_id);

    // First call: log tensor metadata for diagnostics
    static bool logged_shapes = false;
    if (!logged_shapes && layer_id == 0 && expert_id == 0) {
        logged_shapes = true;
        auto log_tensor = [&](const std::string& suffix) {
            std::string name = base + suffix;
            if (loader_->has_tensor(name)) {
                auto m = loader_->get_meta(name);
                std::string shape_str;
                for (size_t di = 0; di < m.shape.size(); di++) {
                    if (di) shape_str += "x";
                    shape_str += std::to_string(m.shape[di]);
                }
                LOG_DEBUG("  %s: dtype=%s shape=[%s] bytes=%zu",
                          suffix.c_str(), m.dtype_str.c_str(), shape_str.c_str(), m.byte_size());
            } else {
                LOG_DEBUG("  %s: NOT FOUND", suffix.c_str());
            }
        };
        LOG_DEBUG("NVFP4 expert tensor shapes for %s:", base.c_str());
        log_tensor(".gate_proj.weight");
        log_tensor(".gate_proj.weight_scale");
        log_tensor(".gate_proj.weight_scale_2");
        log_tensor(".gate_proj.input_scale");
        log_tensor(".up_proj.weight");
        log_tensor(".down_proj.weight");
    }

    uint8_t* cpu = (uint8_t*)nvfp4_load_buf_;
    size_t off = 0;

    // Helper: read a tensor into the pinned CPU buffer
    auto read = [&](const std::string& name, size_t expected) -> bool {
        if (!loader_->has_tensor(name)) {
            LOG_ERROR("NVFP4 expert tensor not found: %s", name.c_str());
            return false;
        }
        ssize_t got = loader_->read_tensor_cpu(name, cpu + off, expected);
        if (got != (ssize_t)expected) {
            auto meta = loader_->get_meta(name);
            LOG_ERROR("NVFP4 tensor size mismatch: %s — expected %zu bytes, tensor has %zu bytes (dtype=%s)",
                      name.c_str(), expected, meta.byte_size(), meta.dtype_str.c_str());
            return false;
        }
        return true;
    };

    auto read_scalar = [&](const std::string& name) -> bool {
        if (!loader_->has_tensor(name)) {
            // Default: 1.0
            float one = 1.0f;
            memcpy(cpu + off, &one, sizeof(float));
            return true;
        }
        auto meta = loader_->get_meta(name);
        size_t tensor_bytes = meta.byte_size();
        if (tensor_bytes == sizeof(float)) {
            ssize_t got = loader_->read_tensor_cpu(name, cpu + off, sizeof(float));
            return got == (ssize_t)sizeof(float);
        } else if (tensor_bytes == sizeof(uint16_t)) {
            // BF16 scalar — convert to F32
            uint16_t bf16 = 0;
            ssize_t got = loader_->read_tensor_cpu(name, &bf16, sizeof(uint16_t));
            if (got != (ssize_t)sizeof(uint16_t)) return false;
            uint32_t f32bits = (uint32_t)bf16 << 16;
            memcpy(cpu + off, &f32bits, sizeof(float));
            return true;
        } else {
            LOG_ERROR("NVFP4 scalar %s has unexpected size %zu bytes (dtype=%s) — using 1.0",
                      name.c_str(), tensor_bytes, meta.dtype_str.c_str());
            float one = 1.0f;
            memcpy(cpu + off, &one, sizeof(float));
            return true;
        }
    };

    // gate
    if (!read(base + ".gate_proj.weight", nvfp4_gate_w_bytes_))     return false;
    off += nvfp4_gate_w_bytes_;
    if (!read(base + ".gate_proj.weight_scale", nvfp4_gate_s_bytes_)) return false;
    off += nvfp4_gate_s_bytes_;
    if (!read_scalar(base + ".gate_proj.weight_scale_2")) return false;
    off += sizeof(float);

    // up
    if (!read(base + ".up_proj.weight", nvfp4_up_w_bytes_))         return false;
    off += nvfp4_up_w_bytes_;
    if (!read(base + ".up_proj.weight_scale", nvfp4_up_s_bytes_))   return false;
    off += nvfp4_up_s_bytes_;
    if (!read_scalar(base + ".up_proj.weight_scale_2")) return false;
    off += sizeof(float);

    // down
    if (!read(base + ".down_proj.weight", nvfp4_down_w_bytes_))     return false;
    off += nvfp4_down_w_bytes_;
    if (!read(base + ".down_proj.weight_scale", nvfp4_down_s_bytes_)) return false;
    off += nvfp4_down_s_bytes_;
    if (!read_scalar(base + ".down_proj.weight_scale_2")) return false;
    off += sizeof(float);

    // Transfer to GPU
    cudaMemcpyAsync(gpu_buf, nvfp4_load_buf_, expert_bytes_,
                     cudaMemcpyHostToDevice, stream);
    return true;
}

void MoEExecutor::embed_token(int token_id, float* output, cudaStream_t stream) {
    if (!embedding_ || token_id < 0 || token_id >= (int)config_.vocab_size) return;
    if (embedding_is_bf16_) {
        cuda::embed_token_bf16(output, embedding_, token_id, config_.hidden_dim, stream);
    } else {
        cudaMemcpyAsync(output, (const float*)embedding_ + (size_t)token_id * config_.hidden_dim,
                         config_.hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream ? stream : 0);
    }
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

    uint32_t nope_hd  = config_.nope_head_dim  > 0 ? config_.nope_head_dim  : config_.head_dim;
    uint32_t rope_hd  = config_.rope_head_dim  > 0 ? config_.rope_head_dim  : 0;
    uint32_t v_hd     = config_.v_head_dim     > 0 ? config_.v_head_dim     : nope_hd;
    uint32_t q_lr     = config_.q_lora_rank    > 0 ? config_.q_lora_rank    : hd;
    uint32_t kv_lr    = config_.kv_lora_rank   > 0 ? config_.kv_lora_rank   : hd;

    if (config_.has_mla && aw.q_a_proj) {
        // MLA two-hop Q: hidden → q_a (BF16) → RMSNorm → q_b (BF16) → Q
        cuda::gemv_bf16_to_fp32(aw.q_a_proj, norm_buf_, c_q_buf_, q_lr, hd, stream);
        cuda::rmsnorm(c_q_buf_, c_q_buf_, aw.q_a_norm, q_lr, 1e-5f, stream);
        cuda::gemv_bf16_to_fp32(aw.q_b_proj, c_q_buf_, q_buf_, qkv, q_lr, stream);

        // MLA two-hop KV: hidden → kv_a (BF16) → RMSNorm → kv_b (BF16) → K_nope+V
        uint32_t kv_a_rows = kv_lr + rope_hd;
        cuda::gemv_bf16_to_fp32(aw.kv_a_proj, norm_buf_, c_kv_buf_, kv_a_rows, hd, stream);
        // c_kv_buf_ layout = [kv_lr elems | rope_hd elems]
        // Apply RMSNorm only to the compressed KV portion (first kv_lr elements)
        cuda::rmsnorm(c_kv_buf_, c_kv_buf_, aw.kv_a_norm, kv_lr, 1e-5f, stream);
        // Expand: kv_b (BF16) [n_kv_heads*(nope_hd+v_hd)] from [kv_lr]
        uint32_t kv_b_rows = config_.num_kv_heads * (nope_hd + v_hd);
        cuda::gemv_bf16_to_fp32(aw.kv_b_proj, c_kv_buf_, kv_expanded_, kv_b_rows, kv_lr, stream);

        // Deinterleave kv_expanded → k_nope_buf_ and v_mla_buf_
        cuda::mla_deinterleave_kv(kv_expanded_, k_nope_buf_, v_mla_buf_,
                                   config_.num_kv_heads, nope_hd, v_hd, stream);

        // Assemble full K with rope component: k_nope || k_rope (broadcast)
        // c_kv_buf_[kv_lr:kv_lr+rope_hd] contains the shared rope key
        if (rope_hd > 0) {
            cuda::mla_assemble_k(k_nope_buf_, c_kv_buf_ + kv_lr, k_full_buf_,
                                  config_.num_kv_heads, nope_hd, rope_hd, stream);
        } else {
            // No rope component — k_full = k_nope
            cudaMemcpyAsync(k_full_buf_, k_nope_buf_,
                            config_.num_kv_heads * nope_hd * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // Apply RoPE to rope portions of Q and K
        // Q: rope portion starts at nope_hd per head; K: assembled k_full has rope at end
        // The standard apply_rope expects Q[n_heads*head_dim] and K[n_kv_heads*head_dim]
        // For MLA: head_dim = nope_hd + rope_hd, rope is applied to last rope_hd dims
        uint32_t eff_head_dim = nope_hd + rope_hd;
        cuda::apply_rope(q_buf_, k_full_buf_,
                         config_.num_attn_heads, config_.num_kv_heads,
                         eff_head_dim, position, config_.rope_theta,
                         config_.rope_scaling, stream);

        // Extract Q nope for attention: use Q_nope × K_nope (simplified — drops RoPE)
        cuda::mla_extract_q_nope(q_buf_, q_nope_buf_,
                                  config_.num_attn_heads, nope_hd, rope_hd, stream);

        // Store K_nope and V in KV cache (head_dim = nope_hd for KV cache in MLA)
        kv_cache_.update(layer_id, position, k_nope_buf_, v_mla_buf_, stream);

        // Attention decode: use Q_nope [n_heads*nope_hd] and K_nope / V [n_kv*nope_hd].
        // This simplification drops RoPE from the computation; positional encoding is
        // approximate until full MLA-RoPE attention is implemented.
        int seq_len = std::max(1, kv_cache_.seq_len());
        cuda::attention_decode(q_nope_buf_, kv_cache_.k_cache(layer_id),
                               kv_cache_.v_cache(layer_id), attn_out_,
                               config_.num_attn_heads, config_.num_kv_heads,
                               nope_hd, seq_len, stream);

        // O proj (BF16): shape [hidden, n_heads*v_hd] = [7168, 8192]
        uint32_t o_in_dim = config_.num_attn_heads * v_hd;
        float* o_tmp = norm_buf_;
        cuda::gemv_bf16_to_fp32(aw.o_proj, attn_out_, o_tmp, hd, o_in_dim, stream);
        cuda::vector_add(residual, residual, o_tmp, hd, stream);
    } else {
        // Standard GQA attention
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

        float* o_tmp = norm_buf_;
        cuda::gemv_fp32((const float*)aw.o_proj, attn_out_, o_tmp, hd, qkv, stream);
        cuda::vector_add(residual, residual, o_tmp, hd, stream);
    }

    // ======= FFN (MoE or Dense) =======
    cuda::rmsnorm(norm_buf_, residual, aw.ffn_norm, hd, 1e-5f, stream);

    // Dense layer path (first_k_dense_replace layers use a standard FFN, not MoE)
    if (ms.is_dense) {
        if (ms.dense_ffn_buf && has_nvfp4_) {
            uint32_t di = config_.intermediate_dim > 0 ? config_.intermediate_dim : moe_inter;
            const uint8_t* d_gate_w = (const uint8_t*)ms.dense_ffn_buf;
            const uint8_t* d_gate_s = d_gate_w + nvfp4_dense_gate_w_bytes_;
            const uint8_t* d_up_w   = d_gate_s + nvfp4_dense_gate_s_bytes_ + sizeof(float);
            const uint8_t* d_up_s   = d_up_w   + nvfp4_dense_up_w_bytes_;
            const uint8_t* d_down_w = d_up_s   + nvfp4_dense_up_s_bytes_   + sizeof(float);
            const uint8_t* d_down_s = d_down_w + nvfp4_dense_down_w_bytes_;

            cuda::dequant_matvec_nvfp4(d_gate_w, d_gate_s, ms.dense_gate_g,
                                        norm_buf_, expert_gate_out_, di, hd, stream);
            cuda::dequant_matvec_nvfp4(d_up_w,   d_up_s,   ms.dense_up_g,
                                        norm_buf_, expert_up_out_,   di, hd, stream);
            cuda::swiglu(expert_gate_out_, expert_gate_out_, expert_up_out_, di, stream);
            cuda::dequant_matvec_nvfp4(d_down_w, d_down_s, ms.dense_down_g,
                                        expert_gate_out_, hidden, hd, di, stream);
            // Add to residual
            cuda::vector_add(residual, residual, hidden, hd, stream);
            cuda::vector_copy(hidden, residual, hd, stream);
        }
        // else: no dense FFN weights loaded → skip (residual passes unchanged)
        return;
    }

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
        void* weight_buf = expert_weight_buf_[e & 1];

        if (has_nvfp4_) {
            // --- NVFP4 path ---
            // CPU staging buffer layout (offsets in bytes):
            //   gate_w [gate_w_sz] | gate_s [gate_s_sz] | gate_g [4]
            //   up_w   [up_w_sz]   | up_s   [up_s_sz]   | up_g   [4]
            //   down_w [down_w_sz] | down_s [down_s_sz] | down_g [4]
            const uint8_t* cpu_buf = (const uint8_t*)nvfp4_load_buf_;
            size_t gate_off = nvfp4_gate_w_bytes_ + nvfp4_gate_s_bytes_;
            size_t up_off   = gate_off + sizeof(float) + nvfp4_up_w_bytes_ + nvfp4_up_s_bytes_;
            size_t down_off = up_off   + sizeof(float) + nvfp4_down_w_bytes_ + nvfp4_down_s_bytes_;

            // Try NVMe bin file first; fall back to on-demand safetensors load
            void* expert_data = memory_->get_expert(layer_id, expert_id, expert_bytes_);
            bool loaded_to_gpu = false;
            float gate_g = 1.0f, up_g = 1.0f, down_g = 1.0f;

            if (expert_data) {
                // Data came from NVMe bin — read global scales from RAM buffer
                const uint8_t* rd = (const uint8_t*)expert_data;
                memcpy(&gate_g, rd + gate_off, sizeof(float));
                memcpy(&up_g,   rd + up_off,   sizeof(float));
                memcpy(&down_g, rd + down_off,  sizeof(float));
                cudaMemcpyAsync(weight_buf, expert_data, expert_bytes_,
                                 cudaMemcpyHostToDevice, stream);
                loaded_to_gpu = true;
            } else {
                // On-demand load: fills nvfp4_load_buf_ then copies to weight_buf
                if (!load_nvfp4_expert(layer_id, expert_id, weight_buf, stream))
                    continue;
                // Read global scales from CPU staging buffer (set by load_nvfp4_expert)
                memcpy(&gate_g, cpu_buf + gate_off, sizeof(float));
                memcpy(&up_g,   cpu_buf + up_off,   sizeof(float));
                memcpy(&down_g, cpu_buf + down_off,  sizeof(float));
                cudaStreamSynchronize(stream); // ensure H→D transfer done
                loaded_to_gpu = true;
                // Cache in RAM so subsequent tokens skip the safetensors read
                if (memory_) memory_->insert_expert(layer_id, expert_id,
                                                     nvfp4_load_buf_, expert_bytes_);
            }
            if (!loaded_to_gpu) continue;

            // GPU buffer layout mirrors CPU layout (same offsets)
            const uint8_t* d_gate_w = (const uint8_t*)weight_buf;
            const uint8_t* d_gate_s = d_gate_w + nvfp4_gate_w_bytes_;
            const uint8_t* d_up_w   = d_gate_s + nvfp4_gate_s_bytes_ + sizeof(float);
            const uint8_t* d_up_s   = d_up_w   + nvfp4_up_w_bytes_;
            const uint8_t* d_down_w = d_up_s   + nvfp4_up_s_bytes_   + sizeof(float);
            const uint8_t* d_down_s = d_down_w + nvfp4_down_w_bytes_;

            cuda::dequant_matvec_nvfp4(d_gate_w, d_gate_s, gate_g,
                                        norm_buf_, expert_gate_out_, moe_inter, hd, stream);
            cuda::dequant_matvec_nvfp4(d_up_w, d_up_s, up_g,
                                        norm_buf_, expert_up_out_, moe_inter, hd, stream);
            cuda::swiglu(expert_gate_out_, expert_gate_out_, expert_up_out_, moe_inter, stream);
            cuda::dequant_matvec_nvfp4(d_down_w, d_down_s, down_g,
                                        expert_gate_out_, expert_out, hd, moe_inter, stream);
        } else {
            // --- FP32 path ---
            void* expert_data = memory_->get_expert(layer_id, expert_id, expert_bytes_);
            if (!expert_data) continue;

            cudaMemcpyAsync(weight_buf, expert_data, expert_bytes_,
                             cudaMemcpyHostToDevice, stream);

            float* d_gate_w = (float*)weight_buf;
            float* d_up_w   = d_gate_w + (size_t)moe_inter * hd;
            float* d_down_w = d_up_w   + (size_t)moe_inter * hd;

            cuda::gemv_fp32(d_gate_w, norm_buf_, expert_gate_out_, moe_inter, hd, stream);
            cuda::gemv_fp32(d_up_w,   norm_buf_, expert_up_out_,   moe_inter, hd, stream);
            cuda::swiglu(expert_gate_out_, expert_gate_out_, expert_up_out_, moe_inter, stream);
            cuda::gemv_fp32(d_down_w, expert_gate_out_, expert_out, hd, moe_inter, stream);
        }
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
    if (embedding_is_bf16_) {
        cuda::gemv_bf16_to_fp32(lm_head_, norm_buf_, logits,
                                config_.vocab_size, config_.hidden_dim, stream);
    } else {
        cuda::gemv_fp32((const float*)lm_head_, norm_buf_, logits,
                        config_.vocab_size, config_.hidden_dim, stream);
    }
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
