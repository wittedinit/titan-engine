#include "model/dense.h"
#include "compute/dispatch.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <fstream>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace titan {

DenseExecutor::~DenseExecutor() {
    free_scratch_buffers();
    // Weight memory is managed by MemoryManager, not freed here
}

void DenseExecutor::allocate_scratch_buffers() {
    uint32_t qkv_dim = config_.num_attn_heads * config_.head_dim;
    uint32_t kv_dim = config_.num_kv_heads * config_.head_dim;

    cudaMalloc(&q_buf_, qkv_dim * sizeof(float));
    cudaMalloc(&k_buf_, kv_dim * sizeof(float));
    cudaMalloc(&v_buf_, kv_dim * sizeof(float));
    cudaMalloc(&attn_out_, qkv_dim * sizeof(float));
    cudaMalloc(&gate_buf_, config_.intermediate_dim * sizeof(float));
    cudaMalloc(&up_buf_, config_.intermediate_dim * sizeof(float));
    cudaMalloc(&down_buf_, config_.hidden_dim * sizeof(float));
    cudaMalloc(&norm_buf_, config_.hidden_dim * sizeof(float));

    LOG_INFO("Scratch buffers allocated: %.1f MB",
             (qkv_dim * 2 + kv_dim * 2 + config_.intermediate_dim * 2 +
              config_.hidden_dim * 2) * sizeof(float) / 1e6);
}

void DenseExecutor::free_scratch_buffers() {
    auto safe_free = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    safe_free(q_buf_); safe_free(k_buf_); safe_free(v_buf_);
    safe_free(attn_out_); safe_free(gate_buf_); safe_free(up_buf_);
    safe_free(down_buf_); safe_free(norm_buf_);
}

// ============================================================================
// Weight Loading
// ============================================================================

bool DenseExecutor::load_weights(const std::string& model_path) {
    // For FP32/FP16 models: mmap the safetensors files and copy to GPU
    // For quantized models: the weights are already in the right format
    //
    // We use a simplified approach:
    // 1. Parse safetensors header to get tensor locations
    // 2. mmap the data region
    // 3. For each tensor, allocate in the appropriate tier and copy

    layer_weights_.resize(config_.num_layers);

    // For now, allocate placeholder weights on GPU (zeros)
    // The full loader integration will populate these from safetensors
    size_t hidden = config_.hidden_dim;
    size_t heads = config_.num_attn_heads;
    size_t kv_heads = config_.num_kv_heads;
    size_t head_dim = config_.head_dim;
    size_t inter = config_.intermediate_dim;
    size_t vocab = config_.vocab_size;

    size_t qkv_dim = heads * head_dim;
    size_t kv_proj_dim = kv_heads * head_dim;

    // Embedding: [vocab_size, hidden_dim] in FP32
    cudaMalloc(&embedding_, vocab * hidden * sizeof(float));
    cudaMemset(embedding_, 0, vocab * hidden * sizeof(float));

    // Final norm
    cudaMalloc(&final_norm_, hidden * sizeof(float));
    // Initialize to ones (identity normalization)
    std::vector<float> ones(hidden, 1.0f);
    cudaMemcpy(final_norm_, ones.data(), hidden * sizeof(float), cudaMemcpyHostToDevice);

    // LM head (reuse embedding for tied weights, or allocate separately)
    lm_head_ = embedding_; // Weight tying

    // Per-layer weights
    for (uint32_t l = 0; l < config_.num_layers; l++) {
        auto& lw = layer_weights_[l];

        // Norms (FP32, small)
        cudaMalloc(&lw.attn_norm, hidden * sizeof(float));
        cudaMalloc(&lw.ffn_norm, hidden * sizeof(float));
        cudaMemcpy(lw.attn_norm, ones.data(), hidden * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(lw.ffn_norm, ones.data(), hidden * sizeof(float), cudaMemcpyHostToDevice);

        // Attention projections (FP32 for now — quantized path uses different layout)
        cudaMalloc(&lw.q_proj, qkv_dim * hidden * sizeof(float));
        cudaMalloc(&lw.k_proj, kv_proj_dim * hidden * sizeof(float));
        cudaMalloc(&lw.v_proj, kv_proj_dim * hidden * sizeof(float));
        cudaMalloc(&lw.o_proj, hidden * qkv_dim * sizeof(float));
        cudaMemset(lw.q_proj, 0, qkv_dim * hidden * sizeof(float));
        cudaMemset(lw.k_proj, 0, kv_proj_dim * hidden * sizeof(float));
        cudaMemset(lw.v_proj, 0, kv_proj_dim * hidden * sizeof(float));
        cudaMemset(lw.o_proj, 0, hidden * qkv_dim * sizeof(float));

        // FFN projections
        cudaMalloc(&lw.gate_proj, inter * hidden * sizeof(float));
        cudaMalloc(&lw.up_proj, inter * hidden * sizeof(float));
        cudaMalloc(&lw.down_proj, hidden * inter * sizeof(float));
        cudaMemset(lw.gate_proj, 0, inter * hidden * sizeof(float));
        cudaMemset(lw.up_proj, 0, inter * hidden * sizeof(float));
        cudaMemset(lw.down_proj, 0, hidden * inter * sizeof(float));
    }

    LOG_INFO("Weight placeholders allocated for %u layers", config_.num_layers);
    return true;
}

// ============================================================================
// Initialization
// ============================================================================

bool DenseExecutor::initialize(const std::string& model_path,
                                MemoryManager& memory,
                                const RuntimeConfig& runtime) {
    memory_ = &memory;
    runtime_ = runtime;

    // Load model config
    config_ = load_model_config(model_path + "/config.json");
    if (config_.hidden_dim == 0) {
        LOG_ERROR("Failed to load model config");
        return false;
    }

    // Infer head_dim if not set
    if (config_.head_dim == 0 && config_.num_attn_heads > 0) {
        config_.head_dim = config_.hidden_dim / config_.num_attn_heads;
    }

    // Initialize KV cache
    if (!kv_cache_.initialize(config_.num_layers, config_.num_kv_heads,
                               config_.head_dim, runtime.max_context_len)) {
        LOG_ERROR("Failed to initialize KV cache");
        return false;
    }

    // Load weights
    if (!load_weights(model_path)) {
        LOG_ERROR("Failed to load weights");
        return false;
    }

    // Allocate scratch
    allocate_scratch_buffers();

    LOG_INFO("Dense executor initialized: %s (%uL, h=%u, heads=%u/%u)",
             config_.name.c_str(), config_.num_layers, config_.hidden_dim,
             config_.num_attn_heads, config_.num_kv_heads);

    return true;
}

// ============================================================================
// Embedding
// ============================================================================

void DenseExecutor::embed_token(int token_id, float* output, cudaStream_t stream) {
    if (token_id < 0 || token_id >= (int)config_.vocab_size) {
        LOG_ERROR("Token ID %d out of range [0, %u)", token_id, config_.vocab_size);
        return;
    }

    size_t offset = (size_t)token_id * config_.hidden_dim;
    if (stream) {
        cudaMemcpyAsync(output, embedding_ + offset,
                         config_.hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy(output, embedding_ + offset,
                    config_.hidden_dim * sizeof(float),
                    cudaMemcpyDeviceToDevice);
    }
}

// ============================================================================
// Forward Pass (single layer)
// ============================================================================

void DenseExecutor::forward_layer(float* hidden, float* residual,
                                   uint32_t layer_id, int position,
                                   void* cuda_stream) {
    if (layer_id >= config_.num_layers) return;

    cudaStream_t stream = (cudaStream_t)cuda_stream;
    const auto& lw = layer_weights_[layer_id];
    uint32_t hd = config_.hidden_dim;
    uint32_t qkv_dim = config_.num_attn_heads * config_.head_dim;
    uint32_t kv_dim = config_.num_kv_heads * config_.head_dim;
    uint32_t inter = config_.intermediate_dim;

    // --- Step 1: Attention norm ---
    cuda::rmsnorm(norm_buf_, hidden, lw.attn_norm, hd, 1e-6f, stream);

    // --- Step 2: Q/K/V projections ---
    // For FP32 weights, use cuBLAS or our matvec kernels
    // norm_buf_ → q_buf_, k_buf_, v_buf_
    // Q: [num_heads * head_dim, hidden_dim] @ [hidden_dim] → [num_heads * head_dim]
    // Since weights are FP32 for now, use a simple kernel dispatch

    // For the placeholder implementation, use the FP32 path
    // In production, this would dispatch to dequant_matvec_int4 for quantized weights

    // Simple GPU matvec for FP32: launch one block per output row
    // (The dequant kernels handle quantized weights)
    // For FP32, we can reuse the structure but without dequantization

    // We'll call the existing kernel infrastructure through dispatch.h
    // For now with FP32 weights, the dequant path doesn't apply.
    // Use cublasSgemv or our own simple kernel.

    // Placeholder: copy norm_buf_ to q/k/v (identity projection for testing)
    cudaMemcpyAsync(q_buf_, norm_buf_, std::min((size_t)qkv_dim, (size_t)hd) * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(k_buf_, norm_buf_, std::min((size_t)kv_dim, (size_t)hd) * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(v_buf_, norm_buf_, std::min((size_t)kv_dim, (size_t)hd) * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream);

    // --- Step 3: RoPE ---
    cuda::apply_rope(q_buf_, k_buf_,
                     config_.num_attn_heads, config_.num_kv_heads,
                     config_.head_dim, position,
                     config_.rope_theta, config_.rope_scaling, stream);

    // --- Step 4: Update KV cache ---
    kv_cache_.update(layer_id, position, k_buf_, v_buf_, stream);

    // --- Step 5: Attention ---
    cuda::attention_decode(q_buf_,
                           kv_cache_.k_cache(layer_id),
                           kv_cache_.v_cache(layer_id),
                           attn_out_,
                           config_.num_attn_heads, config_.num_kv_heads,
                           config_.head_dim, kv_cache_.seq_len(),
                           stream);

    // --- Step 6: O projection + residual ---
    // attn_out_ → down_buf_ (o_proj)
    // For now: identity (placeholder)
    cudaMemcpyAsync(down_buf_, attn_out_,
                     std::min((size_t)hd, (size_t)qkv_dim) * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream);

    // Residual: hidden = residual + o_proj_output, then norm
    cuda::fused_add_rmsnorm(norm_buf_, residual, down_buf_,
                             lw.ffn_norm, hd, 1e-6f, stream);
    // hidden now contains the post-attention, pre-FFN state

    // --- Step 7-10: FFN (SwiGLU) ---
    // gate = gate_proj @ norm_buf_
    // up   = up_proj @ norm_buf_
    // act  = SwiGLU(gate, up)
    // down = down_proj @ act

    // Placeholder: gate and up are identity for testing
    cudaMemcpyAsync(gate_buf_, norm_buf_,
                     std::min((size_t)inter, (size_t)hd) * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(up_buf_, norm_buf_,
                     std::min((size_t)inter, (size_t)hd) * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream);

    cuda::swiglu(gate_buf_, gate_buf_, up_buf_, inter, stream);

    // down_proj: gate_buf_ → down_buf_
    cudaMemcpyAsync(down_buf_, gate_buf_,
                     std::min((size_t)hd, (size_t)inter) * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream);

    // --- Step 11: Final residual ---
    // hidden = residual + down_buf_
    // (For the next layer, hidden will be norm'd at the start)
    // Use a simple add kernel
    cuda::fused_add_rmsnorm(hidden, residual, down_buf_,
                             lw.attn_norm, hd, 1e-6f, stream);
    // Note: This double-norms which isn't correct — in the full implementation,
    // we'd just do an add here and norm at the start of the next layer.
    // The fused_add_rmsnorm is used here as a functional placeholder.
}

// ============================================================================
// Logits
// ============================================================================

void DenseExecutor::compute_logits(const float* hidden, float* logits,
                                    void* cuda_stream) {
    cudaStream_t stream = (cudaStream_t)cuda_stream;

    // Final RMSNorm
    cuda::rmsnorm(norm_buf_, hidden, final_norm_, config_.hidden_dim, 1e-6f, stream);

    // LM head projection: [vocab_size, hidden_dim] @ [hidden_dim] → [vocab_size]
    // For tied weights, lm_head_ == embedding_
    // This is a large matvec — the most expensive single operation

    // Placeholder: fill logits with zeros (in production, use the actual projection)
    cudaMemsetAsync(logits, 0, config_.vocab_size * sizeof(float), stream);

    // In production with FP32 weights:
    // cublasSgemv(handle, CUBLAS_OP_N, vocab_size, hidden_dim,
    //             &one, lm_head_, vocab_size, norm_buf_, 1, &zero, logits, 1);
}

// ============================================================================
// Memory Requirements
// ============================================================================

size_t DenseExecutor::attention_weight_bytes(uint32_t) const {
    size_t qkv = (size_t)(config_.num_attn_heads + 2 * config_.num_kv_heads) *
                 config_.head_dim * config_.hidden_dim;
    size_t o = (size_t)config_.hidden_dim * config_.num_attn_heads * config_.head_dim;
    return (qkv + o) * sizeof(float); // FP32 for now
}

size_t DenseExecutor::ffn_weight_bytes(uint32_t) const {
    return (size_t)3 * config_.hidden_dim * config_.intermediate_dim * sizeof(float);
}

size_t DenseExecutor::kv_cache_bytes_per_token(uint32_t) const {
    return 2 * (size_t)config_.num_kv_heads * config_.head_dim * sizeof(float);
}

void DenseExecutor::update_kv_cache(uint32_t layer_id, int position,
                                     const float* key, const float* value) {
    kv_cache_.update(layer_id, position, key, value);
}

} // namespace titan
