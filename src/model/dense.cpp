#include "model/dense.h"
#include "model/loader.h"
#include "compute/dispatch.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cstring>

// Forward declarations for gemv.cu
namespace titan { namespace cuda {
    void gemv_fp32(const float* A, const float* x, float* y,
                   int rows, int cols, cudaStream_t stream);
    void vector_add(float* y, const float* a, const float* b, int n, cudaStream_t stream);
    void init_cublas();
    void destroy_cublas();
}}

namespace titan {

DenseExecutor::~DenseExecutor() {
    free_scratch_buffers();
    cuda::destroy_cublas();
}

void DenseExecutor::allocate_scratch_buffers() {
    uint32_t qkv_dim = config_.num_attn_heads * config_.head_dim;
    uint32_t kv_dim = config_.num_kv_heads * config_.head_dim;
    uint32_t inter = config_.intermediate_dim;
    uint32_t hd = config_.hidden_dim;

    cudaMalloc(&q_buf_, qkv_dim * sizeof(float));
    cudaMalloc(&k_buf_, kv_dim * sizeof(float));
    cudaMalloc(&v_buf_, kv_dim * sizeof(float));
    cudaMalloc(&attn_out_, qkv_dim * sizeof(float));
    cudaMalloc(&gate_buf_, inter * sizeof(float));
    cudaMalloc(&up_buf_, inter * sizeof(float));
    cudaMalloc(&down_buf_, hd * sizeof(float));
    cudaMalloc(&norm_buf_, hd * sizeof(float));
}

void DenseExecutor::free_scratch_buffers() {
    auto sf = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    sf(q_buf_); sf(k_buf_); sf(v_buf_); sf(attn_out_);
    sf(gate_buf_); sf(up_buf_); sf(down_buf_); sf(norm_buf_);
}

// ============================================================================
// Load weights from safetensors using the ModelLoader
// ============================================================================

// Helper: allocate GPU buffer and load tensor data into it
static float* load_tensor_to_gpu(ModelLoader& loader, const std::string& name,
                                  size_t expected_elements = 0) {
    if (!loader.has_tensor(name)) {
        LOG_WARN("Tensor not found: %s (will use zeros)", name.c_str());
        if (expected_elements > 0) {
            float* buf = nullptr;
            cudaMalloc(&buf, expected_elements * sizeof(float));
            cudaMemset(buf, 0, expected_elements * sizeof(float));
            return buf;
        }
        return nullptr;
    }

    auto meta = loader.get_meta(name);
    size_t bytes = meta.byte_size();

    // For FP16/BF16 weights, we need to convert to FP32 for now
    // In the quantized path, weights stay in their native format
    bool needs_fp32_convert = (meta.dtype == DType::FP16 || meta.dtype == DType::BF16);

    float* gpu_buf = nullptr;

    if (needs_fp32_convert) {
        // Load as raw bytes, then convert FP16→FP32 on CPU
        // (In production, do this conversion on GPU)
        size_t num_elem = meta.numel();
        std::vector<uint16_t> fp16_data(num_elem);
        ssize_t read = loader.read_tensor_cpu(name, fp16_data.data(), bytes);
        if (read != (ssize_t)bytes) {
            LOG_ERROR("Failed to read %s", name.c_str());
            return nullptr;
        }

        // FP16 → FP32 conversion on CPU
        std::vector<float> fp32_data(num_elem);
        for (size_t i = 0; i < num_elem; i++) {
            // IEEE 754 half → float
            uint16_t h = fp16_data[i];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;

            uint32_t f;
            if (exp == 0) {
                if (mant == 0) {
                    f = sign << 31;
                } else {
                    // Denormalized
                    exp = 1;
                    while (!(mant & 0x400)) { mant <<= 1; exp--; }
                    mant &= 0x3FF;
                    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                f = (sign << 31) | 0x7F800000 | (mant << 13);
            } else {
                f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
            }

            memcpy(&fp32_data[i], &f, 4);
        }

        cudaMalloc(&gpu_buf, num_elem * sizeof(float));
        cudaMemcpy(gpu_buf, fp32_data.data(), num_elem * sizeof(float),
                    cudaMemcpyHostToDevice);

        LOG_DEBUG("  Loaded %s: [%s] %s → fp32 (%zu elements)",
                  name.c_str(), meta.dtype_str.c_str(),
                  meta.shape.size() > 0 ? std::to_string(meta.shape[0]).c_str() : "?",
                  num_elem);
    } else {
        // FP32: load directly
        cudaMalloc(&gpu_buf, bytes);
        loader.read_tensor_gpu(name, gpu_buf, bytes);
    }

    return gpu_buf;
}

// Load norm weights (always small, always FP32 or needs conversion)
static float* load_norm_weights(ModelLoader& loader, const std::string& name,
                                 uint32_t dim) {
    float* buf = load_tensor_to_gpu(loader, name, dim);
    if (!buf) {
        // Initialize to ones (identity norm) if not found
        cudaMalloc(&buf, dim * sizeof(float));
        std::vector<float> ones(dim, 1.0f);
        cudaMemcpy(buf, ones.data(), dim * sizeof(float), cudaMemcpyHostToDevice);
    }
    return buf;
}

bool DenseExecutor::load_weights(const std::string& model_path) {
    ModelLoader loader;
    if (!loader.load(model_path)) {
        LOG_ERROR("Failed to load model metadata");
        return false;
    }

    // Log available tensors for debugging
    auto names = loader.tensor_names();
    LOG_INFO("Model has %zu tensors", names.size());

    // Detect naming convention
    // Llama-style: "model.layers.{N}.self_attn.q_proj.weight"
    // Some models: "model.layers.{N}.attention.wq.weight"
    // Try to auto-detect
    std::string layer_prefix = "model.layers.";
    std::string attn_prefix = ".self_attn.";
    std::string ffn_prefix = ".mlp.";

    // Check if first layer exists with this naming
    std::string test = layer_prefix + "0" + attn_prefix + "q_proj.weight";
    if (!loader.has_tensor(test)) {
        // Try alternative naming
        test = layer_prefix + "0.attention.wq.weight";
        if (loader.has_tensor(test)) {
            attn_prefix = ".attention.";
            // Use wq/wk/wv/wo naming
        }
    }

    uint32_t hd = config_.hidden_dim;
    uint32_t qkv_dim = config_.num_attn_heads * config_.head_dim;
    uint32_t kv_dim = config_.num_kv_heads * config_.head_dim;
    uint32_t inter = config_.intermediate_dim;
    uint32_t vocab = config_.vocab_size;

    // Embedding
    LOG_INFO("Loading embedding...");
    embedding_ = load_tensor_to_gpu(loader, "model.embed_tokens.weight", vocab * hd);
    if (!embedding_) {
        // Try alternative names
        embedding_ = load_tensor_to_gpu(loader, "lm_head.weight", vocab * hd);
    }

    // Final norm
    final_norm_ = load_norm_weights(loader, "model.norm.weight", hd);

    // LM head (may be tied with embedding)
    if (loader.has_tensor("lm_head.weight")) {
        lm_head_ = load_tensor_to_gpu(loader, "lm_head.weight", vocab * hd);
    } else {
        lm_head_ = embedding_; // Weight tying
        LOG_INFO("LM head tied with embedding");
    }

    // Per-layer weights
    layer_weights_.resize(config_.num_layers);
    for (uint32_t l = 0; l < config_.num_layers; l++) {
        auto& lw = layer_weights_[l];
        std::string lp = layer_prefix + std::to_string(l);

        LOG_INFO("Loading layer %u/%u...", l + 1, config_.num_layers);

        // Norms
        lw.attn_norm = load_norm_weights(loader,
            lp + ".input_layernorm.weight", hd);
        lw.ffn_norm = load_norm_weights(loader,
            lp + ".post_attention_layernorm.weight", hd);

        // Attention projections
        lw.q_proj = load_tensor_to_gpu(loader,
            lp + attn_prefix + "q_proj.weight", qkv_dim * hd);
        lw.k_proj = load_tensor_to_gpu(loader,
            lp + attn_prefix + "k_proj.weight", kv_dim * hd);
        lw.v_proj = load_tensor_to_gpu(loader,
            lp + attn_prefix + "v_proj.weight", kv_dim * hd);
        lw.o_proj = load_tensor_to_gpu(loader,
            lp + attn_prefix + "o_proj.weight", hd * qkv_dim);

        // FFN projections (SwiGLU: gate + up + down)
        lw.gate_proj = load_tensor_to_gpu(loader,
            lp + ffn_prefix + "gate_proj.weight", inter * hd);
        lw.up_proj = load_tensor_to_gpu(loader,
            lp + ffn_prefix + "up_proj.weight", inter * hd);
        lw.down_proj = load_tensor_to_gpu(loader,
            lp + ffn_prefix + "down_proj.weight", hd * inter);
    }

    LOG_INFO("All weights loaded to GPU");
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

    config_ = load_model_config(model_path + "/config.json");
    if (config_.hidden_dim == 0) {
        LOG_ERROR("Failed to load model config");
        return false;
    }

    if (config_.head_dim == 0 && config_.num_attn_heads > 0)
        config_.head_dim = config_.hidden_dim / config_.num_attn_heads;

    // Initialize cuBLAS
    cuda::init_cublas();

    // Initialize KV cache
    if (!kv_cache_.initialize(config_.num_layers, config_.num_kv_heads,
                               config_.head_dim, runtime.max_context_len)) {
        LOG_ERROR("Failed to initialize KV cache");
        return false;
    }

    // Load weights from safetensors
    if (!load_weights(model_path)) {
        LOG_ERROR("Failed to load weights");
        return false;
    }

    allocate_scratch_buffers();

    LOG_INFO("Dense executor ready: %s (%uL, h=%u, heads=%u/%u, inter=%u)",
             config_.name.c_str(), config_.num_layers, config_.hidden_dim,
             config_.num_attn_heads, config_.num_kv_heads, config_.intermediate_dim);

    return true;
}

// ============================================================================
// Embedding
// ============================================================================

void DenseExecutor::embed_token(int token_id, float* output, cudaStream_t stream) {
    if (!embedding_ || token_id < 0 || token_id >= (int)config_.vocab_size) return;
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
// Forward Pass — Real implementation using cuBLAS + CUDA kernels
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

    // --- 1. Attention Input Norm ---
    cuda::rmsnorm(norm_buf_, hidden, lw.attn_norm, hd, 1e-5f, stream);

    // --- 2. Q/K/V Projections (cuBLAS sgemv) ---
    cuda::gemv_fp32((const float*)lw.q_proj, norm_buf_, q_buf_,
                    qkv_dim, hd, stream);
    cuda::gemv_fp32((const float*)lw.k_proj, norm_buf_, k_buf_,
                    kv_dim, hd, stream);
    cuda::gemv_fp32((const float*)lw.v_proj, norm_buf_, v_buf_,
                    kv_dim, hd, stream);

    // --- 3. RoPE ---
    cuda::apply_rope(q_buf_, k_buf_,
                     config_.num_attn_heads, config_.num_kv_heads,
                     config_.head_dim, position,
                     config_.rope_theta, config_.rope_scaling, stream);

    // --- 4. Update KV Cache ---
    kv_cache_.update(layer_id, position, k_buf_, v_buf_, stream);

    // --- 5. Attention ---
    int seq_len = kv_cache_.seq_len();
    if (seq_len < 1) seq_len = 1;
    cuda::attention_decode(q_buf_,
                           kv_cache_.k_cache(layer_id),
                           kv_cache_.v_cache(layer_id),
                           attn_out_,
                           config_.num_attn_heads, config_.num_kv_heads,
                           config_.head_dim, seq_len, stream);

    // --- 6. O Projection ---
    cuda::gemv_fp32((const float*)lw.o_proj, attn_out_, down_buf_,
                    hd, qkv_dim, stream);

    // --- 7. Residual Add ---
    // residual += down_buf_ (attention output)
    cuda::vector_add(residual, residual, down_buf_, hd, stream);

    // --- 8. FFN Input Norm ---
    cuda::rmsnorm(norm_buf_, residual, lw.ffn_norm, hd, 1e-5f, stream);

    // --- 9. Gate + Up Projections ---
    cuda::gemv_fp32((const float*)lw.gate_proj, norm_buf_, gate_buf_,
                    inter, hd, stream);
    cuda::gemv_fp32((const float*)lw.up_proj, norm_buf_, up_buf_,
                    inter, hd, stream);

    // --- 10. SwiGLU Activation ---
    cuda::swiglu(gate_buf_, gate_buf_, up_buf_, inter, stream);

    // --- 11. Down Projection ---
    cuda::gemv_fp32((const float*)lw.down_proj, gate_buf_, down_buf_,
                    hd, inter, stream);

    // --- 12. Residual Add ---
    cuda::vector_add(residual, residual, down_buf_, hd, stream);

    // Pass the updated residual as hidden for next layer
    cudaMemcpyAsync(hidden, residual, hd * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream);
}

// ============================================================================
// Logits
// ============================================================================

void DenseExecutor::compute_logits(const float* hidden, float* logits,
                                    void* cuda_stream) {
    cudaStream_t stream = (cudaStream_t)cuda_stream;

    // Final RMSNorm
    cuda::rmsnorm(norm_buf_, hidden, final_norm_, config_.hidden_dim, 1e-5f, stream);

    // LM head: [vocab_size, hidden_dim] @ [hidden_dim] → [vocab_size]
    cuda::gemv_fp32((const float*)lm_head_, norm_buf_, logits,
                    config_.vocab_size, config_.hidden_dim, stream);
}

// ============================================================================
// Memory Requirements
// ============================================================================

size_t DenseExecutor::attention_weight_bytes(uint32_t) const {
    size_t qkv = (size_t)(config_.num_attn_heads + 2 * config_.num_kv_heads) *
                 config_.head_dim * config_.hidden_dim;
    size_t o = (size_t)config_.hidden_dim * config_.num_attn_heads * config_.head_dim;
    return (qkv + o) * sizeof(float);
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
