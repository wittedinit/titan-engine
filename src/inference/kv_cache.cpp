#include "inference/kv_cache.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cstring>

namespace titan {

KVCache::~KVCache() {
    if (owns_memory_) {
        if (k_data_) cudaFree(k_data_);
        if (v_data_) cudaFree(v_data_);
    }
}

bool KVCache::initialize(uint32_t num_layers, uint32_t num_kv_heads,
                          uint32_t head_dim, uint32_t max_seq_len, DType dtype) {
    num_layers_ = num_layers;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    seq_len_ = 0;
    // Bug #10: Store the requested dtype for future FP16 KV cache support.
    // Currently always allocates float* (FP32). When FP16 support is added,
    // total_bytes calculation and update() memcpy sizes should use dtype_size().
    dtype_ = dtype;

    layer_stride_ = (size_t)max_seq_len * num_kv_heads * head_dim;
    size_t total_elements = (size_t)num_layers * layer_stride_;
    size_t total_bytes = total_elements * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&k_data_, total_bytes);
    if (err != cudaSuccess) {
        LOG_ERROR("KV cache: failed to allocate K cache (%.1f MB): %s",
                  total_bytes / 1e6, cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc(&v_data_, total_bytes);
    if (err != cudaSuccess) {
        LOG_ERROR("KV cache: failed to allocate V cache (%.1f MB): %s",
                  total_bytes / 1e6, cudaGetErrorString(err));
        cudaFree(k_data_);
        k_data_ = nullptr;
        return false;
    }

    cudaMemset(k_data_, 0, total_bytes);
    cudaMemset(v_data_, 0, total_bytes);

    LOG_INFO("KV cache: %u layers, %u heads, dim=%u, max_seq=%u (%.1f MB total)",
             num_layers, num_kv_heads, head_dim, max_seq_len,
             total_bytes * 2 / 1e6);

    return true;
}

bool KVCache::initialize_external(uint32_t num_layers, uint32_t num_kv_heads,
                                   uint32_t head_dim, uint32_t max_seq_len,
                                   float* k_buf, float* v_buf) {
    if (!k_buf || !v_buf) return false;
    num_layers_   = num_layers;
    num_kv_heads_ = num_kv_heads;
    head_dim_     = head_dim;
    max_seq_len_  = max_seq_len;
    seq_len_      = 0;
    layer_stride_ = (size_t)max_seq_len * num_kv_heads * head_dim;
    k_data_       = k_buf;
    v_data_       = v_buf;
    owns_memory_  = false;  // caller (VRAM pool) owns these pointers

    size_t total_bytes = (size_t)num_layers * layer_stride_ * sizeof(float);
    cudaMemset(k_data_, 0, total_bytes);
    cudaMemset(v_data_, 0, total_bytes);

    LOG_INFO("KV cache: %u layers, %u heads, dim=%u, max_seq=%u (%.1f MB, pool-backed)",
             num_layers, num_kv_heads, head_dim, max_seq_len,
             total_bytes * 2 / 1e6);
    return true;
}

bool KVCache::initialize_external_mla(uint32_t num_layers, uint32_t num_kv_heads,
                                       uint32_t k_hd, uint32_t v_hd,
                                       uint32_t max_seq_len,
                                       float* k_buf, float* v_buf) {
    if (!k_buf || !v_buf) return false;
    num_layers_    = num_layers;
    num_kv_heads_  = num_kv_heads;
    head_dim_      = k_hd;  // legacy field — use k_head_dim/v_head_dim for MLA
    k_head_dim_    = k_hd;
    v_head_dim_    = v_hd;
    max_seq_len_   = max_seq_len;
    seq_len_       = 0;
    layer_stride_  = (size_t)max_seq_len * num_kv_heads * k_hd;
    v_layer_stride_= (size_t)max_seq_len * num_kv_heads * v_hd;
    k_data_        = k_buf;
    v_data_        = v_buf;
    owns_memory_   = false;

    cudaMemset(k_data_, 0, (size_t)num_layers * layer_stride_ * sizeof(float));
    cudaMemset(v_data_, 0, (size_t)num_layers * v_layer_stride_ * sizeof(float));

    LOG_INFO("KV cache (MLA): %u layers, %u heads, k_dim=%u, v_dim=%u, max_seq=%u (%.1f MB, pool-backed)",
             num_layers, num_kv_heads, k_hd, v_hd, max_seq_len,
             ((size_t)num_layers * layer_stride_ + (size_t)num_layers * v_layer_stride_) * sizeof(float) / 1e6);
    return true;
}

float* KVCache::k_cache(uint32_t layer) const {
    if (!k_data_ || layer >= num_layers_) return nullptr;
    return k_data_ + layer * layer_stride_;
}

float* KVCache::v_cache(uint32_t layer) const {
    if (!v_data_ || layer >= num_layers_) return nullptr;
    size_t vs = v_layer_stride_ > 0 ? v_layer_stride_ : layer_stride_;
    return v_data_ + layer * vs;
}

void KVCache::update(uint32_t layer, int position,
                      const float* key, const float* value,
                      cudaStream_t stream) {
    if (!k_data_ || !v_data_) return;
    if (position < 0) return;
    if (layer >= num_layers_ || position >= (int)max_seq_len_) return;

    uint32_t k_hd = k_head_dim_ > 0 ? k_head_dim_ : head_dim_;
    uint32_t v_hd = v_head_dim_ > 0 ? v_head_dim_ : head_dim_;
    size_t k_size = (size_t)num_kv_heads_ * k_hd;
    size_t v_size = (size_t)num_kv_heads_ * v_hd;
    size_t k_offset = layer * layer_stride_ + position * k_size;
    size_t vs = v_layer_stride_ > 0 ? v_layer_stride_ : layer_stride_;
    size_t v_offset = layer * vs + position * v_size;

    if (stream) {
        cudaMemcpyAsync(k_data_ + k_offset, key, k_size * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(v_data_ + v_offset, value, v_size * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy(k_data_ + k_offset, key, k_size * sizeof(float),
                    cudaMemcpyDeviceToDevice);
        cudaMemcpy(v_data_ + v_offset, value, v_size * sizeof(float),
                    cudaMemcpyDeviceToDevice);
    }

    if (position >= seq_len_) {
        seq_len_ = position + 1;
    }
}

size_t KVCache::memory_bytes() const {
    size_t vs = v_layer_stride_ > 0 ? v_layer_stride_ : layer_stride_;
    return ((size_t)num_layers_ * layer_stride_ + (size_t)num_layers_ * vs) * sizeof(float);
}

void KVCache::clear() {
    seq_len_ = 0;
    if (k_data_) {
        cudaMemset(k_data_, 0, (size_t)num_layers_ * layer_stride_ * sizeof(float));
    }
    if (v_data_) {
        size_t vs = v_layer_stride_ > 0 ? v_layer_stride_ : layer_stride_;
        cudaMemset(v_data_, 0, (size_t)num_layers_ * vs * sizeof(float));
    }
}

} // namespace titan
