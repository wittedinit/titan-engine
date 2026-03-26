#include "inference/kv_cache.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cstring>

namespace titan {

KVCache::~KVCache() {
    if (k_data_) cudaFree(k_data_);
    if (v_data_) cudaFree(v_data_);
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

float* KVCache::k_cache(uint32_t layer) const {
    if (!k_data_ || layer >= num_layers_) return nullptr;
    return k_data_ + layer * layer_stride_;
}

float* KVCache::v_cache(uint32_t layer) const {
    if (!v_data_ || layer >= num_layers_) return nullptr;
    return v_data_ + layer * layer_stride_;
}

void KVCache::update(uint32_t layer, int position,
                      const float* key, const float* value,
                      cudaStream_t stream) {
    if (!k_data_ || !v_data_) return;
    if (position < 0) return;  // Bug #11: Guard against negative position
    if (layer >= num_layers_ || position >= (int)max_seq_len_) return;

    size_t kv_size = (size_t)num_kv_heads_ * head_dim_;
    size_t offset = layer * layer_stride_ + position * kv_size;

    if (stream) {
        cudaMemcpyAsync(k_data_ + offset, key, kv_size * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(v_data_ + offset, value, kv_size * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy(k_data_ + offset, key, kv_size * sizeof(float),
                    cudaMemcpyDeviceToDevice);
        cudaMemcpy(v_data_ + offset, value, kv_size * sizeof(float),
                    cudaMemcpyDeviceToDevice);
    }

    if (position >= seq_len_) {
        seq_len_ = position + 1;
    }
}

size_t KVCache::memory_bytes() const {
    return 2 * (size_t)num_layers_ * layer_stride_ * sizeof(float);
}

void KVCache::clear() {
    seq_len_ = 0;
    if (k_data_) {
        size_t total_bytes = (size_t)num_layers_ * layer_stride_ * sizeof(float);
        cudaMemset(k_data_, 0, total_bytes);
        cudaMemset(v_data_, 0, total_bytes);
    }
}

} // namespace titan
