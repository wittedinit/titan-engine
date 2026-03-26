#pragma once

#include "core/types.h"
#include <cuda_runtime.h>

namespace titan {

// ============================================================================
// KV Cache — Stores key/value pairs for all layers across the sequence
//
// Layout: [num_layers][max_seq_len][num_kv_heads][head_dim]
// Stored in GPU memory (VRAM) for fast attention access.
// ============================================================================

class KVCache {
public:
    KVCache() = default;
    ~KVCache();

    // Initialize cache for given model dimensions
    bool initialize(uint32_t num_layers, uint32_t num_kv_heads,
                    uint32_t head_dim, uint32_t max_seq_len,
                    DType dtype = DType::FP32);

    // Get pointer to K cache for a specific layer
    // Returns pointer to [max_seq_len, num_kv_heads, head_dim]
    float* k_cache(uint32_t layer) const;

    // Get pointer to V cache for a specific layer
    float* v_cache(uint32_t layer) const;

    // Update KV cache at a specific position for a layer
    void update(uint32_t layer, int position,
                const float* key, const float* value,
                cudaStream_t stream = nullptr);

    // Current sequence length (number of tokens stored)
    int seq_len() const { return seq_len_; }
    void set_seq_len(int len) { seq_len_ = len; }

    // Dimensions
    uint32_t num_layers() const { return num_layers_; }
    uint32_t num_kv_heads() const { return num_kv_heads_; }
    uint32_t head_dim() const { return head_dim_; }
    uint32_t max_seq_len() const { return max_seq_len_; }

    // Total memory used
    size_t memory_bytes() const;

    // Clear cache (reset sequence)
    void clear();

private:
    float*      k_data_ = nullptr;  // [num_layers * max_seq_len * num_kv_heads * head_dim]
    float*      v_data_ = nullptr;
    uint32_t    num_layers_ = 0;
    uint32_t    num_kv_heads_ = 0;
    uint32_t    head_dim_ = 0;
    uint32_t    max_seq_len_ = 0;
    int         seq_len_ = 0;
    size_t      layer_stride_ = 0;  // Elements per layer
    DType       dtype_ = DType::FP32; // Stored for future FP16 KV cache support
};

} // namespace titan
