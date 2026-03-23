#pragma once

#include "core/types.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace titan {

// ============================================================================
// Safetensors Model Loader
//
// Reads tensor metadata from safetensors headers, then provides methods
// to load tensor data directly into GPU/CPU memory buffers.
// Supports both single-file and sharded models.
// ============================================================================

struct SafetensorsMeta {
    std::string name;
    std::string dtype_str;
    DType       dtype;
    std::vector<int64_t> shape;
    size_t      data_start = 0;  // Offset within the data region
    size_t      data_end = 0;
    size_t      byte_size() const { return data_end - data_start; }
    int64_t     numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader() = default;

    // Load model metadata (parses headers, doesn't load weight data)
    bool load(const std::string& model_dir);

    // Get model config
    const ModelConfig& config() const { return config_; }

    // Check if a tensor exists
    bool has_tensor(const std::string& name) const;

    // Get tensor metadata
    SafetensorsMeta get_meta(const std::string& name) const;

    // Read tensor data into a CPU buffer (caller must allocate)
    // Returns bytes read, or -1 on error
    ssize_t read_tensor_cpu(const std::string& name, void* dst, size_t dst_size);

    // Read tensor data directly into GPU memory via staging buffer
    // Allocates pinned staging buffer internally
    ssize_t read_tensor_gpu(const std::string& name, void* dst_gpu, size_t dst_size);

    // List all tensor names
    std::vector<std::string> tensor_names() const;

    // Model directory
    const std::string& model_dir() const { return model_dir_; }

private:
    ModelConfig config_;
    std::string model_dir_;

    struct ShardInfo {
        std::string path;
        size_t data_offset = 0;  // Bytes from file start to data region
    };
    std::vector<ShardInfo> shards_;

    // tensor_name → (shard_index, SafetensorsMeta)
    struct TensorLocation {
        size_t shard_idx;
        SafetensorsMeta meta;
    };
    std::unordered_map<std::string, TensorLocation> tensors_;

    bool load_single(const std::string& st_path);
    bool load_sharded(const std::string& index_path);
    bool parse_shard_header(size_t shard_idx);
};

} // namespace titan
