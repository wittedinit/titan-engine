#pragma once

#include "core/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace titan {

// ============================================================================
// GGUF Format Loader
//
// GGUF (GGML Universal Format) is the standard format for llama.cpp models.
// Supports quantized weights (Q4_K, Q3_K, Q8_0, etc.) in a single file.
//
// File layout:
//   - Magic: "GGUF" (4 bytes)
//   - Version: uint32
//   - Tensor count: uint64
//   - Metadata KV count: uint64
//   - Metadata key-value pairs
//   - Tensor descriptors (name, shape, type, offset)
//   - Padding to alignment
//   - Tensor data (contiguous, aligned)
// ============================================================================

// GGUF data types
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
};

// Block sizes for quantized types
inline size_t ggml_type_block_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return 1;
        case GGMLType::F16:  return 1;
        case GGMLType::BF16: return 1;
        case GGMLType::Q4_0: return 32;
        case GGMLType::Q4_1: return 32;
        case GGMLType::Q5_0: return 32;
        case GGMLType::Q5_1: return 32;
        case GGMLType::Q8_0: return 32;
        case GGMLType::Q8_1: return 32;
        case GGMLType::Q2_K: return 256;
        case GGMLType::Q3_K: return 256;
        case GGMLType::Q4_K: return 256;
        case GGMLType::Q5_K: return 256;
        case GGMLType::Q6_K: return 256;
        case GGMLType::Q8_K: return 256;
        default: return 1;
    }
}

// Bytes per block for quantized types
inline size_t ggml_type_block_bytes(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return 4;
        case GGMLType::F16:  return 2;
        case GGMLType::BF16: return 2;
        case GGMLType::Q4_0: return 18;   // 32 * 4/8 + 2 (scale)
        case GGMLType::Q4_1: return 20;   // 32 * 4/8 + 2 + 2
        case GGMLType::Q8_0: return 34;   // 32 * 1 + 2
        case GGMLType::Q2_K: return 84;
        case GGMLType::Q3_K: return 110;
        case GGMLType::Q4_K: return 144;
        case GGMLType::Q5_K: return 176;
        case GGMLType::Q6_K: return 210;
        case GGMLType::Q8_K: return 292;
        default: return 0;
    }
}

struct GGUFTensorMeta {
    std::string name;
    GGMLType type;
    std::vector<int64_t> shape;
    size_t offset;  // Byte offset from data start
    size_t size_bytes;

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

class GGUFLoader {
public:
    bool load(const std::string& gguf_path);

    // Model metadata
    const std::string& model_arch() const { return arch_; }
    ModelConfig to_model_config() const;

    // Tensor access
    bool has_tensor(const std::string& name) const;
    GGUFTensorMeta get_meta(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

    // Read raw tensor data to CPU buffer
    ssize_t read_tensor_cpu(const std::string& name, void* dst, size_t dst_size);

    // Read tensor data to GPU
    ssize_t read_tensor_gpu(const std::string& name, void* dst_gpu, size_t dst_size);

private:
    std::string file_path_;
    size_t data_offset_ = 0;   // Byte offset to tensor data start

    // Metadata
    std::string arch_;
    std::unordered_map<std::string, std::string> metadata_str_;
    std::unordered_map<std::string, uint64_t> metadata_uint_;
    std::unordered_map<std::string, double> metadata_float_;

    // Tensors
    std::unordered_map<std::string, GGUFTensorMeta> tensors_;

    // Internal parsing
    bool parse_header(int fd);
    std::string read_string(int fd);
    uint64_t read_uint(int fd, int bytes);
};

} // namespace titan
