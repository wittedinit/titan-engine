#include "model/gguf_loader.h"
#include "core/config.h"
#include "core/logging.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include <algorithm>

#include <cuda_runtime.h>

namespace titan {

// GGUF magic and version
static constexpr uint32_t GGUF_MAGIC = 0x46475547; // "GGUF"
static constexpr uint32_t GGUF_VERSION_3 = 3;

// GGUF metadata value types
enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

static size_t gguf_value_type_size(GGUFValueType t) {
    switch (t) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

// ============================================================================
// Low-level read helpers
// ============================================================================

std::string GGUFLoader::read_string(int fd) {
    uint64_t len;
    if (read(fd, &len, 8) != 8) return "";
    std::string s(len, '\0');
    if (read(fd, s.data(), len) != (ssize_t)len) return "";
    return s;
}

uint64_t GGUFLoader::read_uint(int fd, int bytes) {
    uint64_t val = 0;
    read(fd, &val, bytes);
    return val;
}

// ============================================================================
// Parse GGUF header
// ============================================================================

bool GGUFLoader::parse_header(int fd) {
    // Magic
    uint32_t magic;
    if (read(fd, &magic, 4) != 4 || magic != GGUF_MAGIC) {
        LOG_ERROR("Not a GGUF file (bad magic)");
        return false;
    }

    // Version
    uint32_t version;
    read(fd, &version, 4);
    if (version < GGUF_VERSION_3) {
        LOG_ERROR("GGUF version %u not supported (need >= 3)", version);
        return false;
    }

    // Counts
    uint64_t tensor_count, metadata_count;
    read(fd, &tensor_count, 8);
    read(fd, &metadata_count, 8);

    LOG_INFO("GGUF v%u: %lu tensors, %lu metadata entries",
             version, tensor_count, metadata_count);

    // Parse metadata
    for (uint64_t i = 0; i < metadata_count; i++) {
        std::string key = read_string(fd);
        uint32_t value_type;
        read(fd, &value_type, 4);

        switch (value_type) {
            case GGUF_TYPE_STRING: {
                std::string val = read_string(fd);
                metadata_str_[key] = val;
                if (key == "general.architecture") arch_ = val;
                break;
            }
            case GGUF_TYPE_UINT32: {
                uint32_t val;
                read(fd, &val, 4);
                metadata_uint_[key] = val;
                break;
            }
            case GGUF_TYPE_UINT64: {
                uint64_t val;
                read(fd, &val, 8);
                metadata_uint_[key] = val;
                break;
            }
            case GGUF_TYPE_INT32: {
                int32_t val;
                read(fd, &val, 4);
                metadata_uint_[key] = (uint64_t)val;
                break;
            }
            case GGUF_TYPE_FLOAT32: {
                float val;
                read(fd, &val, 4);
                metadata_float_[key] = val;
                break;
            }
            case GGUF_TYPE_FLOAT64: {
                double val;
                read(fd, &val, 8);
                metadata_float_[key] = val;
                break;
            }
            case GGUF_TYPE_BOOL: {
                uint8_t val;
                read(fd, &val, 1);
                metadata_uint_[key] = val;
                break;
            }
            case GGUF_TYPE_ARRAY: {
                // Read array type and length, then skip the data
                uint32_t arr_type;
                uint64_t arr_len;
                read(fd, &arr_type, 4);
                read(fd, &arr_len, 8);

                if (arr_type == GGUF_TYPE_STRING) {
                    for (uint64_t j = 0; j < arr_len; j++) {
                        read_string(fd); // Skip
                    }
                } else {
                    size_t elem_size = gguf_value_type_size((GGUFValueType)arr_type);
                    lseek(fd, arr_len * elem_size, SEEK_CUR);
                }
                break;
            }
            default: {
                size_t sz = gguf_value_type_size((GGUFValueType)value_type);
                if (sz > 0) lseek(fd, sz, SEEK_CUR);
                break;
            }
        }
    }

    // Parse tensor descriptors
    for (uint64_t i = 0; i < tensor_count; i++) {
        GGUFTensorMeta meta;
        meta.name = read_string(fd);

        uint32_t n_dims;
        read(fd, &n_dims, 4);
        meta.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim;
            read(fd, &dim, 8);
            meta.shape[d] = dim;
        }

        uint32_t type;
        read(fd, &type, 4);
        meta.type = (GGMLType)type;

        read(fd, &meta.offset, 8);

        // Compute size
        int64_t numel = meta.numel();
        size_t block_size = ggml_type_block_size(meta.type);
        size_t block_bytes = ggml_type_block_bytes(meta.type);
        if (block_size > 0 && block_bytes > 0) {
            meta.size_bytes = (numel / block_size) * block_bytes;
        } else {
            meta.size_bytes = numel * 4; // Fallback: assume FP32
        }

        tensors_[meta.name] = meta;
    }

    // Data starts at next alignment boundary (default 32 bytes)
    off_t current = lseek(fd, 0, SEEK_CUR);
    size_t alignment = 32;
    if (metadata_uint_.count("general.alignment")) {
        alignment = metadata_uint_["general.alignment"];
    }
    data_offset_ = (current + alignment - 1) & ~(alignment - 1);

    return true;
}

// ============================================================================
// Public API
// ============================================================================

bool GGUFLoader::load(const std::string& gguf_path) {
    file_path_ = gguf_path;

    int fd = open(gguf_path.c_str(), O_RDONLY);
    if (fd < 0) {
        LOG_ERROR("Cannot open GGUF file: %s", gguf_path.c_str());
        return false;
    }

    bool ok = parse_header(fd);
    close(fd);

    if (ok) {
        LOG_INFO("GGUF loaded: arch=%s, %zu tensors, data at offset %zu",
                 arch_.c_str(), tensors_.size(), data_offset_);
    }
    return ok;
}

ModelConfig GGUFLoader::to_model_config() const {
    ModelConfig cfg;
    cfg.name = arch_;

    auto get_uint = [this](const std::string& key, uint64_t def = 0) -> uint64_t {
        // Try arch-prefixed key first, then generic
        std::string prefixed = arch_ + "." + key;
        auto it = metadata_uint_.find(prefixed);
        if (it != metadata_uint_.end()) return it->second;
        it = metadata_uint_.find(key);
        if (it != metadata_uint_.end()) return it->second;
        return def;
    };
    auto get_float = [this](const std::string& key, double def = 0) -> double {
        std::string prefixed = arch_ + "." + key;
        auto it = metadata_float_.find(prefixed);
        if (it != metadata_float_.end()) return it->second;
        it = metadata_float_.find(key);
        if (it != metadata_float_.end()) return it->second;
        return def;
    };

    cfg.hidden_dim = get_uint("embedding_length", 4096);
    cfg.num_layers = get_uint("block_count", 32);
    cfg.num_attn_heads = get_uint("attention.head_count", 32);
    cfg.num_kv_heads = get_uint("attention.head_count_kv", cfg.num_attn_heads);
    cfg.head_dim = cfg.hidden_dim / cfg.num_attn_heads;
    cfg.intermediate_dim = get_uint("feed_forward_length", cfg.hidden_dim * 4);
    cfg.vocab_size = get_uint("vocab_size", 32000);
    cfg.rope_theta = get_float("rope.freq_base", 10000.0);
    cfg.max_position = get_uint("context_length", 131072);

    // MoE fields
    cfg.num_experts = get_uint("expert_count", 0);
    cfg.experts_per_tok = get_uint("expert_used_count", 0);
    if (cfg.num_experts > 0 && cfg.experts_per_tok > 0) {
        cfg.model_type = ModelType::MOE;
    }

    return cfg;
}

bool GGUFLoader::has_tensor(const std::string& name) const {
    return tensors_.count(name) > 0;
}

GGUFTensorMeta GGUFLoader::get_meta(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return {};
    return it->second;
}

std::vector<std::string> GGUFLoader::tensor_names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : tensors_) names.push_back(name);
    std::sort(names.begin(), names.end());
    return names;
}

ssize_t GGUFLoader::read_tensor_cpu(const std::string& name, void* dst, size_t dst_size) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return -1;

    const auto& meta = it->second;
    if (meta.size_bytes > dst_size) return -1;

    int fd = open(file_path_.c_str(), O_RDONLY);
    if (fd < 0) return -1;

    off_t offset = data_offset_ + meta.offset;
    ssize_t total = 0;
    char* buf = (char*)dst;
    while ((size_t)total < meta.size_bytes) {
        ssize_t n = pread(fd, buf + total, meta.size_bytes - total, offset + total);
        if (n <= 0) { if (n < 0 && errno == EINTR) continue; break; }
        total += n;
    }
    close(fd);
    return total;
}

ssize_t GGUFLoader::read_tensor_gpu(const std::string& name, void* dst_gpu, size_t dst_size) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return -1;

    size_t bytes = it->second.size_bytes;
    if (bytes > dst_size) return -1;

    void* staging = nullptr;
    cudaMallocHost(&staging, bytes);
    if (!staging) {
        staging = malloc(bytes);
        ssize_t n = read_tensor_cpu(name, staging, bytes);
        if (n > 0) cudaMemcpy(dst_gpu, staging, bytes, cudaMemcpyHostToDevice);
        free(staging);
        return n;
    }

    ssize_t n = read_tensor_cpu(name, staging, bytes);
    if (n > 0) cudaMemcpy(dst_gpu, staging, bytes, cudaMemcpyHostToDevice);
    cudaFreeHost(staging);
    return n;
}

} // namespace titan
