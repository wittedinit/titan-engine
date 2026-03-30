#include "model/loader.h"
#include "core/config.h"
#include "core/logging.h"

#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unordered_set>

#include <cuda_runtime.h>

namespace titan {

// ============================================================================
// Safetensors JSON header parser
// ============================================================================

static DType parse_st_dtype(const std::string& s) {
    if (s == "F32")     return DType::FP32;
    if (s == "F16")     return DType::FP16;
    if (s == "BF16")    return DType::BF16;
    if (s == "I8")      return DType::INT8;
    if (s == "U8")      return DType::INT8;   // Unsigned 8-bit — used for packed FP4 (NVFP4)
    if (s == "I32")     return DType::FP32;   // Treat I32 as FP32 size
    if (s == "F8_E4M3") return DType::FP8_E4M3;
    if (s == "F8_E5M2") return DType::FP8_E5M2;
    return DType::FP16; // Unknown dtype — assume 2 bytes, log at parse time
}

// Parse a safetensors JSON header into tensor metadata
static bool parse_header(const std::string& json,
                          std::vector<SafetensorsMeta>& out) {
    size_t pos = 0;

    while (pos < json.size()) {
        // Find opening quote of key
        auto q1 = json.find('"', pos);
        if (q1 == std::string::npos) break;
        auto q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        std::string key = json.substr(q1 + 1, q2 - q1 - 1);

        // Skip __metadata__
        if (key == "__metadata__") {
            // Skip the entire value (could be nested object)
            int depth = 0;
            pos = q2 + 1;
            while (pos < json.size()) {
                if (json[pos] == '{') depth++;
                else if (json[pos] == '}') {
                    depth--;
                    if (depth <= 0) { pos++; break; }
                }
                pos++;
            }
            continue;
        }

        // Find the value object (starts with {)
        auto obj_start = json.find('{', q2);
        if (obj_start == std::string::npos) break;

        // Find matching close brace
        int depth = 1;
        size_t obj_end = obj_start + 1;
        while (obj_end < json.size() && depth > 0) {
            if (json[obj_end] == '{') depth++;
            else if (json[obj_end] == '}') depth--;
            obj_end++;
        }

        std::string obj = json.substr(obj_start, obj_end - obj_start);

        SafetensorsMeta meta;
        meta.name = key;

        // Parse dtype
        auto dt_pos = obj.find("\"dtype\"");
        if (dt_pos != std::string::npos) {
            auto dq1 = obj.find('"', dt_pos + 7);
            auto dq2 = obj.find('"', dq1 + 1);
            if (dq1 != std::string::npos && dq2 != std::string::npos) {
                meta.dtype_str = obj.substr(dq1 + 1, dq2 - dq1 - 1);
                meta.dtype = parse_st_dtype(meta.dtype_str);
            }
        }

        // Parse shape
        auto sh_pos = obj.find("\"shape\"");
        if (sh_pos != std::string::npos) {
            auto br1 = obj.find('[', sh_pos);
            auto br2 = obj.find(']', br1);
            if (br1 != std::string::npos && br2 != std::string::npos) {
                std::string s = obj.substr(br1 + 1, br2 - br1 - 1);
                size_t sp = 0;
                while (sp < s.size()) {
                    while (sp < s.size() && !isdigit(s[sp]) && s[sp] != '-') sp++;
                    if (sp >= s.size()) break;
                    size_t start = sp;
                    while (sp < s.size() && (isdigit(s[sp]) || s[sp] == '-')) sp++;
                    meta.shape.push_back(std::stoll(s.substr(start, sp - start)));
                }
            }
        }

        // Parse data_offsets
        auto off_pos = obj.find("\"data_offsets\"");
        if (off_pos != std::string::npos) {
            auto br1 = obj.find('[', off_pos);
            auto br2 = obj.find(']', br1);
            if (br1 != std::string::npos && br2 != std::string::npos) {
                std::string s = obj.substr(br1 + 1, br2 - br1 - 1);
                size_t sp = 0;
                std::vector<size_t> offs;
                while (sp < s.size()) {
                    while (sp < s.size() && !isdigit(s[sp])) sp++;
                    if (sp >= s.size()) break;
                    size_t start = sp;
                    while (sp < s.size() && isdigit(s[sp])) sp++;
                    offs.push_back(std::stoull(s.substr(start, sp - start)));
                }
                if (offs.size() >= 2) {
                    meta.data_start = offs[0];
                    meta.data_end = offs[1];
                }
            }
        }

        out.push_back(meta);
        pos = obj_end;
    }

    return !out.empty();
}

// ============================================================================
// ModelLoader
// ============================================================================

bool ModelLoader::load(const std::string& model_dir) {
    model_dir_ = model_dir;

    // Load config
    config_ = load_model_config(model_dir + "/config.json");
    if (config_.hidden_dim == 0) {
        LOG_ERROR("Failed to load config.json from %s", model_dir.c_str());
        return false;
    }

    // Check for sharded vs single file
    std::string index_path = model_dir + "/model.safetensors.index.json";
    std::ifstream idx(index_path);
    if (idx.good()) {
        idx.close();
        return load_sharded(index_path);
    }

    std::string single_path = model_dir + "/model.safetensors";
    struct stat st;
    if (stat(single_path.c_str(), &st) == 0) {
        return load_single(single_path);
    }

    LOG_ERROR("No safetensors files found in %s", model_dir.c_str());
    return false;
}

bool ModelLoader::load_single(const std::string& st_path) {
    ShardInfo shard;
    shard.path = st_path;

    std::ifstream f(st_path, std::ios::binary);
    if (!f.good()) return false;

    uint64_t header_size;
    f.read(reinterpret_cast<char*>(&header_size), 8);
    shard.data_offset = 8 + header_size;

    std::string header(header_size, '\0');
    f.read(header.data(), header_size);

    std::vector<SafetensorsMeta> metas;
    if (!parse_header(header, metas)) {
        LOG_ERROR("Failed to parse safetensors header");
        return false;
    }

    size_t shard_idx = shards_.size();
    shards_.push_back(shard);

    for (auto& m : metas) {
        tensors_[m.name] = {shard_idx, m};
    }

    LOG_INFO("Loaded %zu tensors from %s", metas.size(), st_path.c_str());
    normalize_tensor_names();
    return true;
}

bool ModelLoader::load_sharded(const std::string& index_path) {
    std::ifstream f(index_path);
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    // Parse weight_map: {"tensor_name": "shard_filename", ...}
    auto wm_pos = json.find("\"weight_map\"");
    if (wm_pos == std::string::npos) {
        LOG_ERROR("No weight_map in index");
        return false;
    }

    auto brace = json.find('{', wm_pos + 12);
    // Find matching close
    int depth = 1;
    size_t end = brace + 1;
    while (end < json.size() && depth > 0) {
        if (json[end] == '{') depth++;
        else if (json[end] == '}') depth--;
        end++;
    }
    std::string wm = json.substr(brace, end - brace);

    // Parse tensor → filename pairs
    std::unordered_map<std::string, std::string> weight_map;
    size_t pos = 0;
    while (pos < wm.size()) {
        auto q1 = wm.find('"', pos);
        if (q1 == std::string::npos) break;
        auto q2 = wm.find('"', q1 + 1);
        auto q3 = wm.find('"', q2 + 1);
        auto q4 = wm.find('"', q3 + 1);
        if (q4 == std::string::npos) break;

        weight_map[wm.substr(q1 + 1, q2 - q1 - 1)] = wm.substr(q3 + 1, q4 - q3 - 1);
        pos = q4 + 1;
    }

    // Group by shard file and parse each shard's header
    std::unordered_map<std::string, size_t> file_to_idx;
    for (const auto& [tensor, file] : weight_map) {
        if (file_to_idx.find(file) == file_to_idx.end()) {
            size_t idx = shards_.size();
            file_to_idx[file] = idx;
            ShardInfo si;
            si.path = model_dir_ + "/" + file;
            shards_.push_back(si);
        }
    }

    // Parse headers for each shard
    for (size_t i = 0; i < shards_.size(); i++) {
        if (!parse_shard_header(i)) {
            LOG_ERROR("Failed to parse shard %s", shards_[i].path.c_str());
            return false;
        }
    }

    // Map tensor names from weight_map to their shard locations
    for (const auto& [tensor_name, file] : weight_map) {
        size_t shard_idx = file_to_idx[file];
        // Find tensor meta in this shard
        // (already populated by parse_shard_header)
    }

    // Also scan any safetensors files in the directory that aren't in the index.
    // Some models (e.g. NVFP4-quantized MoE models) have expert weight shards that
    // are not listed in model.safetensors.index.json but are present on disk.
    {
        std::unordered_set<std::string> indexed_files;
        for (const auto& [_, file] : weight_map) indexed_files.insert(file);

        // Glob for *.safetensors in model_dir_
        std::string dir = model_dir_;
        DIR* dp = opendir(dir.c_str());
        if (dp) {
            struct dirent* ep;
            while ((ep = readdir(dp))) {
                std::string fname(ep->d_name);
                if (fname.size() > 12 && fname.substr(fname.size() - 12) == ".safetensors") {
                    if (indexed_files.find(fname) == indexed_files.end()) {
                        // Unindexed shard — parse its header and add tensors
                        size_t shard_idx = shards_.size();
                        ShardInfo si;
                        si.path = dir + "/" + fname;
                        shards_.push_back(si);
                        parse_shard_header(shard_idx);  // errors silently skipped
                    }
                }
            }
            closedir(dp);
        }
    }

    LOG_INFO("Loaded %zu shards, %zu tensors", shards_.size(), tensors_.size());
    normalize_tensor_names();
    return true;
}

bool ModelLoader::parse_shard_header(size_t shard_idx) {
    auto& shard = shards_[shard_idx];

    std::ifstream f(shard.path, std::ios::binary);
    if (!f.good()) return false;

    uint64_t header_size;
    f.read(reinterpret_cast<char*>(&header_size), 8);
    shard.data_offset = 8 + header_size;

    std::string header(header_size, '\0');
    f.read(header.data(), header_size);

    std::vector<SafetensorsMeta> metas;
    if (!parse_header(header, metas)) return false;

    for (auto& m : metas) {
        tensors_[m.name] = {shard_idx, m};
    }

    return true;
}

void ModelLoader::normalize_tensor_names() {
    // Strip known LLM wrapper prefixes from tensors that have them.
    // For pure-text models ALL tensors match; for multimodal models (Kimi K2.5,
    // LLaVA, etc.) only LLM tensors carry the prefix — vision_tower.* and
    // mm_projector.* are left untouched. We strip when >= 50% of tensors carry
    // the prefix, which means it's intentional, not accidental.
    static const std::vector<std::string> prefixes = {
        "language_model.",   // Kimi K2.5, LLaVA-style multimodal wrappers
        "transformer.",      // Some GPT-style models
    };

    for (const auto& prefix : prefixes) {
        size_t match_count = 0;
        for (const auto& [name, _] : tensors_) {
            if (name.rfind(prefix, 0) == 0) match_count++;
        }

        // Strip if majority of tensors have this prefix
        if (match_count > tensors_.size() / 2) {
            LOG_INFO("Stripping tensor name prefix '%s' (%zu/%zu tensors)",
                     prefix.c_str(), match_count, tensors_.size());
            std::unordered_map<std::string, TensorLocation> remapped;
            remapped.reserve(tensors_.size());
            for (auto& [name, loc] : tensors_) {
                std::string new_name = (name.rfind(prefix, 0) == 0)
                                       ? name.substr(prefix.size())
                                       : name;
                loc.meta.name = new_name;
                remapped[new_name] = std::move(loc);
            }
            tensors_ = std::move(remapped);
            break;  // Only strip one prefix
        }
    }
}

bool ModelLoader::has_tensor(const std::string& name) const {
    return tensors_.count(name) > 0;
}

SafetensorsMeta ModelLoader::get_meta(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return {};
    return it->second.meta;
}

std::vector<std::string> ModelLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

// ============================================================================
// Read tensor data
// ============================================================================

ssize_t ModelLoader::read_tensor_cpu(const std::string& name, void* dst, size_t dst_size) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        LOG_ERROR("Tensor not found: %s", name.c_str());
        return -1;
    }

    const auto& loc = it->second;
    const auto& shard = shards_[loc.shard_idx];
    size_t bytes = loc.meta.byte_size();

    if (bytes > dst_size) {
        LOG_ERROR("Buffer too small for %s: need %zu, have %zu",
                  name.c_str(), bytes, dst_size);
        return -1;
    }

    // Read from file using pread
    int fd = open(shard.path.c_str(), O_RDONLY);
    if (fd < 0) {
        LOG_ERROR("Cannot open %s: %s", shard.path.c_str(), strerror(errno));
        return -1;
    }

    off_t file_offset = shard.data_offset + loc.meta.data_start;
    ssize_t total = 0;
    char* buf = (char*)dst;

    while ((size_t)total < bytes) {
        ssize_t n = pread(fd, buf + total, bytes - total, file_offset + total);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            break;
        }
        total += n;
    }

    close(fd);

    if ((size_t)total != bytes) {
        LOG_ERROR("Short read for %s: got %zd / %zu", name.c_str(), total, bytes);
        return -1;
    }

    return total;
}

ssize_t ModelLoader::read_tensor_gpu(const std::string& name, void* dst_gpu, size_t dst_size) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        LOG_ERROR("Tensor not found: %s", name.c_str());
        return -1;
    }

    size_t bytes = it->second.meta.byte_size();
    if (bytes > dst_size) {
        LOG_ERROR("GPU buffer too small for %s: need %zu, have %zu",
                  name.c_str(), bytes, dst_size);
        return -1;
    }

    // Use pinned staging buffer for fast CPU→GPU transfer
    void* staging = nullptr;
    cudaError_t err = cudaMallocHost(&staging, bytes);
    if (err != cudaSuccess) {
        // Fallback: regular malloc + cudaMemcpy
        staging = malloc(bytes);
        if (!staging) return -1;

        ssize_t read = read_tensor_cpu(name, staging, bytes);
        if (read > 0) {
            cudaMemcpy(dst_gpu, staging, bytes, cudaMemcpyHostToDevice);
        }
        free(staging);
        return read;
    }

    // Read to pinned memory, then async copy to GPU
    ssize_t read = read_tensor_cpu(name, staging, bytes);
    if (read > 0) {
        cudaMemcpy(dst_gpu, staging, bytes, cudaMemcpyHostToDevice);
    }

    cudaFreeHost(staging);
    return read;
}

} // namespace titan
