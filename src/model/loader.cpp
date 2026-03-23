#include "core/types.h"
#include "core/config.h"
#include "core/logging.h"

#include <fstream>
#include <cstring>
#include <vector>
#include <unordered_map>

namespace titan {

// ============================================================================
// Safetensors Format Reader
//
// Safetensors is a simple format:
//   [8 bytes] header_size (uint64 LE)
//   [header_size bytes] JSON header mapping tensor names → {dtype, shape, offsets}
//   [remaining bytes] raw tensor data
// ============================================================================

struct SafetensorsFile {
    std::string path;
    size_t header_size = 0;
    size_t data_offset = 0;
    int fd = -1;

    struct TensorMeta {
        std::string name;
        std::string dtype_str;
        std::vector<int64_t> shape;
        size_t data_start = 0;
        size_t data_end = 0;
    };
    std::vector<TensorMeta> tensors;
};

static DType parse_safetensors_dtype(const std::string& s) {
    if (s == "F32")  return DType::FP32;
    if (s == "F16")  return DType::FP16;
    if (s == "BF16") return DType::BF16;
    if (s == "I8")   return DType::INT8;
    if (s == "F8_E4M3") return DType::FP8_E4M3;
    return DType::FP16; // Default
}

// Minimal JSON parser for safetensors header
// Extracts tensor names, dtypes, shapes, and offsets
static bool parse_safetensors_header(const std::string& json, SafetensorsFile& sf) {
    // This is a simplified parser for the safetensors JSON format
    // Real implementation would use a proper JSON parser (rapidjson, nlohmann)

    size_t pos = 0;
    while (pos < json.size()) {
        // Find tensor name (key)
        auto q1 = json.find('"', pos);
        if (q1 == std::string::npos) break;
        auto q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        std::string key = json.substr(q1 + 1, q2 - q1 - 1);

        // Skip __metadata__ entries
        if (key == "__metadata__") {
            pos = json.find('}', q2) + 1;
            continue;
        }

        SafetensorsFile::TensorMeta meta;
        meta.name = key;

        // Find dtype
        auto dtype_pos = json.find("\"dtype\"", q2);
        if (dtype_pos != std::string::npos) {
            auto dq1 = json.find('"', dtype_pos + 7);
            auto dq2 = json.find('"', dq1 + 1);
            if (dq1 != std::string::npos && dq2 != std::string::npos) {
                meta.dtype_str = json.substr(dq1 + 1, dq2 - dq1 - 1);
            }
        }

        // Find shape
        auto shape_pos = json.find("\"shape\"", q2);
        if (shape_pos != std::string::npos) {
            auto bracket_start = json.find('[', shape_pos);
            auto bracket_end = json.find(']', bracket_start);
            if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
                std::string shape_str = json.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                // Parse comma-separated integers
                size_t sp = 0;
                while (sp < shape_str.size()) {
                    while (sp < shape_str.size() && !isdigit(shape_str[sp])) sp++;
                    if (sp >= shape_str.size()) break;
                    meta.shape.push_back(std::stoll(shape_str.substr(sp)));
                    while (sp < shape_str.size() && isdigit(shape_str[sp])) sp++;
                }
            }
        }

        // Find data_offsets
        auto off_pos = json.find("\"data_offsets\"", q2);
        if (off_pos != std::string::npos) {
            auto ob_start = json.find('[', off_pos);
            auto ob_end = json.find(']', ob_start);
            if (ob_start != std::string::npos && ob_end != std::string::npos) {
                std::string off_str = json.substr(ob_start + 1, ob_end - ob_start - 1);
                size_t sp = 0;
                std::vector<size_t> offsets;
                while (sp < off_str.size()) {
                    while (sp < off_str.size() && !isdigit(off_str[sp])) sp++;
                    if (sp >= off_str.size()) break;
                    offsets.push_back(std::stoull(off_str.substr(sp)));
                    while (sp < off_str.size() && isdigit(off_str[sp])) sp++;
                }
                if (offsets.size() >= 2) {
                    meta.data_start = offsets[0];
                    meta.data_end = offsets[1];
                }
            }
        }

        sf.tensors.push_back(meta);

        // Advance past the current entry
        pos = json.find('}', q2);
        if (pos != std::string::npos) pos++;
    }

    return !sf.tensors.empty();
}

// ============================================================================
// Model Loading Pipeline
// ============================================================================

// Load safetensors index (for sharded models, reads model.safetensors.index.json)
// Returns list of shard files and tensor→shard mapping

struct ModelLoader {
    ModelConfig config;
    std::string model_dir;
    std::vector<SafetensorsFile> shards;
    std::unordered_map<std::string, size_t> tensor_to_shard; // tensor_name → shard index

    bool load(const std::string& path) {
        model_dir = path;

        // Load model config
        config = load_model_config(path + "/config.json");

        // Check for sharded model (index file)
        std::string index_path = path + "/model.safetensors.index.json";
        std::ifstream index_file(index_path);

        if (index_file.good()) {
            return load_sharded(index_path);
        }

        // Single file model
        std::string st_path = path + "/model.safetensors";
        return load_single(st_path);
    }

    bool load_single(const std::string& st_path) {
        std::ifstream f(st_path, std::ios::binary);
        if (!f.good()) {
            LOG_ERROR("Cannot open %s", st_path.c_str());
            return false;
        }

        SafetensorsFile sf;
        sf.path = st_path;

        // Read header size
        uint64_t header_size;
        f.read(reinterpret_cast<char*>(&header_size), 8);
        sf.header_size = header_size;
        sf.data_offset = 8 + header_size;

        // Read header JSON
        std::string header(header_size, '\0');
        f.read(header.data(), header_size);

        if (!parse_safetensors_header(header, sf)) {
            LOG_ERROR("Failed to parse safetensors header");
            return false;
        }

        LOG_INFO("Loaded %zu tensors from %s", sf.tensors.size(), st_path.c_str());

        for (size_t i = 0; i < sf.tensors.size(); i++) {
            tensor_to_shard[sf.tensors[i].name] = 0;
        }
        shards.push_back(std::move(sf));

        return true;
    }

    bool load_sharded(const std::string& index_path) {
        // Read index JSON to find shard files
        std::ifstream f(index_path);
        std::string json((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

        // Parse weight_map from index
        auto wm_pos = json.find("\"weight_map\"");
        if (wm_pos == std::string::npos) {
            LOG_ERROR("No weight_map in index file");
            return false;
        }

        // Extract tensor→filename mapping
        std::unordered_map<std::string, std::string> weight_map;
        size_t pos = json.find('{', wm_pos + 12);
        size_t end = json.find('}', pos);
        // Simplified parsing
        std::string wm_json = json.substr(pos, end - pos + 1);

        size_t p = 0;
        while (p < wm_json.size()) {
            auto q1 = wm_json.find('"', p);
            if (q1 == std::string::npos) break;
            auto q2 = wm_json.find('"', q1 + 1);
            auto q3 = wm_json.find('"', q2 + 1);
            auto q4 = wm_json.find('"', q3 + 1);
            if (q4 == std::string::npos) break;

            std::string tensor = wm_json.substr(q1 + 1, q2 - q1 - 1);
            std::string file = wm_json.substr(q3 + 1, q4 - q3 - 1);
            weight_map[tensor] = file;
            p = q4 + 1;
        }

        // Group by shard file
        std::unordered_map<std::string, size_t> file_to_shard;
        for (const auto& [tensor, file] : weight_map) {
            if (file_to_shard.find(file) == file_to_shard.end()) {
                size_t idx = shards.size();
                file_to_shard[file] = idx;

                SafetensorsFile sf;
                sf.path = model_dir + "/" + file;
                shards.push_back(sf);
            }
            tensor_to_shard[tensor] = file_to_shard[file];
        }

        LOG_INFO("Model has %zu shards, %zu tensors", shards.size(), weight_map.size());

        // Load headers for each shard
        for (auto& sf : shards) {
            std::ifstream sf_file(sf.path, std::ios::binary);
            if (!sf_file.good()) {
                LOG_ERROR("Cannot open shard %s", sf.path.c_str());
                return false;
            }

            uint64_t header_size;
            sf_file.read(reinterpret_cast<char*>(&header_size), 8);
            sf.header_size = header_size;
            sf.data_offset = 8 + header_size;

            std::string header(header_size, '\0');
            sf_file.read(header.data(), header_size);
            parse_safetensors_header(header, sf);
        }

        return true;
    }

    // Get tensor descriptor by name
    TensorDesc get_tensor_desc(const std::string& name) const {
        TensorDesc desc;
        desc.name = name;

        auto it = tensor_to_shard.find(name);
        if (it == tensor_to_shard.end()) return desc;

        const auto& sf = shards[it->second];
        for (const auto& meta : sf.tensors) {
            if (meta.name == name) {
                desc.dtype = parse_safetensors_dtype(meta.dtype_str);
                desc.shape = meta.shape;
                desc.byte_offset = sf.data_offset + meta.data_start;
                desc.byte_size = meta.data_end - meta.data_start;
                break;
            }
        }

        return desc;
    }
};

} // namespace titan
