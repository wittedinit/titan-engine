#include "core/config.h"
#include "core/logging.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace titan {

// Minimal JSON parser for HuggingFace config.json
// We avoid external dependencies — this handles the flat key-value structure
// that HF configs use.

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r\"");
    size_t end = s.find_last_not_of(" \t\n\r\",");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

static int64_t json_int(const std::string& json, const std::string& key, int64_t def = 0) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    try {
        return std::stoll(json.substr(pos));
    } catch (...) {
        return def;
    }
}

static double json_float(const std::string& json, const std::string& key, double def = 0.0) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    try {
        return std::stod(json.substr(pos));
    } catch (...) {
        return def;
    }
}

static std::string json_string(const std::string& json, const std::string& key, const std::string& def = "") {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return def;
    auto q1 = json.find('"', pos + 1);
    if (q1 == std::string::npos) return def;
    auto q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return def;
    return json.substr(q1 + 1, q2 - q1 - 1);
}

ModelConfig load_model_config(const std::string& config_path) {
    ModelConfig cfg;

    std::ifstream f(config_path);
    if (!f.good()) {
        LOG_ERROR("Cannot open model config: %s", config_path.c_str());
        return cfg;
    }

    std::stringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();

    // Parse common HuggingFace fields
    cfg.name = json_string(json, "model_type", "unknown");
    cfg.hidden_dim = (uint32_t)json_int(json, "hidden_size", 4096);
    cfg.num_layers = (uint32_t)json_int(json, "num_hidden_layers", 32);
    cfg.num_attn_heads = (uint32_t)json_int(json, "num_attention_heads", 32);
    cfg.num_kv_heads = (uint32_t)json_int(json, "num_key_value_heads",
                                           cfg.num_attn_heads); // Default to MHA
    cfg.head_dim = cfg.hidden_dim / cfg.num_attn_heads;
    cfg.intermediate_dim = (uint32_t)json_int(json, "intermediate_size", cfg.hidden_dim * 4);
    cfg.vocab_size = (uint32_t)json_int(json, "vocab_size", 32000);

    // RoPE
    cfg.rope_theta = (float)json_float(json, "rope_theta", 10000.0);
    cfg.max_position = (uint32_t)json_int(json, "max_position_embeddings", 131072);

    // MoE detection
    cfg.num_experts = (uint32_t)json_int(json, "num_local_experts",
                      (int64_t)json_int(json, "num_experts", 0));
    cfg.experts_per_tok = (uint32_t)json_int(json, "num_experts_per_tok",
                          (int64_t)json_int(json, "num_selected_experts", 0));
    cfg.num_shared_experts = (uint32_t)json_int(json, "num_shared_experts", 0);
    cfg.moe_intermediate_dim = (uint32_t)json_int(json, "moe_intermediate_size",
                                                   cfg.intermediate_dim);

    // Determine model type
    if (cfg.num_experts > 0 && cfg.experts_per_tok > 0) {
        cfg.model_type = ModelType::MOE;
    }

    // Activation
    std::string act = json_string(json, "hidden_act", "silu");
    if (act == "silu" || act == "swiglu") cfg.activation = ActivationType::SWIGLU;
    else if (act == "gelu") cfg.activation = ActivationType::GELU;
    else if (act == "relu") cfg.activation = ActivationType::RELU;

    LOG_INFO("Loaded model config: %s", cfg.name.c_str());
    LOG_INFO("  %u layers, hidden=%u, heads=%u/%u, vocab=%u",
             cfg.num_layers, cfg.hidden_dim, cfg.num_attn_heads, cfg.num_kv_heads,
             cfg.vocab_size);
    if (cfg.num_experts > 0) {
        LOG_INFO("  MoE: %u experts, %u per token, %u shared",
                 cfg.num_experts, cfg.experts_per_tok, cfg.num_shared_experts);
    }
    LOG_INFO("  Total params: %.1fB, Active per token: %.1fB",
             cfg.total_params() / 1e9, cfg.active_params_per_token() / 1e9);

    return cfg;
}

RuntimeConfig load_runtime_config(const std::string& config_path) {
    RuntimeConfig cfg;

    std::ifstream f(config_path);
    if (!f.good()) {
        LOG_WARN("Cannot open runtime config: %s — using defaults", config_path.c_str());
        return cfg;
    }

    std::stringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();

    // Parse runtime settings from JSON
    std::string model = json_string(json, "model_path", "");
    if (!model.empty()) cfg.model_path = model;

    std::string quant = json_string(json, "weight_dtype", "");
    if (quant == "fp16") cfg.weight_dtype = DType::FP16;
    else if (quant == "fp4") cfg.weight_dtype = DType::FP4;
    else if (quant == "int4" || quant == "q4") cfg.weight_dtype = DType::INT4;
    else if (quant == "q4_k") cfg.weight_dtype = DType::Q4_K;
    else if (quant == "int8" || quant == "q8") cfg.weight_dtype = DType::INT8;

    int64_t ctx = json_int(json, "max_context_len", 0);
    if (ctx > 0) cfg.max_context_len = (uint32_t)ctx;

    int64_t vram = json_int(json, "vram_budget_mb", 0);
    if (vram > 0) cfg.vram_budget_mb = (uint64_t)vram;

    int64_t ram = json_int(json, "ram_budget_mb", 0);
    if (ram > 0) cfg.ram_budget_mb = (uint64_t)ram;

    int64_t threads = json_int(json, "io_threads", 0);
    if (threads > 0) cfg.io_threads = (uint32_t)threads;

    std::string nvme = json_string(json, "nvme_cache_path", "");
    if (!nvme.empty()) cfg.nvme_cache_path = nvme;

    LOG_INFO("Loaded runtime config from: %s", config_path.c_str());
    return cfg;
}

RuntimeConfig parse_cli_args(int argc, char** argv) {
    RuntimeConfig cfg;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : "";
        };

        if (arg == "--model" || arg == "-m") {
            cfg.model_path = next();
        } else if (arg == "--quant" || arg == "-q") {
            std::string q = next();
            if (q == "fp16") cfg.weight_dtype = DType::FP16;
            else if (q == "fp8") cfg.weight_dtype = DType::FP8_E4M3;
            else if (q == "fp4") cfg.weight_dtype = DType::FP4;
            else if (q == "int8" || q == "q8") cfg.weight_dtype = DType::INT8;
            else if (q == "int4" || q == "q4") cfg.weight_dtype = DType::INT4;
            else if (q == "int2" || q == "q2") cfg.weight_dtype = DType::INT2;
            else if (q == "q4_k") cfg.weight_dtype = DType::Q4_K;
            else if (q == "q3_k") cfg.weight_dtype = DType::Q3_K;
        } else if (arg == "--context" || arg == "-c") {
            cfg.max_context_len = std::stoul(next());
        } else if (arg == "--vram") {
            cfg.vram_budget_mb = std::stoull(next());
        } else if (arg == "--ram") {
            cfg.ram_budget_mb = std::stoull(next());
        } else if (arg == "--threads") {
            cfg.io_threads = std::stoul(next());
        } else if (arg == "--no-prefetch") {
            cfg.enable_prefetch = false;
        } else if (arg == "--speculative") {
            cfg.num_speculative = std::stoul(next());
        } else if (arg == "--nvme-cache") {
            cfg.nvme_cache_path = next();
        }
    }

    return cfg;
}

std::string model_config_to_string(const ModelConfig& cfg) {
    std::ostringstream ss;
    ss << "Model: " << cfg.name << "\n";
    ss << "Type: " << (cfg.model_type == ModelType::DENSE ? "Dense" :
                       cfg.model_type == ModelType::MOE ? "MoE" : "Hybrid") << "\n";
    ss << "Layers: " << cfg.num_layers << "\n";
    ss << "Hidden: " << cfg.hidden_dim << "\n";
    ss << "Heads: " << cfg.num_attn_heads << "/" << cfg.num_kv_heads << "\n";
    ss << "Vocab: " << cfg.vocab_size << "\n";
    if (cfg.num_experts > 0) {
        ss << "Experts: " << cfg.num_experts << " (K=" << cfg.experts_per_tok << ")\n";
    }
    ss << "Total params: " << cfg.total_params() / 1e9 << "B\n";
    ss << "Active/token: " << cfg.active_params_per_token() / 1e9 << "B\n";
    return ss.str();
}

std::string runtime_config_to_string(const RuntimeConfig& cfg) {
    std::ostringstream ss;
    ss << "Weight dtype: " << dtype_name(cfg.weight_dtype) << "\n";
    ss << "Context: " << cfg.max_context_len << "\n";
    ss << "VRAM budget: " << (cfg.vram_budget_mb ? std::to_string(cfg.vram_budget_mb) + " MB" : "auto") << "\n";
    ss << "RAM budget: " << (cfg.ram_budget_mb ? std::to_string(cfg.ram_budget_mb) + " MB" : "auto") << "\n";
    ss << "I/O threads: " << cfg.io_threads << "\n";
    ss << "io_uring: " << (cfg.use_io_uring ? "yes" : "no") << "\n";
    ss << "Prefetch: " << (cfg.enable_prefetch ? "yes" : "no") << "\n";
    return ss.str();
}

} // namespace titan
