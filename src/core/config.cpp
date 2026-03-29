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

// Extract the JSON object value for a given key as a string
// Returns the raw substring from '{' to matching '}' for a nested object
static std::string json_nested_object(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n'))
        pos++;
    if (pos >= json.size() || json[pos] != '{') return "";
    int depth = 1;
    size_t start = pos;
    pos++;
    while (pos < json.size() && depth > 0) {
        if (json[pos] == '{') depth++;
        else if (json[pos] == '}') depth--;
        pos++;
    }
    return json.substr(start, pos - start);
}

static void parse_model_fields(const std::string& json, ModelConfig& cfg) {
    // Model type name — try both model_type and _name_or_path
    std::string mtype = json_string(json, "model_type", "");
    if (!mtype.empty()) cfg.name = mtype;

    // Core dimensions
    int64_t hs = json_int(json, "hidden_size", 0);
    if (hs > 0) cfg.hidden_dim = (uint32_t)hs;

    int64_t nl = json_int(json, "num_hidden_layers", 0);
    if (nl > 0) cfg.num_layers = (uint32_t)nl;

    int64_t nh = json_int(json, "num_attention_heads", 0);
    if (nh > 0) cfg.num_attn_heads = (uint32_t)nh;

    int64_t nkv = json_int(json, "num_key_value_heads", 0);
    if (nkv > 0) cfg.num_kv_heads = (uint32_t)nkv;

    int64_t inter = json_int(json, "intermediate_size", 0);
    if (inter > 0) cfg.intermediate_dim = (uint32_t)inter;

    int64_t vocab = json_int(json, "vocab_size", 0);
    if (vocab > 0) cfg.vocab_size = (uint32_t)vocab;

    // RoPE
    double rt = json_float(json, "rope_theta", 0.0);
    if (rt > 0.0) cfg.rope_theta = (float)rt;
    int64_t maxpos = json_int(json, "max_position_embeddings", 0);
    if (maxpos > 0) cfg.max_position = (uint32_t)maxpos;

    // MoE — try all naming conventions
    int64_t ne = json_int(json, "num_local_experts",
                 json_int(json, "num_experts",
                 json_int(json, "n_routed_experts", 0)));
    if (ne > 0) cfg.num_experts = (uint32_t)ne;

    int64_t ek = json_int(json, "num_experts_per_tok",
                 json_int(json, "num_selected_experts", 0));
    if (ek > 0) cfg.experts_per_tok = (uint32_t)ek;

    int64_t nse = json_int(json, "num_shared_experts",
                  json_int(json, "n_shared_experts", 0));
    if (nse > 0) cfg.num_shared_experts = (uint32_t)nse;

    int64_t moe_inter = json_int(json, "moe_intermediate_size", 0);
    if (moe_inter > 0) cfg.moe_intermediate_dim = (uint32_t)moe_inter;

    int64_t fkd = json_int(json, "first_k_dense_replace", 0);
    if (fkd > 0) cfg.first_k_dense_replace = (uint32_t)fkd;

    int64_t mlf = json_int(json, "moe_layer_freq", 0);
    if (mlf > 0) cfg.moe_layer_freq = (uint32_t)mlf;

    // MLA (Multi-head Latent Attention) — DeepSeek V3 / Kimi K2 style
    int64_t kv_lr = json_int(json, "kv_lora_rank", 0);
    if (kv_lr > 0) { cfg.kv_lora_rank = (uint32_t)kv_lr; cfg.has_mla = true; }

    int64_t q_lr = json_int(json, "q_lora_rank", 0);
    if (q_lr > 0) cfg.q_lora_rank = (uint32_t)q_lr;

    // In MLA: qk_rope_head_dim is the decoupled RoPE portion of each head
    int64_t rope_hd = json_int(json, "qk_rope_head_dim", 0);
    if (rope_hd > 0) cfg.rope_head_dim = (uint32_t)rope_hd;

    int64_t nope_hd = json_int(json, "qk_nope_head_dim", 0);
    if (nope_hd > 0) cfg.nope_head_dim = (uint32_t)nope_hd;

    int64_t v_hd = json_int(json, "v_head_dim", 0);
    if (v_hd > 0) cfg.v_head_dim = (uint32_t)v_hd;

    // Activation
    std::string act = json_string(json, "hidden_act", "");
    if (!act.empty()) {
        if (act == "silu" || act == "swiglu") cfg.activation = ActivationType::SWIGLU;
        else if (act == "gelu") cfg.activation = ActivationType::GELU;
        else if (act == "relu") cfg.activation = ActivationType::RELU;
    }
}

ModelConfig load_model_config(const std::string& config_path) {
    ModelConfig cfg;
    // Defaults before any parsing
    cfg.hidden_dim = 4096;
    cfg.num_layers = 32;
    cfg.num_attn_heads = 32;
    cfg.vocab_size = 32000;
    cfg.intermediate_dim = 0;
    cfg.rope_theta = 10000.0f;
    cfg.max_position = 131072;

    std::ifstream f(config_path);
    if (!f.good()) {
        LOG_ERROR("Cannot open model config: %s", config_path.c_str());
        return cfg;
    }

    std::stringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();

    // Top-level pass — picks up top-level model_type and vocab_size
    parse_model_fields(json, cfg);

    // If there's a nested text_config (Kimi K2.5, multimodal wrappers, etc.),
    // parse it and let it override — text_config holds the LLM parameters
    std::string text_cfg = json_nested_object(json, "text_config");
    if (!text_cfg.empty()) {
        LOG_INFO("Found nested text_config — parsing LLM parameters from it");
        parse_model_fields(text_cfg, cfg);
    }

    // Derived values
    if (cfg.num_kv_heads == 0) cfg.num_kv_heads = cfg.num_attn_heads;
    if (cfg.intermediate_dim == 0) cfg.intermediate_dim = cfg.hidden_dim * 4;
    if (cfg.moe_intermediate_dim == 0) cfg.moe_intermediate_dim = cfg.intermediate_dim;
    if (cfg.moe_layer_freq == 0) cfg.moe_layer_freq = 1;

    // head_dim: in MLA, use nope_head_dim + rope_head_dim if set, else hidden/heads
    if (cfg.has_mla && cfg.nope_head_dim > 0 && cfg.rope_head_dim > 0) {
        cfg.head_dim = cfg.nope_head_dim + cfg.rope_head_dim;
    } else {
        cfg.head_dim = cfg.hidden_dim / cfg.num_attn_heads;
    }

    // v_head_dim fallback
    if (cfg.v_head_dim == 0) cfg.v_head_dim = cfg.head_dim;

    // Determine model type
    if (cfg.num_experts > 0 && cfg.experts_per_tok > 0) {
        cfg.model_type = ModelType::MOE;
    }

    LOG_INFO("Loaded model config: %s", cfg.name.c_str());
    LOG_INFO("  %u layers, hidden=%u, heads=%u/%u (kv), vocab=%u",
             cfg.num_layers, cfg.hidden_dim, cfg.num_attn_heads, cfg.num_kv_heads,
             cfg.vocab_size);
    if (cfg.has_mla) {
        LOG_INFO("  MLA attention: kv_lora=%u, q_lora=%u, nope=%u, rope=%u",
                 cfg.kv_lora_rank, cfg.q_lora_rank, cfg.nope_head_dim, cfg.rope_head_dim);
    }
    if (cfg.num_experts > 0) {
        LOG_INFO("  MoE: %u experts, %u per token, %u shared, first_k_dense=%u",
                 cfg.num_experts, cfg.experts_per_tok, cfg.num_shared_experts,
                 cfg.first_k_dense_replace);
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
