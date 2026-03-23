#pragma once

#include "core/types.h"
#include <string>

namespace titan {

// Parse a HuggingFace config.json into ModelConfig
ModelConfig load_model_config(const std::string& config_path);

// Parse runtime configuration from TOML or CLI args
RuntimeConfig load_runtime_config(const std::string& config_path);
RuntimeConfig parse_cli_args(int argc, char** argv);

// Serialize configs for logging/debugging
std::string model_config_to_string(const ModelConfig& config);
std::string runtime_config_to_string(const RuntimeConfig& config);

} // namespace titan
