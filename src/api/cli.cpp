#include "core/types.h"
#include "core/config.h"
#include "core/hardware.h"
#include "core/logging.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>

namespace titan {
    // Forward declarations
    class InferenceEngine;
}

static void print_usage(const char* prog) {
    fprintf(stderr, R"(
Titan Engine — Universal LLM Inference Engine
Run models up to 1T parameters on consumer hardware.

Usage: %s [options]

Options:
  -m, --model PATH       Path to model directory (HuggingFace format)
  -q, --quant TYPE       Quantization: fp16, fp8, fp4, int8, int4, int2, q4_k, q3_k
  -c, --context N        Max context length (default: 8192)
  --vram N               VRAM budget in MB (default: auto)
  --ram N                RAM budget in MB (default: auto)
  --threads N            I/O threads (default: 4)
  --no-prefetch          Disable expert prefetching
  --speculative N        Enable speculative decoding with N draft tokens
  --nvme-cache PATH      NVMe cache directory
  --timing               Show per-layer timing breakdown
  --hardware             Print hardware info and exit
  -h, --help             Show this help

Examples:
  %s -m ./llama-8b -q q4_k
  %s -m ./deepseek-v3 -q int4 --vram 28000 --ram 100000
  %s -m ./qwen-1t-moe -q q3_k --nvme-cache /mnt/nvme_raid

)", prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Check for simple flags first
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--hardware") == 0) {
            auto hw = titan::detect_hardware();
            titan::print_hardware_summary(hw);
            return 0;
        }
    }

    // Parse CLI args
    titan::RuntimeConfig config = titan::parse_cli_args(argc, argv);

    if (config.model_path.empty()) {
        fprintf(stderr, "Error: --model/-m is required\n");
        print_usage(argv[0]);
        return 1;
    }

    titan::LOG_INFO("Titan Engine starting...");
    titan::LOG_INFO("Model: %s", config.model_path.c_str());
    titan::LOG_INFO("Quantization: %s", titan::dtype_name(config.weight_dtype));

    // Detect hardware
    auto hw = titan::detect_hardware();
    titan::print_hardware_summary(hw);

    // Load model config
    auto model_config = titan::load_model_config(config.model_path + "/config.json");

    // Show model info
    titan::LOG_INFO("Model: %s (%.1fB total, %.1fB active per token)",
                    model_config.name.c_str(),
                    model_config.total_params() / 1e9,
                    model_config.active_params_per_token() / 1e9);

    size_t est_bytes = model_config.estimated_weight_bytes(config.weight_dtype);
    titan::LOG_INFO("Estimated weight size at %s: %.1f GB",
                    titan::dtype_name(config.weight_dtype), est_bytes / 1e9);

    // Generate execution plan
    auto plan = titan::plan_execution(model_config, hw, config);
    titan::LOG_INFO("Execution plan: VRAM=%.1f GB, RAM=%.1f GB",
                    plan.vram_used / 1e9, plan.ram_used / 1e9);
    if (plan.expert_cache_ram_mb > 0) {
        titan::LOG_INFO("Expert cache: %.1f GB RAM, %.1f GB VRAM",
                        plan.expert_cache_ram_mb / 1024.0,
                        plan.expert_cache_vram_mb / 1024.0);
    }

    // Interactive chat loop
    titan::LOG_INFO("Starting interactive chat (type 'exit' to quit)");
    printf("\n");

    std::string input;
    while (true) {
        printf("> ");
        fflush(stdout);

        if (!std::getline(std::cin, input)) break;
        if (input.empty()) continue;
        if (input == "exit" || input == "quit") break;

        // TODO: Tokenize, run inference, decode output
        printf("[Titan] Inference engine not yet fully connected. Model loaded: %s\n",
               model_config.name.c_str());
        printf("  This is the project skeleton — full inference pipeline coming soon.\n\n");
    }

    titan::LOG_INFO("Titan Engine shutting down");
    return 0;
}
