#include "core/types.h"
#include "core/config.h"
#include "core/hardware.h"
#include "core/logging.h"
#include "inference/engine.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>

using namespace titan;

// Forward declaration from server.cpp
namespace titan {
    int run_server(InferenceEngine& engine, const std::string& model_name,
                   const std::string& host, int port);
}

static void print_banner() {
    printf("\n");
    printf("  ████████╗██╗████████╗ █████╗ ███╗   ██╗\n");
    printf("  ╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║\n");
    printf("     ██║   ██║   ██║   ███████║██╔██╗ ██║\n");
    printf("     ██║   ██║   ██║   ██╔══██║██║╚██╗██║\n");
    printf("     ██║   ██║   ██║   ██║  ██║██║ ╚████║\n");
    printf("     ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝\n");
    printf("  Universal LLM Inference Engine\n");
    printf("  Run models up to 1T parameters on consumer hardware.\n\n");
}

static void print_usage(const char* prog) {
    fprintf(stderr,
"Usage: %s [options]\n\n"
"Options:\n"
"  -m, --model PATH       Model directory (HuggingFace format)\n"
"  -q, --quant TYPE       Quantization: fp16, fp8, int4, q4_k, q3_k, int2\n"
"  -c, --context N        Max context length (default: 8192)\n"
"  --vram N               VRAM budget in MB (default: auto)\n"
"  --ram N                RAM budget in MB (default: auto)\n"
"  --threads N            I/O threads (default: 4)\n"
"  --temp T               Temperature (default: 0.7)\n"
"  --top-p P              Top-p / nucleus sampling (default: 0.9)\n"
"  --top-k K              Top-k sampling (default: 40)\n"
"  --max-tokens N         Max tokens to generate (default: 2048)\n"
"  --no-prefetch          Disable expert prefetching\n"
"  --serve                Start as HTTP API server (OpenAI-compatible)\n"
"  --host HOST            Server bind address (default: 0.0.0.0)\n"
"  --port PORT            Server port (default: 8080)\n"
"  --hardware             Print hardware info and exit\n"
"  -v, --verbose          Verbose logging\n"
"  -h, --help             Show this help\n\n"
"Examples:\n"
"  %s -m ./llama-8b -q q4_k\n"
"  %s -m ./deepseek-v3 -q int4 --vram 28000\n"
"  %s -m ./kimi-k2.5 -q q3_k --max-tokens 4096\n\n"
"Server mode:\n"
"  %s -m ./llama-8b -q q4_k --serve --port 8080\n\n"
"  Then use with any OpenAI client:\n"
"    curl http://localhost:8080/v1/chat/completions \\\n"
"      -H 'Content-Type: application/json' \\\n"
"      -d '{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'\n\n",
    prog, prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_banner();
        print_usage(argv[0]);
        return 1;
    }

    // Parse sampling params alongside runtime config
    SamplingParams sampling;
    bool verbose = false;
    bool serve_mode = false;
    std::string serve_host = "0.0.0.0";
    int serve_port = 8080;

    // Check simple flags
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--hardware") == 0) {
            auto hw = detect_hardware();
            print_hardware_summary(hw);
            return 0;
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        }
    }

    if (verbose) {
        set_log_level(LogLevel::DEBUG);
    }

    // Parse config
    RuntimeConfig config = parse_cli_args(argc, argv);

    // Parse extra sampling flags
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if (arg == "--temp") sampling.temperature = std::stof(next());
        else if (arg == "--top-p") sampling.top_p = std::stof(next());
        else if (arg == "--top-k") sampling.top_k = std::stoul(next());
        else if (arg == "--max-tokens") sampling.max_tokens = std::stoul(next());
        else if (arg == "--serve") serve_mode = true;
        else if (arg == "--host") serve_host = next();
        else if (arg == "--port") serve_port = std::stoi(next());
    }

    if (config.model_path.empty()) {
        fprintf(stderr, "Error: --model/-m is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    print_banner();

    // Initialize engine
    InferenceEngine engine;
    if (!engine.initialize(config)) {
        LOG_ERROR("Failed to initialize engine");
        return 1;
    }

    // Load model
    LOG_INFO("Loading model from %s...", config.model_path.c_str());
    if (!engine.load_model(config.model_path)) {
        LOG_ERROR("Failed to load model");
        return 1;
    }

    printf("\n");
    printf("Model: %s (%.1fB params)\n",
           engine.model_config().name.c_str(),
           engine.model_config().total_params() / 1e9);
    printf("Quant: %s | Context: %u | Temp: %.1f\n",
           dtype_name(config.weight_dtype), config.max_context_len,
           sampling.temperature);

    // Server mode: start HTTP API instead of interactive chat
    if (serve_mode) {
        printf("\nStarting API server on %s:%d...\n\n", serve_host.c_str(), serve_port);
        return run_server(engine, engine.model_config().name, serve_host, serve_port);
    }

    printf("Type 'exit' or 'quit' to stop. Ctrl+C to abort.\n");
    printf("\n");

    // Interactive chat loop
    std::string input;
    while (true) {
        printf("\033[1;32m> \033[0m");
        fflush(stdout);

        if (!std::getline(std::cin, input)) break;
        if (input.empty()) continue;
        if (input == "exit" || input == "quit") break;
        if (input == "/stats") {
            engine.print_stats();
            continue;
        }
        if (input == "/help") {
            printf("Commands:\n");
            printf("  /stats   — Show memory and performance stats\n");
            printf("  /help    — Show this help\n");
            printf("  exit     — Quit\n\n");
            continue;
        }

        // Generate response
        printf("\033[1;34m");  // Blue text for model output
        fflush(stdout);

        engine.generate(input, sampling,
            [](int token_id, const std::string& text) {
                printf("%s", text.c_str());
                fflush(stdout);
            });

        printf("\033[0m\n\n"); // Reset color
    }

    printf("\nGoodbye.\n");
    return 0;
}
