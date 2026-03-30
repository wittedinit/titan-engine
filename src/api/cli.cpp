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
"  --chatml               Apply ChatML format (<|im_start|>/<|im_end|>) for instruct models\n"
"  --system PROMPT        System prompt to use with --chatml (default: none)\n"
"  --serve                Start as HTTP API server (OpenAI-compatible)\n"
"  --host HOST            Server bind address (default: 0.0.0.0)\n"
"  --port PORT            Server port (default: 8080)\n"
"  --hardware             Print hardware info and exit\n"
"  -v, --verbose          Verbose logging\n"
"  -h, --help             Show this help\n\n"
"Examples:\n"
"  %s -m ./llama-8b -q q4_k\n"
"  %s -m ./deepseek-v3 -q int4 --vram 28000\n"
"  %s -m ./kimi-k2.5-nvfp4 --chatml --system \"You are Kimi, an AI assistant made by Moonshot AI.\"\n\n"
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
    bool use_chatml = false;
    bool quant_explicit = false;
    std::string system_prompt;
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
        if (arg == "--quant" || arg == "-q") { quant_explicit = true; next(); } // already parsed by parse_cli_args
        else if (arg == "--temp") sampling.temperature = std::stof(next());
        else if (arg == "--top-p") sampling.top_p = std::stof(next());
        else if (arg == "--top-k") sampling.top_k = std::stoul(next());
        else if (arg == "--max-tokens") sampling.max_tokens = std::stoul(next());
        else if (arg == "--serve") serve_mode = true;
        else if (arg == "--host") serve_host = next();
        else if (arg == "--port") serve_port = std::stoi(next());
        else if (arg == "--chatml") use_chatml = true;
        else if (arg == "--system") system_prompt = next();
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
           quant_explicit ? dtype_name(config.weight_dtype) : "auto-detect",
           config.max_context_len, sampling.temperature);

    // Server mode: start HTTP API instead of interactive chat
    if (serve_mode) {
        printf("\nStarting API server on %s:%d...\n\n", serve_host.c_str(), serve_port);
        return run_server(engine, engine.model_config().name, serve_host, serve_port);
    }

    if (use_chatml) {
        printf("Chat mode (ChatML). Type 'exit' to stop, '/reset' to clear history, '/stats' for usage.\n");
    } else {
        printf("Type 'exit' or 'quit' to stop. Ctrl+C to abort.\n");
    }
    printf("\n");

    // Chat template history.
    // Kimi K2.5 uses: <|im_system|>...<|im_end|><|im_user|>...<|im_end|><|im_assistant|>
    // Standard ChatML uses: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n
    // Auto-detect: if the tokenizer has <|im_user|>, use Kimi format; else standard ChatML.
    bool kimi_format = (engine.tokenizer().token_to_id("<|im_user|>") >= 0);
    std::string chatml_history;
    if (use_chatml && !system_prompt.empty()) {
        if (kimi_format)
            chatml_history = "<|im_system|>" + system_prompt + "<|im_end|>";
        else
            chatml_history = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }

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
        if (input == "/reset") {
            chatml_history.clear();
            if (!system_prompt.empty()) {
                if (kimi_format)
                    chatml_history = "<|im_system|>" + system_prompt + "<|im_end|>";
                else
                    chatml_history = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
            }
            printf("(conversation reset)\n\n");
            continue;
        }
        if (input == "/help") {
            printf("Commands:\n");
            printf("  /stats   — Show memory and performance stats\n");
            printf("  /reset   — Clear conversation history\n");
            printf("  /help    — Show this help\n");
            printf("  exit     — Quit\n\n");
            continue;
        }

        // Build prompt
        std::string prompt;
        if (use_chatml) {
            if (kimi_format)
                chatml_history += "<|im_user|>" + input + "<|im_end|><|im_assistant|>";
            else
                chatml_history += "<|im_start|>user\n" + input + "<|im_end|>\n<|im_start|>assistant\n";
            prompt = chatml_history;
        } else {
            prompt = input;
        }

        // Generate response, capturing text for history
        printf("\033[1;34m");  // Blue text for model output
        fflush(stdout);

        std::string response_text;
        engine.generate(prompt, sampling,
            [&response_text](int /*token_id*/, const std::string& text) {
                printf("%s", text.c_str());
                fflush(stdout);
                response_text += text;
            });

        printf("\033[0m\n\n"); // Reset color

        // Append assistant response to history for next turn
        if (use_chatml) {
            chatml_history += response_text + "<|im_end|>\n";
        }
    }

    printf("\nGoodbye.\n");
    return 0;
}
