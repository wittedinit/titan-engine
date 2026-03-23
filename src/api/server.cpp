#include "api/http.h"
#include "inference/engine.h"
#include "core/logging.h"
#include "core/config.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <sstream>
#include <chrono>
#include <mutex>
#include <atomic>

using namespace titan;

// ============================================================================
// OpenAI-Compatible API Server
//
// Endpoints:
//   GET  /v1/models              — List available models
//   POST /v1/chat/completions    — Generate chat completion (streaming + non-streaming)
//   POST /v1/completions         — Generate text completion
//   GET  /health                 — Health check
//
// Compatible with:
//   - OpenAI Python SDK: openai.ChatCompletion.create(...)
//   - curl
//   - Any OpenAI-compatible client (Open WebUI, LM Studio, etc.)
// ============================================================================

static InferenceEngine* g_engine = nullptr;
static std::string g_model_name;
static std::mutex g_inference_mutex; // Serialize inference (single-GPU)
static std::atomic<int> g_request_id{0};

// ============================================================================
// JSON Helpers (minimal, no external deps)
// ============================================================================

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Extract the last "content" string from a messages array in the request body
static std::string extract_prompt(const std::string& body) {
    // Find the last "content" field in the messages array
    // This is simplified — a full implementation would parse the messages array properly
    std::string result;
    size_t pos = 0;

    while (true) {
        auto content_pos = body.find("\"content\"", pos);
        if (content_pos == std::string::npos) break;

        auto colon = body.find(':', content_pos + 9);
        if (colon == std::string::npos) break;

        // Skip whitespace
        size_t p = colon + 1;
        while (p < body.size() && (body[p] == ' ' || body[p] == '\t' || body[p] == '\n'))
            p++;

        if (p < body.size() && body[p] == '"') {
            p++; // skip opening quote
            std::string content;
            while (p < body.size() && body[p] != '"') {
                if (body[p] == '\\' && p + 1 < body.size()) {
                    p++;
                    if (body[p] == 'n') content += '\n';
                    else if (body[p] == 't') content += '\t';
                    else content += body[p];
                } else {
                    content += body[p];
                }
                p++;
            }
            result = content; // Keep last content found
        }

        pos = content_pos + 9;
    }

    return result;
}

// ============================================================================
// Endpoint Handlers
// ============================================================================

static HttpResponse handle_models(const HttpRequest& req) {
    auto now = std::chrono::system_clock::now().time_since_epoch();
    int64_t created = std::chrono::duration_cast<std::chrono::seconds>(now).count();

    std::ostringstream ss;
    ss << "{\"object\":\"list\",\"data\":[{";
    ss << "\"id\":\"" << json_escape(g_model_name) << "\",";
    ss << "\"object\":\"model\",";
    ss << "\"created\":" << created << ",";
    ss << "\"owned_by\":\"titan-engine\"";
    ss << "}]}";

    return HttpResponse::json(200, ss.str());
}

static HttpResponse handle_health(const HttpRequest& req) {
    return HttpResponse::json(200, "{\"status\":\"ok\",\"engine\":\"titan\"}");
}

// Non-streaming chat completion
static HttpResponse handle_chat_non_stream(const HttpRequest& req) {
    std::string prompt = extract_prompt(req.body);
    if (prompt.empty()) {
        return HttpResponse::error(400, "No content in messages");
    }

    std::string model = req.json_string("model");
    float temperature = (float)req.json_float("temperature", 0.7);
    int max_tokens = (int)req.json_int("max_tokens", 2048);
    float top_p = (float)req.json_float("top_p", 0.9);

    SamplingParams sampling;
    sampling.temperature = temperature;
    sampling.max_tokens = max_tokens;
    sampling.top_p = top_p;

    std::string response_text;
    int completion_tokens = 0;

    {
        std::lock_guard<std::mutex> lock(g_inference_mutex);
        g_engine->generate(prompt, sampling,
            [&](int token_id, const std::string& text) {
                response_text += text;
                completion_tokens++;
            });
    }

    int request_id = g_request_id++;
    auto now = std::chrono::system_clock::now().time_since_epoch();
    int64_t created = std::chrono::duration_cast<std::chrono::seconds>(now).count();

    int prompt_tokens = (int)g_engine->tokenizer().encode(prompt).size();

    std::ostringstream ss;
    ss << "{";
    ss << "\"id\":\"chatcmpl-titan-" << request_id << "\",";
    ss << "\"object\":\"chat.completion\",";
    ss << "\"created\":" << created << ",";
    ss << "\"model\":\"" << json_escape(g_model_name) << "\",";
    ss << "\"choices\":[{";
    ss << "\"index\":0,";
    ss << "\"message\":{\"role\":\"assistant\",\"content\":\"" << json_escape(response_text) << "\"},";
    ss << "\"finish_reason\":\"stop\"";
    ss << "}],";
    ss << "\"usage\":{";
    ss << "\"prompt_tokens\":" << prompt_tokens << ",";
    ss << "\"completion_tokens\":" << completion_tokens << ",";
    ss << "\"total_tokens\":" << (prompt_tokens + completion_tokens);
    ss << "}}";

    return HttpResponse::json(200, ss.str());
}

// Streaming chat completion (SSE)
static void handle_chat_stream(const HttpRequest& req, SseWriter& writer) {
    std::string prompt = extract_prompt(req.body);
    if (prompt.empty()) {
        writer.send_event("{\"error\":\"No content in messages\"}");
        return;
    }

    float temperature = (float)req.json_float("temperature", 0.7);
    int max_tokens = (int)req.json_int("max_tokens", 2048);
    float top_p = (float)req.json_float("top_p", 0.9);

    SamplingParams sampling;
    sampling.temperature = temperature;
    sampling.max_tokens = max_tokens;
    sampling.top_p = top_p;

    int request_id = g_request_id++;
    auto now = std::chrono::system_clock::now().time_since_epoch();
    int64_t created = std::chrono::duration_cast<std::chrono::seconds>(now).count();

    // Send role chunk first
    {
        std::ostringstream ss;
        ss << "{\"id\":\"chatcmpl-titan-" << request_id << "\",";
        ss << "\"object\":\"chat.completion.chunk\",";
        ss << "\"created\":" << created << ",";
        ss << "\"model\":\"" << json_escape(g_model_name) << "\",";
        ss << "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}";
        writer.send_event(ss.str());
    }

    // Generate tokens and stream each one
    {
        std::lock_guard<std::mutex> lock(g_inference_mutex);
        g_engine->generate(prompt, sampling,
            [&](int token_id, const std::string& text) {
                if (!writer.is_open()) return;

                std::ostringstream ss;
                ss << "{\"id\":\"chatcmpl-titan-" << request_id << "\",";
                ss << "\"object\":\"chat.completion.chunk\",";
                ss << "\"created\":" << created << ",";
                ss << "\"model\":\"" << json_escape(g_model_name) << "\",";
                ss << "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" << json_escape(text) << "\"},\"finish_reason\":null}]}";
                writer.send_event(ss.str());
            });
    }

    // Send finish chunk
    {
        std::ostringstream ss;
        ss << "{\"id\":\"chatcmpl-titan-" << request_id << "\",";
        ss << "\"object\":\"chat.completion.chunk\",";
        ss << "\"created\":" << created << ",";
        ss << "\"model\":\"" << json_escape(g_model_name) << "\",";
        ss << "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}";
        writer.send_event(ss.str());
    }
}

// ============================================================================
// Server Main (called from CLI with --serve flag)
// ============================================================================

namespace titan {

int run_server(InferenceEngine& engine, const std::string& model_name,
               const std::string& host, int port) {
    g_engine = &engine;
    g_model_name = model_name;

    HttpServer server;

    // Register endpoints
    server.get("/v1/models", handle_models);
    server.get("/health", handle_health);

    // Chat completions: supports both streaming and non-streaming
    server.post("/v1/chat/completions", handle_chat_non_stream);
    server.post_stream("/v1/chat/completions", handle_chat_stream);

    // Text completions (simplified — maps to same handler)
    server.post("/v1/completions", [](const HttpRequest& req) -> HttpResponse {
        std::string prompt = req.json_string("prompt");
        if (prompt.empty()) {
            return HttpResponse::error(400, "No prompt provided");
        }

        float temperature = (float)req.json_float("temperature", 0.7);
        int max_tokens = (int)req.json_int("max_tokens", 2048);

        SamplingParams sampling;
        sampling.temperature = temperature;
        sampling.max_tokens = max_tokens;

        std::string text;
        int tokens = 0;
        {
            std::lock_guard<std::mutex> lock(g_inference_mutex);
            g_engine->generate(prompt, sampling,
                [&](int token_id, const std::string& t) {
                    text += t;
                    tokens++;
                });
        }

        int request_id = g_request_id++;

        std::ostringstream ss;
        ss << "{\"id\":\"cmpl-titan-" << request_id << "\",";
        ss << "\"object\":\"text_completion\",";
        ss << "\"model\":\"" << json_escape(g_model_name) << "\",";
        ss << "\"choices\":[{\"text\":\"" << json_escape(text) << "\",";
        ss << "\"index\":0,\"finish_reason\":\"stop\"}],";
        ss << "\"usage\":{\"completion_tokens\":" << tokens << "}}";

        return HttpResponse::json(200, ss.str());
    });

    LOG_INFO("Starting OpenAI-compatible API server on %s:%d", host.c_str(), port);
    LOG_INFO("Endpoints:");
    LOG_INFO("  GET  /health                 — Health check");
    LOG_INFO("  GET  /v1/models              — List models");
    LOG_INFO("  POST /v1/chat/completions    — Chat (streaming + non-streaming)");
    LOG_INFO("  POST /v1/completions         — Text completion");

    // This blocks
    return server.listen(host, port) ? 0 : 1;
}

} // namespace titan
