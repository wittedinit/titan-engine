#pragma once

#include <string>
#include <functional>
#include <unordered_map>
#include <vector>
#include <thread>
#include <atomic>

namespace titan {

// ============================================================================
// Minimal HTTP Server — No external dependencies
//
// Supports:
// - GET and POST requests
// - JSON request/response
// - Server-Sent Events (SSE) for streaming
// - Keep-alive connections
// - Concurrent request handling via thread pool
// ============================================================================

struct HttpRequest {
    std::string method;     // GET, POST, etc.
    std::string path;       // /v1/chat/completions
    std::string body;       // Raw body (JSON for POST)
    std::unordered_map<std::string, std::string> headers;

    // Parse JSON body helpers (minimal, no external dependency)
    std::string json_string(const std::string& key) const;
    int64_t json_int(const std::string& key, int64_t def = 0) const;
    double json_float(const std::string& key, double def = 0) const;
    bool json_bool(const std::string& key, bool def = false) const;
};

struct HttpResponse {
    int status_code = 200;
    std::string content_type = "application/json";
    std::string body;
    std::unordered_map<std::string, std::string> headers;

    // Convenience constructors
    static HttpResponse json(int status, const std::string& body);
    static HttpResponse error(int status, const std::string& message);
    static HttpResponse sse_start(); // Begin SSE stream

    std::string serialize() const;
};

// Handler for SSE streaming — called to write chunks
class SseWriter {
public:
    explicit SseWriter(int client_fd) : fd_(client_fd) {}

    // Send an SSE event. data will be serialized as "data: {json}\n\n"
    bool send_event(const std::string& data);

    // Send the final [DONE] marker and close
    void finish();

    bool is_open() const { return fd_ >= 0; }

private:
    int fd_;
};

// Route handler types
using RouteHandler = std::function<HttpResponse(const HttpRequest&)>;
using StreamHandler = std::function<void(const HttpRequest&, SseWriter&)>;

class HttpServer {
public:
    HttpServer() = default;
    ~HttpServer();

    // Register route handlers
    void get(const std::string& path, RouteHandler handler);
    void post(const std::string& path, RouteHandler handler);

    // Register streaming handler (for SSE endpoints like chat completions)
    void post_stream(const std::string& path, StreamHandler handler);

    // Start serving (blocking)
    bool listen(const std::string& host, int port, int num_threads = 4);

    // Stop serving
    void stop();

    bool is_running() const { return running_; }

private:
    struct Route {
        std::string method;
        std::string path;
        RouteHandler handler;
        StreamHandler stream_handler;
        bool is_stream = false;
    };

    std::vector<Route> routes_;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::vector<std::thread> workers_;

    void handle_client(int client_fd);
    HttpRequest parse_request(int client_fd);
    void send_response(int client_fd, const HttpResponse& resp);
};

} // namespace titan
