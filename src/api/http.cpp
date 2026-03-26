#include "api/http.h"
#include "core/logging.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <csignal>
#include <cstring>
#include <sstream>
#include <algorithm>

namespace titan {

// ============================================================================
// HttpRequest JSON helpers (minimal parser, no external deps)
// ============================================================================

static std::string find_json_value(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n'))
        pos++;
    return json.substr(pos);
}

std::string HttpRequest::json_string(const std::string& key) const {
    auto val = find_json_value(body, key);
    if (val.empty() || val[0] != '"') return "";
    size_t end = val.find('"', 1);
    if (end == std::string::npos) return "";
    return val.substr(1, end - 1);
}

int64_t HttpRequest::json_int(const std::string& key, int64_t def) const {
    auto val = find_json_value(body, key);
    if (val.empty()) return def;
    try { return std::stoll(val); } catch (...) { return def; }
}

double HttpRequest::json_float(const std::string& key, double def) const {
    auto val = find_json_value(body, key);
    if (val.empty()) return def;
    try { return std::stod(val); } catch (...) { return def; }
}

bool HttpRequest::json_bool(const std::string& key, bool def) const {
    auto val = find_json_value(body, key);
    if (val.substr(0, 4) == "true") return true;
    if (val.substr(0, 5) == "false") return false;
    return def;
}

// ============================================================================
// HttpResponse
// ============================================================================

HttpResponse HttpResponse::json(int status, const std::string& body) {
    HttpResponse r;
    r.status_code = status;
    r.content_type = "application/json";
    r.body = body;
    return r;
}

HttpResponse HttpResponse::error(int status, const std::string& message) {
    std::string body = "{\"error\":{\"message\":\"" + message + "\",\"type\":\"server_error\",\"code\":" +
                       std::to_string(status) + "}}";
    return json(status, body);
}

HttpResponse HttpResponse::sse_start() {
    HttpResponse r;
    r.status_code = 200;
    r.content_type = "text/event-stream";
    r.headers["Cache-Control"] = "no-cache";
    r.headers["Connection"] = "keep-alive";
    r.body = ""; // Body sent via SSE events
    return r;
}

std::string HttpResponse::serialize() const {
    std::string status_text;
    switch (status_code) {
        case 200: status_text = "OK"; break;
        case 400: status_text = "Bad Request"; break;
        case 404: status_text = "Not Found"; break;
        case 500: status_text = "Internal Server Error"; break;
        default: status_text = "Unknown"; break;
    }

    std::ostringstream ss;
    ss << "HTTP/1.1 " << status_code << " " << status_text << "\r\n";
    ss << "Content-Type: " << content_type << "\r\n";
    ss << "Access-Control-Allow-Origin: *\r\n";
    ss << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    ss << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";

    for (const auto& [k, v] : headers) {
        ss << k << ": " << v << "\r\n";
    }

    if (!body.empty()) {
        ss << "Content-Length: " << body.size() << "\r\n";
    }
    ss << "\r\n";
    ss << body;

    return ss.str();
}

// ============================================================================
// SseWriter
// ============================================================================

bool SseWriter::send_event(const std::string& data) {
    if (fd_ < 0) return false;
    std::string event = "data: " + data + "\n\n";
    size_t sent = 0;
    while (sent < event.size()) {
        ssize_t n = write(fd_, event.c_str() + sent, event.size() - sent);
        if (n <= 0) {
            fd_ = -1;
            return false;
        }
        sent += n;
    }
    return true;
}

void SseWriter::finish() {
    if (fd_ < 0) return;
    std::string done = "data: [DONE]\n\n";
    write(fd_, done.c_str(), done.size());
    fd_ = -1;
}

// ============================================================================
// HttpServer
// ============================================================================

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::get(const std::string& path, RouteHandler handler) {
    routes_.push_back({"GET", path, handler, nullptr, false});
}

void HttpServer::post(const std::string& path, RouteHandler handler) {
    routes_.push_back({"POST", path, handler, nullptr, false});
}

void HttpServer::post_stream(const std::string& path, StreamHandler handler) {
    routes_.push_back({"POST", path, nullptr, handler, true});
}

void HttpServer::post_with_stream(const std::string& path, RouteHandler handler, StreamHandler stream_handler) {
    routes_.push_back({"POST", path, handler, stream_handler, true});
}

HttpRequest HttpServer::parse_request(int client_fd) {
    HttpRequest req;

    // Read headers (up to 64KB)
    char buf[65536];
    ssize_t total = 0;
    ssize_t header_end = -1;

    while (total < (ssize_t)sizeof(buf) - 1) {
        ssize_t n = read(client_fd, buf + total, sizeof(buf) - 1 - total);
        if (n <= 0) break;
        total += n;
        buf[total] = '\0';

        // Check for end of headers
        char* hend = strstr(buf, "\r\n\r\n");
        if (hend) {
            header_end = hend - buf + 4;
            break;
        }
    }

    if (header_end < 0) return req;

    // Parse request line
    std::string header_str(buf, header_end);
    auto first_line_end = header_str.find("\r\n");
    std::string first_line = header_str.substr(0, first_line_end);

    auto sp1 = first_line.find(' ');
    auto sp2 = first_line.find(' ', sp1 + 1);
    if (sp1 != std::string::npos && sp2 != std::string::npos) {
        req.method = first_line.substr(0, sp1);
        req.path = first_line.substr(sp1 + 1, sp2 - sp1 - 1);
    }

    // Parse headers
    size_t pos = first_line_end + 2;
    while (pos < header_str.size()) {
        auto line_end = header_str.find("\r\n", pos);
        if (line_end == std::string::npos || line_end == pos) break;
        std::string line = header_str.substr(pos, line_end - pos);
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            while (!val.empty() && val[0] == ' ') val.erase(0, 1);
            // Lowercase key for case-insensitive lookup
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
            req.headers[key] = val;
        }
        pos = line_end + 2;
    }

    // Read body if Content-Length present
    auto cl_it = req.headers.find("content-length");
    if (cl_it != req.headers.end()) {
        size_t content_length = std::stoull(cl_it->second);
        size_t body_already = total - header_end;

        req.body = std::string(buf + header_end, body_already);

        // Read remaining body
        while (req.body.size() < content_length) {
            ssize_t n = read(client_fd, buf, std::min(sizeof(buf),
                             content_length - req.body.size()));
            if (n <= 0) break;
            req.body.append(buf, n);
        }
    }

    return req;
}

void HttpServer::send_response(int client_fd, const HttpResponse& resp) {
    std::string data = resp.serialize();
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = write(client_fd, data.c_str() + sent, data.size() - sent);
        if (n <= 0) break;
        sent += n;
    }
}

void HttpServer::handle_client(int client_fd) {
    HttpRequest req = parse_request(client_fd);

    if (req.method.empty()) {
        close(client_fd);
        return;
    }

    LOG_DEBUG("HTTP %s %s", req.method.c_str(), req.path.c_str());

    // Handle CORS preflight
    if (req.method == "OPTIONS") {
        HttpResponse resp;
        resp.status_code = 204;
        resp.headers["Access-Control-Allow-Origin"] = "*";
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // Find matching route
    for (const auto& route : routes_) {
        if (route.method == req.method && route.path == req.path) {
            if (route.is_stream) {
                // Check if client wants streaming
                bool wants_stream = req.json_bool("stream", false);

                if (wants_stream && route.stream_handler) {
                    // Send SSE headers first
                    HttpResponse sse = HttpResponse::sse_start();
                    std::string headers = "HTTP/1.1 200 OK\r\n"
                                          "Content-Type: text/event-stream\r\n"
                                          "Cache-Control: no-cache\r\n"
                                          "Connection: keep-alive\r\n"
                                          "Access-Control-Allow-Origin: *\r\n"
                                          "\r\n";
                    write(client_fd, headers.c_str(), headers.size());

                    SseWriter writer(client_fd);
                    route.stream_handler(req, writer);
                    writer.finish();
                } else if (route.handler) {
                    // Non-streaming fallback
                    HttpResponse resp = route.handler(req);
                    send_response(client_fd, resp);
                } else {
                    // Stream handler only but client didn't request streaming
                    // Call stream handler and collect all output
                    HttpResponse sse = HttpResponse::sse_start();
                    std::string headers_str = "HTTP/1.1 200 OK\r\n"
                                              "Content-Type: text/event-stream\r\n"
                                              "Cache-Control: no-cache\r\n"
                                              "Connection: keep-alive\r\n"
                                              "Access-Control-Allow-Origin: *\r\n"
                                              "\r\n";
                    write(client_fd, headers_str.c_str(), headers_str.size());
                    SseWriter writer(client_fd);
                    route.stream_handler(req, writer);
                    writer.finish();
                }
            } else {
                HttpResponse resp = route.handler(req);
                send_response(client_fd, resp);
            }
            close(client_fd);
            return;
        }
    }

    // 404
    HttpResponse resp = HttpResponse::error(404, "Not found: " + req.path);
    send_response(client_fd, resp);
    close(client_fd);
}

bool HttpServer::listen(const std::string& host, int port, int num_threads) {
    // Ignore SIGPIPE so writing to a closed socket returns EPIPE instead of
    // killing the server process.
    signal(SIGPIPE, SIG_IGN);

    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        LOG_ERROR("Failed to create socket");
        return false;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (host == "0.0.0.0" || host.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    }

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERROR("Failed to bind to %s:%d: %s", host.c_str(), port, strerror(errno));
        close(server_fd_);
        return false;
    }

    if (::listen(server_fd_, 128) < 0) {
        LOG_ERROR("Failed to listen: %s", strerror(errno));
        close(server_fd_);
        return false;
    }

    running_ = true;
    LOG_INFO("HTTP server listening on %s:%d", host.c_str(), port);

    // Accept loop
    while (running_) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (running_) LOG_ERROR("Accept failed: %s", strerror(errno));
            continue;
        }

        // Handle each client in a detached thread
        std::thread([this, client_fd]() {
            handle_client(client_fd);
        }).detach();
    }

    return true;
}

void HttpServer::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }
}

} // namespace titan
