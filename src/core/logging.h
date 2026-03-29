#pragma once

#include <cstdio>
#include <cstdarg>
#include <chrono>

namespace titan {

enum class LogLevel : int {
    DEBUG   = 0,
    INFO    = 1,
    WARN    = 2,
    ERROR   = 3,
    NONE    = 4,
};

inline LogLevel g_log_level = LogLevel::INFO;

inline void set_log_level(LogLevel level) { g_log_level = level; }

inline void log_msg(LogLevel level, const char* file, int line, const char* fmt, ...) {
    if (level < g_log_level) return;

    static auto start = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start).count();

    const char* prefix = "";
    FILE* out = stderr;  // All log output goes to stderr so stdout is clean token text
    switch (level) {
        case LogLevel::DEBUG: prefix = "DEBUG"; break;
        case LogLevel::INFO:  prefix = "INFO "; break;
        case LogLevel::WARN:  prefix = "WARN "; break;
        case LogLevel::ERROR: prefix = "ERROR"; break;
        default: break;
    }

    fprintf(out, "[%8.3f] %s  ", elapsed, prefix);

    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);

    fprintf(out, "\n");
    fflush(out);
}

#define LOG_DEBUG(fmt, ...) ::titan::log_msg(::titan::LogLevel::DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  ::titan::log_msg(::titan::LogLevel::INFO,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  ::titan::log_msg(::titan::LogLevel::WARN,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) ::titan::log_msg(::titan::LogLevel::ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

} // namespace titan
