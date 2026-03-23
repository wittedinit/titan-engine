#include "memory/memory_manager.h"
#include "core/logging.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include <thread>
#include <queue>
#include <condition_variable>
#include <chrono>

#ifdef TITAN_HAS_IO_URING
#include <liburing.h>
#endif

namespace titan {

// ============================================================================
// I/O Context — io_uring or thread pool based async I/O
// ============================================================================

struct NvmePool::IoContext {
    bool use_io_uring = false;

#ifdef TITAN_HAS_IO_URING
    struct io_uring ring;
    bool ring_initialized = false;
#endif

    // Thread pool fallback for systems without io_uring
    struct IoTask {
        std::string path;
        void*       dst;
        size_t      bytes;
        off_t       offset;
        NvmePool::ReadCallback callback;
    };

    std::vector<std::thread> workers;
    std::queue<IoTask> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool shutdown = false;

    void worker_thread() {
        while (true) {
            IoTask task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] { return !task_queue.empty() || shutdown; });
                if (shutdown && task_queue.empty()) return;
                task = std::move(task_queue.front());
                task_queue.pop();
            }

            // Perform synchronous pread
            int flags = O_RDONLY;
#ifdef O_DIRECT
            flags |= O_DIRECT;
#endif
            int fd = open(task.path.c_str(), flags);
            if (fd < 0) {
                // Retry without O_DIRECT
                fd = open(task.path.c_str(), O_RDONLY);
            }

            ssize_t result = -1;
            if (fd >= 0) {
                result = pread(fd, task.dst, task.bytes, task.offset);
                close(fd);
            }

            if (task.callback) {
                task.callback(result);
            }
        }
    }
};

// ============================================================================
// NVMe Pool Implementation
// ============================================================================

NvmePool::NvmePool(const std::string& base_path, size_t cache_bytes)
    : base_path_(base_path), capacity_(cache_bytes) {
    io_ctx_ = std::make_unique<IoContext>();
    init_io_context(4); // Default 4 I/O threads
}

NvmePool::~NvmePool() {
    if (io_ctx_) {
        {
            std::lock_guard<std::mutex> lock(io_ctx_->queue_mutex);
            io_ctx_->shutdown = true;
        }
        io_ctx_->queue_cv.notify_all();
        for (auto& t : io_ctx_->workers) {
            if (t.joinable()) t.join();
        }

#ifdef TITAN_HAS_IO_URING
        if (io_ctx_->ring_initialized) {
            io_uring_queue_exit(&io_ctx_->ring);
        }
#endif
    }
}

void NvmePool::init_io_context(int num_threads) {
#ifdef TITAN_HAS_IO_URING
    // Try to initialize io_uring
    struct io_uring_params params = {};
    params.flags = IORING_SETUP_SQPOLL; // Kernel-side polling for lower latency
    params.sq_thread_idle = 2000;       // 2s idle before sleeping

    if (io_uring_queue_init_params(256, &io_ctx_->ring, &params) == 0) {
        io_ctx_->use_io_uring = true;
        io_ctx_->ring_initialized = true;
        LOG_INFO("NVMe I/O: using io_uring (SQPOLL mode)");
        return;
    }

    // Fallback: try without SQPOLL
    memset(&params, 0, sizeof(params));
    if (io_uring_queue_init_params(256, &io_ctx_->ring, &params) == 0) {
        io_ctx_->use_io_uring = true;
        io_ctx_->ring_initialized = true;
        LOG_INFO("NVMe I/O: using io_uring (standard mode)");
        return;
    }

    LOG_WARN("io_uring initialization failed, falling back to thread pool");
#endif

    // Thread pool fallback
    io_ctx_->use_io_uring = false;
    for (int i = 0; i < num_threads; i++) {
        io_ctx_->workers.emplace_back(&IoContext::worker_thread, io_ctx_.get());
    }
    LOG_INFO("NVMe I/O: using thread pool (%d threads)", num_threads);
}

ssize_t NvmePool::read_file(const std::string& path, void* dst, size_t bytes,
                             off_t offset) {
    auto t0 = std::chrono::steady_clock::now();

    int flags = O_RDONLY;
#ifdef O_DIRECT
    // O_DIRECT requires aligned buffers — check alignment
    if (((uintptr_t)dst & 511) == 0 && (bytes & 511) == 0 && (offset & 511) == 0) {
        flags |= O_DIRECT;
    }
#endif

    int fd = open(path.c_str(), flags);
    if (fd < 0) {
        LOG_ERROR("Cannot open %s: %s", path.c_str(), strerror(errno));
        return -1;
    }

    ssize_t total_read = 0;
    char* buf = (char*)dst;

    while ((size_t)total_read < bytes) {
        ssize_t n = pread(fd, buf + total_read, bytes - total_read,
                          offset + total_read);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            break;
        }
        total_read += n;
    }

    close(fd);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed > 0) {
        last_bandwidth_ = (float)(total_read / elapsed / 1e9);
    }
    total_read_ += total_read;

    return total_read;
}

void NvmePool::async_read_file(const std::string& path, void* dst, size_t bytes,
                                off_t offset, ReadCallback callback) {
#ifdef TITAN_HAS_IO_URING
    if (io_ctx_->use_io_uring) {
        // Submit io_uring read
        int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) {
            fd = open(path.c_str(), O_RDONLY);
        }
        if (fd < 0) {
            if (callback) callback(-1);
            return;
        }

        struct io_uring_sqe* sqe = io_uring_get_sqe(&io_ctx_->ring);
        if (!sqe) {
            close(fd);
            if (callback) callback(-1);
            return;
        }

        io_uring_prep_read(sqe, fd, dst, bytes, offset);
        // Store callback info as user data
        // In production, use a proper completion handler structure
        io_uring_sqe_set_data(sqe, (void*)(uintptr_t)fd);
        io_uring_submit(&io_ctx_->ring);

        // For now, wait for completion synchronously
        // TODO: Implement proper async completion handling with epoll
        struct io_uring_cqe* cqe;
        io_uring_wait_cqe(&io_ctx_->ring, &cqe);
        ssize_t result = cqe->res;
        io_uring_cqe_seen(&io_ctx_->ring, cqe);
        close(fd);

        total_read_ += (result > 0 ? result : 0);
        if (callback) callback(result);
        return;
    }
#endif

    // Thread pool fallback
    IoContext::IoTask task;
    task.path = path;
    task.dst = dst;
    task.bytes = bytes;
    task.offset = offset;
    task.callback = [this, callback](ssize_t n) {
        total_read_ += (n > 0 ? n : 0);
        if (callback) callback(n);
    };

    {
        std::lock_guard<std::mutex> lock(io_ctx_->queue_mutex);
        io_ctx_->task_queue.push(std::move(task));
    }
    io_ctx_->queue_cv.notify_one();
}

void NvmePool::batch_read(const std::vector<ReadRequest>& requests,
                           ReadCallback on_all_complete) {
    if (requests.empty()) {
        if (on_all_complete) on_all_complete(0);
        return;
    }

    auto remaining = std::make_shared<std::atomic<int>>(requests.size());
    auto total_bytes = std::make_shared<std::atomic<ssize_t>>(0);

    for (const auto& req : requests) {
        async_read_file(req.path, req.dst, req.bytes, req.offset,
            [remaining, total_bytes, on_all_complete](ssize_t n) {
                if (n > 0) total_bytes->fetch_add(n);
                if (remaining->fetch_sub(1) == 1) {
                    // All reads complete
                    if (on_all_complete) {
                        on_all_complete(total_bytes->load());
                    }
                }
            });
    }
}

void NvmePool::copy_to(void* dst, const void* src, size_t bytes, MemoryTier dst_tier) {
    // NVMe -> RAM: read from file
    // This is a simplified path; real usage goes through read_file()
    LOG_WARN("NvmePool::copy_to called directly — use read_file() instead");
}

void NvmePool::copy_from(void* dst, const void* src, size_t bytes, MemoryTier src_tier) {
    LOG_WARN("NvmePool::copy_from not implemented (NVMe is read-only for inference)");
}

} // namespace titan
