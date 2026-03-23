#include "memory/memory_manager.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace titan {

// Track allocations for proper cleanup
static std::unordered_map<void*, size_t> s_alloc_sizes;
static std::unordered_map<void*, bool> s_is_pinned;

RamPool::RamPool(size_t budget_bytes)
    : capacity_(budget_bytes) {
}

RamPool::~RamPool() {
    // All allocations should be freed by callers before destruction
    if (used_ > 0) {
        LOG_WARN("RAM pool destroyed with %zu bytes still allocated", used_);
    }
}

void* RamPool::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (used_ + bytes > capacity_) {
        LOG_ERROR("RAM pool: cannot allocate %zu bytes (%.1f GB used / %.1f GB total)",
                  bytes, used_ / 1e9, capacity_ / 1e9);
        return nullptr;
    }

    // Align to 64 bytes for AVX-512
    size_t aligned_bytes = (bytes + 63) & ~(size_t)63;

    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, aligned_bytes) != 0) {
        LOG_ERROR("RAM pool: posix_memalign failed for %zu bytes", aligned_bytes);
        return nullptr;
    }

    used_ += aligned_bytes;
    s_alloc_sizes[ptr] = aligned_bytes;
    s_is_pinned[ptr] = false;
    return ptr;
}

void RamPool::free(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = s_alloc_sizes.find(ptr);
    if (it == s_alloc_sizes.end()) {
        LOG_WARN("RAM pool: free() called with unknown pointer %p", ptr);
        return;
    }

    // Unpin if pinned
    if (s_is_pinned[ptr]) {
        cudaHostUnregister(ptr);
    }

    used_ -= it->second;
    s_alloc_sizes.erase(it);
    s_is_pinned.erase(ptr);
    ::free(ptr);
}

void* RamPool::allocate_pinned(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (used_ + bytes > capacity_) {
        LOG_ERROR("RAM pool: cannot allocate %zu pinned bytes", bytes);
        return nullptr;
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaMallocHost failed: %s", cudaGetErrorString(err));
        return nullptr;
    }

    used_ += bytes;
    s_alloc_sizes[ptr] = bytes;
    s_is_pinned[ptr] = true;
    return ptr;
}

void RamPool::free_pinned(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = s_alloc_sizes.find(ptr);
    if (it != s_alloc_sizes.end()) {
        used_ -= it->second;
        s_alloc_sizes.erase(it);
        s_is_pinned.erase(ptr);
    }
    cudaFreeHost(ptr);
}

void RamPool::copy_to(void* dst, const void* src, size_t bytes, MemoryTier dst_tier) {
    switch (dst_tier) {
        case MemoryTier::RAM:
            memcpy(dst, src, bytes);
            break;
        case MemoryTier::VRAM:
            cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
            break;
        default:
            LOG_ERROR("RamPool::copy_to: unsupported destination tier");
            break;
    }
}

void RamPool::copy_from(void* dst, const void* src, size_t bytes, MemoryTier src_tier) {
    switch (src_tier) {
        case MemoryTier::RAM:
            memcpy(dst, src, bytes);
            break;
        case MemoryTier::VRAM:
            cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
            break;
        default:
            LOG_ERROR("RamPool::copy_from: unsupported source tier");
            break;
    }
}

} // namespace titan
