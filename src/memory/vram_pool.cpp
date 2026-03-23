#include "memory/memory_manager.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace titan {

VramPool::VramPool(size_t budget_bytes, int gpu_id)
    : gpu_id_(gpu_id), capacity_(budget_bytes) {
    cudaSetDevice(gpu_id);

    // Pre-allocate a large chunk to avoid repeated cudaMalloc calls
    void* base = nullptr;
    cudaError_t err = cudaMalloc(&base, budget_bytes);
    if (err != cudaSuccess) {
        // Try smaller allocation
        budget_bytes = budget_bytes * 3 / 4;
        err = cudaMalloc(&base, budget_bytes);
        if (err != cudaSuccess) {
            LOG_ERROR("cudaMalloc failed for VRAM pool: %s", cudaGetErrorString(err));
            capacity_ = 0;
            return;
        }
        capacity_ = budget_bytes;
        LOG_WARN("VRAM pool reduced to %.1f GB", capacity_ / 1e9);
    }

    // Initialize with a single free block
    blocks_.push_back({base, budget_bytes, true});
}

VramPool::~VramPool() {
    if (!blocks_.empty() && blocks_[0].ptr) {
        // All sub-allocations come from the first block's base pointer
        // Find the base (smallest address)
        void* base = blocks_[0].ptr;
        for (auto& b : blocks_) {
            if (b.ptr < base) base = b.ptr;
        }
        cudaSetDevice(gpu_id_);
        cudaFree(base);
    }
}

void* VramPool::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Align to 256 bytes (CUDA alignment requirement)
    bytes = (bytes + 255) & ~(size_t)255;

    // First-fit allocation
    for (size_t i = 0; i < blocks_.size(); i++) {
        if (blocks_[i].free && blocks_[i].size >= bytes) {
            blocks_[i].free = false;

            // Split if there's significant leftover
            size_t leftover = blocks_[i].size - bytes;
            if (leftover > 4096) {
                Block remainder;
                remainder.ptr = (char*)blocks_[i].ptr + bytes;
                remainder.size = leftover;
                remainder.free = true;
                blocks_[i].size = bytes;
                blocks_.insert(blocks_.begin() + i + 1, remainder);
            }

            used_ += blocks_[i].size;
            return blocks_[i].ptr;
        }
    }

    LOG_ERROR("VRAM pool: cannot allocate %zu bytes (%.1f MB used / %.1f MB total)",
              bytes, used_ / 1e6, capacity_ / 1e6);
    return nullptr;
}

void VramPool::free(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t i = 0; i < blocks_.size(); i++) {
        if (blocks_[i].ptr == ptr) {
            blocks_[i].free = true;
            used_ -= blocks_[i].size;

            // Coalesce with next block
            if (i + 1 < blocks_.size() && blocks_[i + 1].free) {
                blocks_[i].size += blocks_[i + 1].size;
                blocks_.erase(blocks_.begin() + i + 1);
            }
            // Coalesce with previous block
            if (i > 0 && blocks_[i - 1].free) {
                blocks_[i - 1].size += blocks_[i].size;
                blocks_.erase(blocks_.begin() + i);
            }
            return;
        }
    }
    LOG_WARN("VRAM pool: free() called with unknown pointer %p", ptr);
}

void VramPool::copy_to(void* dst, const void* src, size_t bytes, MemoryTier dst_tier) {
    cudaSetDevice(gpu_id_);
    switch (dst_tier) {
        case MemoryTier::RAM:
            cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
            break;
        case MemoryTier::VRAM:
            cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
            break;
        default:
            LOG_ERROR("VramPool::copy_to: unsupported destination tier");
            break;
    }
}

void VramPool::copy_from(void* dst, const void* src, size_t bytes, MemoryTier src_tier) {
    cudaSetDevice(gpu_id_);
    switch (src_tier) {
        case MemoryTier::RAM:
            cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
            break;
        case MemoryTier::VRAM:
            cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
            break;
        default:
            LOG_ERROR("VramPool::copy_from: unsupported source tier");
            break;
    }
}

void VramPool::async_copy_from_ram(void* dst_vram, const void* src_ram, size_t bytes,
                                    void* cuda_stream) {
    cudaSetDevice(gpu_id_);
    cudaStream_t stream = cuda_stream ? (cudaStream_t)cuda_stream : 0;
    cudaMemcpyAsync(dst_vram, src_ram, bytes, cudaMemcpyHostToDevice, stream);
}

} // namespace titan
