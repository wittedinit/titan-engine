#pragma once

#include "core/types.h"
#include "core/hardware.h"
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <list>
#include <functional>

namespace titan {

// ============================================================================
// Abstract Memory Pool Interface
// ============================================================================

class MemoryPool {
public:
    virtual ~MemoryPool() = default;

    virtual MemoryTier tier() const = 0;
    virtual size_t capacity() const = 0;
    virtual size_t used() const = 0;
    virtual size_t available() const { return capacity() - used(); }

    // Allocate a buffer of given size. Returns nullptr on failure.
    virtual void* allocate(size_t bytes) = 0;

    // Free a previously allocated buffer.
    virtual void free(void* ptr) = 0;

    // Copy data between this pool and another
    // src_pool/dst_pool indicate the tier of the source/destination
    virtual void copy_to(void* dst, const void* src, size_t bytes,
                         MemoryTier dst_tier) = 0;
    virtual void copy_from(void* dst, const void* src, size_t bytes,
                           MemoryTier src_tier) = 0;
};

// ============================================================================
// VRAM Pool — CUDA device memory with sub-allocation
// ============================================================================

class VramPool : public MemoryPool {
public:
    explicit VramPool(size_t budget_bytes, int gpu_id = 0);
    ~VramPool() override;

    MemoryTier tier() const override { return MemoryTier::VRAM; }
    size_t capacity() const override { return capacity_; }
    size_t used() const override { return used_; }

    void* allocate(size_t bytes) override;
    void free(void* ptr) override;
    void copy_to(void* dst, const void* src, size_t bytes, MemoryTier dst_tier) override;
    void copy_from(void* dst, const void* src, size_t bytes, MemoryTier src_tier) override;

    // Async copy (returns CUDA stream event)
    void async_copy_from_ram(void* dst_vram, const void* src_ram, size_t bytes,
                             void* cuda_stream = nullptr);

    int gpu_id() const { return gpu_id_; }

private:
    int         gpu_id_;
    size_t      capacity_;
    size_t      used_ = 0;
    std::mutex  mutex_;

    // Simple block allocator (upgrade to buddy/slab later if needed)
    struct Block {
        void*   ptr;
        size_t  size;
        bool    free;
    };
    std::vector<Block> blocks_;
};

// ============================================================================
// RAM Pool — Pinned system memory for fast GPU transfers
// ============================================================================

class RamPool : public MemoryPool {
public:
    explicit RamPool(size_t budget_bytes);
    ~RamPool() override;

    MemoryTier tier() const override { return MemoryTier::RAM; }
    size_t capacity() const override { return capacity_; }
    size_t used() const override { return used_; }

    void* allocate(size_t bytes) override;
    void free(void* ptr) override;
    void copy_to(void* dst, const void* src, size_t bytes, MemoryTier dst_tier) override;
    void copy_from(void* dst, const void* src, size_t bytes, MemoryTier src_tier) override;

    // Allocate pinned (page-locked) memory for DMA transfers
    void* allocate_pinned(size_t bytes);
    void free_pinned(void* ptr);

private:
    size_t      capacity_;
    size_t      used_ = 0;
    std::mutex  mutex_;

    // Per-instance allocation tracking (not file-scoped statics, which would be
    // shared across multiple RamPool instances).
    std::unordered_map<void*, size_t> alloc_sizes_;
    std::unordered_map<void*, bool>   is_pinned_;
};

// ============================================================================
// NVMe Pool — Direct I/O from NVMe storage
// ============================================================================

class NvmePool : public MemoryPool {
public:
    NvmePool(const std::string& base_path, size_t cache_bytes = 0);
    ~NvmePool() override;

    MemoryTier tier() const override { return MemoryTier::NVME; }
    size_t capacity() const override { return capacity_; }
    size_t used() const override { return used_; }

    // NVMe pool doesn't do traditional allocate/free
    // Instead, it reads from files on demand
    void* allocate(size_t bytes) override { return nullptr; } // Not applicable
    void free(void* ptr) override {} // Not applicable
    void copy_to(void* dst, const void* src, size_t bytes, MemoryTier dst_tier) override;
    void copy_from(void* dst, const void* src, size_t bytes, MemoryTier src_tier) override;

    // Read data from a file at given offset into a destination buffer
    // dst must be in RAM or pinned memory
    ssize_t read_file(const std::string& path, void* dst, size_t bytes,
                      off_t offset = 0);

    // Async read (io_uring or threaded pread)
    // Calls callback when complete
    using ReadCallback = std::function<void(ssize_t bytes_read)>;
    void async_read_file(const std::string& path, void* dst, size_t bytes,
                         off_t offset, ReadCallback callback);

    // Batch read: read multiple expert chunks in parallel
    struct ReadRequest {
        std::string path;
        void*       dst;
        size_t      bytes;
        off_t       offset;
    };
    void batch_read(const std::vector<ReadRequest>& requests,
                    ReadCallback on_all_complete);

    // Telemetry
    float last_read_bandwidth() const { return last_bandwidth_.load(std::memory_order_relaxed); }
    size_t total_bytes_read() const { return total_read_.load(std::memory_order_relaxed); }

private:
    std::string base_path_;
    size_t      capacity_ = 0;
    size_t      used_ = 0;
    std::atomic<float>  last_bandwidth_{0};
    std::atomic<size_t> total_read_{0};

    // io_uring or thread pool for async I/O
    struct IoContext;
    std::unique_ptr<IoContext> io_ctx_;

    void init_io_context(int num_threads);
};

// ============================================================================
// 3-Tier Memory Manager
// ============================================================================

class MemoryManager {
public:
    MemoryManager(const HardwareProfile& hw, const RuntimeConfig& cfg);
    ~MemoryManager();

    // Access individual pools
    VramPool& vram() { return *vram_; }
    RamPool& ram() { return *ram_; }
    NvmePool& nvme() { return *nvme_; }

    // Convenience wrappers for VRAM pool sub-allocation
    size_t vram_free_bytes() const { return vram_ ? vram_->available() : 0; }
    void*  vram_alloc(size_t bytes) { return vram_ ? vram_->allocate(bytes) : nullptr; }
    void   vram_free(void* ptr)     { if (vram_) vram_->free(ptr); }

    // High-level: load tensor to target tier
    // Handles the chain: NVMe → RAM → VRAM with proper staging
    void load_tensor(Tensor& tensor, MemoryTier target);

    // Move tensor between tiers
    void migrate(Tensor& tensor, MemoryTier target);

    // Expert cache for MoE models
    // Returns pointer to expert weights in RAM (loading from NVMe if needed)
    void* get_expert(uint32_t layer, uint32_t expert_id, size_t expert_bytes);

    // Prefetch expert to RAM (non-blocking)
    void prefetch_expert(uint32_t layer, uint32_t expert_id, size_t expert_bytes);

    // Insert expert data into RAM cache (called after on-demand load from safetensors).
    // Copies expert_bytes from src into a new RAM allocation and inserts into LRU.
    // No-op if the expert is already cached.
    void insert_expert(uint32_t layer, uint32_t expert_id,
                       const void* src, size_t expert_bytes);

    // Expert cache stats
    struct CacheStats {
        size_t  hits = 0;
        size_t  misses = 0;
        size_t  evictions = 0;
        float   hit_rate() const {
            size_t total = hits + misses;
            return total > 0 ? (float)hits / total : 0;
        }
    };
    CacheStats expert_cache_stats() const { return cache_stats_; }

    // Print memory usage summary
    void print_usage() const;

private:
    std::unique_ptr<VramPool> vram_;
    std::unique_ptr<RamPool>  ram_;
    std::unique_ptr<NvmePool> nvme_;

    // Expert LRU cache in RAM
    struct CacheEntry {
        uint64_t    key;        // (layer << 32) | expert_id
        void*       data;
        size_t      size;
    };
    std::list<CacheEntry> cache_lru_;
    std::unordered_map<uint64_t, std::list<CacheEntry>::iterator> cache_map_;
    size_t cache_used_ = 0;
    size_t cache_budget_ = 0;
    std::mutex cache_mutex_;
    CacheStats cache_stats_;

    // Expert file paths (set during model loading)
    std::string expert_base_path_;

    void evict_experts(size_t needed_bytes);
};

} // namespace titan
