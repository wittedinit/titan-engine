#include "memory/memory_manager.h"
#include "core/logging.h"
#include <sys/stat.h>

namespace titan {

MemoryManager::MemoryManager(const HardwareProfile& hw, const RuntimeConfig& cfg) {
    size_t vram_budget = cfg.vram_budget_mb > 0
        ? cfg.vram_budget_mb * 1024ULL * 1024
        : hw.optimal_vram_budget();
    size_t ram_budget = cfg.ram_budget_mb > 0
        ? cfg.ram_budget_mb * 1024ULL * 1024
        : hw.optimal_ram_budget();

    int gpu_id = hw.gpus.empty() ? -1 : 0;

    if (gpu_id >= 0) {
        vram_ = std::make_unique<VramPool>(vram_budget, gpu_id);
        LOG_INFO("VRAM pool: %.1f GB on GPU %d", vram_budget / 1e9, gpu_id);
    }

    ram_ = std::make_unique<RamPool>(ram_budget);
    LOG_INFO("RAM pool: %.1f GB budget (demand-allocated, not reserved upfront)", ram_budget / 1e9);

    std::string nvme_path = cfg.nvme_cache_path.empty() ? cfg.model_path : cfg.nvme_cache_path;
    nvme_ = std::make_unique<NvmePool>(nvme_path, cfg.nvme_cache_mb * 1024ULL * 1024);
    expert_base_path_ = nvme_path;
    LOG_INFO("NVMe pool: %s", nvme_path.c_str());

    // Expert cache budget: use remaining RAM after model weights
    cache_budget_ = ram_budget / 2; // Conservative: 50% of RAM for expert cache
    LOG_INFO("Expert cache budget: %.1f GB", cache_budget_ / 1e9);
}

MemoryManager::~MemoryManager() {
    // Clean up expert cache
    for (auto& entry : cache_lru_) {
        if (entry.data) {
            ram_->free(entry.data);
        }
    }
}

void MemoryManager::load_tensor(Tensor& tensor, MemoryTier target) {
    if (tensor.desc.tier == target) return;

    void* dst = nullptr;

    switch (target) {
        case MemoryTier::VRAM:
            dst = vram_->allocate(tensor.desc.byte_size);
            if (!dst) {
                LOG_ERROR("Failed to allocate %.1f MB in VRAM",
                          tensor.desc.byte_size / 1e6);
                return;
            }
            if (tensor.desc.tier == MemoryTier::RAM && tensor.data) {
                vram_->copy_from(dst, tensor.data, tensor.desc.byte_size, MemoryTier::RAM);
            }
            break;

        case MemoryTier::RAM:
            dst = ram_->allocate(tensor.desc.byte_size);
            if (!dst) {
                LOG_ERROR("Failed to allocate %.1f MB in RAM",
                          tensor.desc.byte_size / 1e6);
                return;
            }
            break;

        default:
            LOG_ERROR("Cannot load tensor to tier %s", tier_name(target));
            return;
    }

    // Free old allocation
    if (tensor.data && tensor.owns) {
        switch (tensor.desc.tier) {
            case MemoryTier::VRAM: vram_->free(tensor.data); break;
            case MemoryTier::RAM:  ram_->free(tensor.data); break;
            default: break;
        }
    }

    tensor.data = dst;
    tensor.desc.tier = target;
    tensor.owns = true;
}

void MemoryManager::migrate(Tensor& tensor, MemoryTier target) {
    if (tensor.desc.tier == target) return;

    void* new_data = nullptr;
    size_t bytes = tensor.desc.byte_size;

    // Allocate in target tier
    switch (target) {
        case MemoryTier::VRAM:
            new_data = vram_->allocate(bytes);
            break;
        case MemoryTier::RAM:
            new_data = ram_->allocate(bytes);
            break;
        default:
            return;
    }

    if (!new_data) {
        LOG_ERROR("Migration failed: cannot allocate %zu bytes in %s",
                  bytes, tier_name(target));
        return;
    }

    // Copy data
    if (tensor.data) {
        if (target == MemoryTier::VRAM) {
            vram_->copy_from(new_data, tensor.data, bytes, tensor.desc.tier);
        } else if (target == MemoryTier::RAM && tensor.desc.tier == MemoryTier::VRAM) {
            vram_->copy_to(new_data, tensor.data, bytes, MemoryTier::RAM);
        }
    }

    // Free old
    if (tensor.data && tensor.owns) {
        switch (tensor.desc.tier) {
            case MemoryTier::VRAM: vram_->free(tensor.data); break;
            case MemoryTier::RAM:  ram_->free(tensor.data); break;
            default: break;
        }
    }

    tensor.data = new_data;
    tensor.desc.tier = target;
    tensor.owns = true;
}

void* MemoryManager::get_expert(uint32_t layer, uint32_t expert_id, size_t expert_bytes) {
    uint64_t key = ((uint64_t)layer << 32) | expert_id;

    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Check cache
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
        // Move to front (most recently used)
        cache_lru_.splice(cache_lru_.begin(), cache_lru_, it->second);
        cache_stats_.hits++;
        return it->second->data;
    }

    cache_stats_.misses++;

    // Check if .bin file exists before allocating RAM — avoids thousands of
    // alloc+free cycles for safetensors-backed models that never use .bin files.
    char path[256];
    snprintf(path, sizeof(path), "%s/layer_%02u.bin",
             expert_base_path_.c_str(), layer);
    {
        struct stat st;
        if (stat(path, &st) != 0) {
            // .bin file doesn't exist — caller must use insert_expert() after loading
            // from safetensors. Return nullptr without wasting a RAM allocation.
            LOG_DEBUG("Expert %u/%u: no .bin file, must be loaded via safetensors",
                      layer, expert_id);
            return nullptr;
        }
    }

    // Need to load from NVMe
    // First, ensure we have space
    if (cache_used_ + expert_bytes > cache_budget_) {
        evict_experts(expert_bytes);
    }

    // Allocate in RAM
    void* data = ram_->allocate(expert_bytes);
    if (!data) {
        LOG_ERROR("Cannot allocate %zu bytes for expert %u/%u in RAM",
                  expert_bytes, layer, expert_id);
        return nullptr;
    }

    // Read from NVMe
    off_t offset = (off_t)expert_id * expert_bytes;
    ssize_t read = nvme_->read_file(path, data, expert_bytes, offset);

    if (read != (ssize_t)expert_bytes) {
        LOG_ERROR("Failed to read expert %u/%u from NVMe (got %zd/%zu)",
                  layer, expert_id, read, expert_bytes);
        ram_->free(data);
        return nullptr;
    }

    // Insert into cache
    CacheEntry entry{key, data, expert_bytes};
    cache_lru_.push_front(entry);
    cache_map_[key] = cache_lru_.begin();
    cache_used_ += expert_bytes;

    return data;
}

void MemoryManager::prefetch_expert(uint32_t layer, uint32_t expert_id, size_t expert_bytes) {
    uint64_t key = ((uint64_t)layer << 32) | expert_id;

    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Already cached?
    if (cache_map_.find(key) != cache_map_.end()) return;

    // Ensure space
    if (cache_used_ + expert_bytes > cache_budget_) {
        evict_experts(expert_bytes);
    }

    void* data = ram_->allocate(expert_bytes);
    if (!data) return;

    // Async read from NVMe
    char path[256];
    snprintf(path, sizeof(path), "%s/layer_%02u.bin",
             expert_base_path_.c_str(), layer);
    off_t offset = (off_t)expert_id * expert_bytes;

    nvme_->async_read_file(path, data, expert_bytes, offset,
        [this, key, data, expert_bytes](ssize_t bytes_read) {
            if (bytes_read != (ssize_t)expert_bytes) {
                ram_->free(data);
                return;
            }
            std::lock_guard<std::mutex> lock(cache_mutex_);
            if (cache_map_.find(key) != cache_map_.end()) {
                // Already cached (race condition)
                ram_->free(data);
                return;
            }
            CacheEntry entry{key, data, expert_bytes};
            cache_lru_.push_front(entry);
            cache_map_[key] = cache_lru_.begin();
            cache_used_ += expert_bytes;
        });
}

void MemoryManager::insert_expert(uint32_t layer, uint32_t expert_id,
                                   const void* src, size_t expert_bytes) {
    uint64_t key = ((uint64_t)layer << 32) | expert_id;
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Already cached — no-op
    if (cache_map_.find(key) != cache_map_.end()) return;

    if (cache_used_ + expert_bytes > cache_budget_) {
        evict_experts(expert_bytes);
    }

    void* data = ram_->allocate(expert_bytes);
    if (!data) return;

    memcpy(data, src, expert_bytes);
    CacheEntry entry{key, data, expert_bytes};
    cache_lru_.push_front(entry);
    cache_map_[key] = cache_lru_.begin();
    cache_used_ += expert_bytes;
}

void MemoryManager::evict_experts(size_t needed_bytes) {
    // LRU eviction from the back
    while (cache_used_ + needed_bytes > cache_budget_ && !cache_lru_.empty()) {
        auto& victim = cache_lru_.back();
        cache_map_.erase(victim.key);
        if (victim.data) {
            ram_->free(victim.data);
        }
        cache_used_ -= victim.size;
        cache_stats_.evictions++;
        cache_lru_.pop_back();
    }
}

void MemoryManager::print_usage() const {
    LOG_INFO("=== Memory Usage ===");
    if (vram_) {
        LOG_INFO("VRAM: %.1f / %.1f GB (%.0f%%)",
                 vram_->used() / 1e9, vram_->capacity() / 1e9,
                 100.0 * vram_->used() / vram_->capacity());
    }
    LOG_INFO("RAM: %.1f / %.1f GB (%.0f%%)",
             ram_->used() / 1e9, ram_->capacity() / 1e9,
             100.0 * ram_->used() / ram_->capacity());
    LOG_INFO("Expert cache: %.1f / %.1f GB (%.0f%% hit rate, %zu evictions)",
             cache_used_ / 1e9, cache_budget_ / 1e9,
             cache_stats_.hit_rate() * 100,
             cache_stats_.evictions);
    LOG_INFO("NVMe total read: %.1f GB (last bw: %.1f GB/s)",
             nvme_->total_bytes_read() / 1e9,
             nvme_->last_read_bandwidth());
}

} // namespace titan
