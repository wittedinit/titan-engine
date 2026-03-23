#pragma once

#include "core/types.h"
#include <string>
#include <vector>

namespace titan {

// ============================================================================
// GPU Information
// ============================================================================

struct GpuInfo {
    int             device_id       = -1;
    std::string     name;
    size_t          vram_total      = 0;    // bytes
    size_t          vram_free       = 0;    // bytes
    int             compute_cap_major = 0;
    int             compute_cap_minor = 0;
    int             sm_count        = 0;
    size_t          l2_cache_size   = 0;
    int             pcie_gen        = 0;
    int             pcie_width      = 0;
    float           pcie_bandwidth  = 0;    // GB/s theoretical

    // Capability flags
    bool            has_fp8         = false; // Compute >= 8.9 (Ada+)
    bool            has_fp4         = false; // Compute >= 10.0 (Blackwell)
    bool            has_bf16        = false; // Compute >= 8.0 (Ampere+)
    bool            has_int8_tc     = false; // Tensor core INT8
};

// ============================================================================
// CPU Information
// ============================================================================

struct CpuInfo {
    std::string     model_name;
    int             physical_cores  = 0;
    int             logical_cores   = 0;
    int             numa_nodes      = 0;
    size_t          l1_cache        = 0;
    size_t          l2_cache        = 0;
    size_t          l3_cache        = 0;

    // Instruction set flags
    bool            has_avx2        = false;
    bool            has_avx512f     = false;
    bool            has_avx512_vnni = false;
    bool            has_amx_bf16    = false;
    bool            has_amx_int8    = false;
};

// ============================================================================
// Memory Information
// ============================================================================

struct MemoryInfo {
    size_t          total_ram       = 0;    // bytes
    size_t          available_ram   = 0;    // bytes
    size_t          swap_total      = 0;
    int             num_channels    = 0;    // DDR channels
    float           bandwidth       = 0;    // GB/s estimated
};

// ============================================================================
// Storage Information
// ============================================================================

struct StorageInfo {
    std::string     path;
    std::string     device;
    std::string     fs_type;
    size_t          total_bytes     = 0;
    size_t          free_bytes      = 0;
    float           seq_read_bw     = 0;    // GB/s measured
    float           seq_write_bw    = 0;    // GB/s measured
    bool            is_nvme         = false;
    bool            supports_io_uring = false;
    bool            supports_direct_io = false;
    int             raid_level      = -1;   // -1 = not RAID
    int             raid_disks      = 0;
};

// ============================================================================
// Hardware Profile (complete system view)
// ============================================================================

struct HardwareProfile {
    std::vector<GpuInfo>    gpus;
    CpuInfo                 cpu;
    MemoryInfo              memory;
    std::vector<StorageInfo> storage;

    // Derived optimal configuration
    size_t  optimal_vram_budget() const;
    size_t  optimal_ram_budget() const;
    float   estimated_nvme_bandwidth() const;
    DType   best_gpu_dtype() const;
    bool    can_use_io_uring() const;
};

// ============================================================================
// Detection Functions
// ============================================================================

// Detect all hardware and return a complete profile
HardwareProfile detect_hardware();

// Individual detection (for testing)
std::vector<GpuInfo> detect_gpus();
CpuInfo detect_cpu();
MemoryInfo detect_memory();
StorageInfo detect_storage(const std::string& path);

// Print a human-readable hardware summary
void print_hardware_summary(const HardwareProfile& hw);

// ============================================================================
// Execution Plan (what goes where)
// ============================================================================

struct ExecutionPlan {
    // Which layers go on which device
    struct LayerPlacement {
        uint32_t    layer_id;
        MemoryTier  attention_weights;  // Where to store attention params
        MemoryTier  ffn_weights;        // Where to store FFN/expert params
        MemoryTier  kv_cache;           // Where to store KV cache for this layer
        int         gpu_id;             // -1 = CPU
    };

    std::vector<LayerPlacement> layers;

    MemoryTier  embedding_tier  = MemoryTier::VRAM;
    MemoryTier  lm_head_tier    = MemoryTier::VRAM;

    // Memory allocation summary
    size_t      vram_used       = 0;
    size_t      ram_used        = 0;
    size_t      nvme_used       = 0;

    // Expert caching strategy (MoE only)
    size_t      expert_cache_ram_mb     = 0;
    size_t      expert_cache_vram_mb    = 0;
    uint32_t    expert_prefetch_layers  = 1;
};

// Generate an optimal execution plan for given model + hardware
ExecutionPlan plan_execution(const ModelConfig& model, const HardwareProfile& hw,
                             const RuntimeConfig& runtime);

} // namespace titan
