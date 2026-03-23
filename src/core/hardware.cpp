#include "core/hardware.h"
#include "core/logging.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <sys/sysinfo.h>
#include <unistd.h>

#ifdef __x86_64__
#include <cpuid.h>
#endif

#ifdef TITAN_HAS_IO_URING
#include <sys/utsname.h>
#endif

// Forward declare CUDA runtime functions to avoid header dependency in .cpp
// (linked via CUDA::cudart)
extern "C" {
    struct cudaDeviceProp;
    int cudaGetDeviceCount(int* count);
    int cudaGetDeviceProperties(void* prop, int device);
    int cudaMemGetInfo(size_t* free, size_t* total);
    int cudaSetDevice(int device);
}

namespace titan {

// ============================================================================
// GPU Detection
// ============================================================================

std::vector<GpuInfo> detect_gpus() {
    std::vector<GpuInfo> gpus;

    int count = 0;
    if (cudaGetDeviceCount(&count) != 0 || count == 0) {
        LOG_WARN("No CUDA GPUs detected");
        return gpus;
    }

    // cudaDeviceProp is a large struct; we allocate a buffer
    // In practice, link against CUDA headers properly
    struct CudaDevPropMinimal {
        char name[256];
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int warpSize;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        int clockRate;
        size_t totalConstMem;
        int major;
        int minor;
        size_t textureAlignment;
        size_t texturePitchAlignment;
        int deviceOverlap;
        int multiProcessorCount;
        // ... many more fields, but we only need a few
        char _pad[4096]; // Oversized buffer for safety
    };

    for (int i = 0; i < count; i++) {
        CudaDevPropMinimal prop = {};
        cudaSetDevice(i);

        if (cudaGetDeviceProperties(&prop, i) != 0) continue;

        GpuInfo gpu;
        gpu.device_id = i;
        gpu.name = prop.name;
        gpu.vram_total = prop.totalGlobalMem;
        gpu.compute_cap_major = prop.major;
        gpu.compute_cap_minor = prop.minor;
        gpu.sm_count = prop.multiProcessorCount;

        // Get free memory
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == 0) {
            gpu.vram_free = free_mem;
        }

        // Capability flags based on compute capability
        float cc = prop.major + prop.minor * 0.1f;
        gpu.has_bf16     = (cc >= 8.0f);    // Ampere+
        gpu.has_int8_tc  = (cc >= 7.5f);    // Turing+
        gpu.has_fp8      = (cc >= 8.9f);    // Ada Lovelace+
        gpu.has_fp4      = (cc >= 10.0f);   // Blackwell+

        // Estimate PCIe bandwidth (rough heuristic)
        // This should be measured, but reasonable defaults:
        if (cc >= 10.0f) {
            gpu.pcie_gen = 5;
            gpu.pcie_width = 16;
            gpu.pcie_bandwidth = 63.0f; // PCIe 5.0 x16 theoretical
        } else if (cc >= 8.0f) {
            gpu.pcie_gen = 4;
            gpu.pcie_width = 16;
            gpu.pcie_bandwidth = 31.5f;
        } else {
            gpu.pcie_gen = 3;
            gpu.pcie_width = 16;
            gpu.pcie_bandwidth = 15.75f;
        }

        gpus.push_back(gpu);
    }

    return gpus;
}

// ============================================================================
// CPU Detection
// ============================================================================

CpuInfo detect_cpu() {
    CpuInfo info;

    // Read model name from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                info.model_name = line.substr(pos + 2);
            }
            break;
        }
    }

    // Core counts
    info.logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    info.physical_cores = info.logical_cores / 2; // Assume HT/SMT

    // Read physical core count more accurately
    std::ifstream cores_file("/sys/devices/system/cpu/cpu0/topology/core_siblings_list");
    // Simplified: just use logical / 2 for now

    // NUMA nodes
    info.numa_nodes = 1;
    std::ifstream numa_file("/sys/devices/system/node/online");
    if (numa_file.good()) {
        std::string numa_str;
        std::getline(numa_file, numa_str);
        // Parse "0-N" format
        auto dash = numa_str.find('-');
        if (dash != std::string::npos) {
            info.numa_nodes = std::stoi(numa_str.substr(dash + 1)) + 1;
        }
    }

#ifdef __x86_64__
    // CPUID-based feature detection
    unsigned int eax, ebx, ecx, edx;

    // Check AVX2 (leaf 7, subleaf 0, ebx bit 5)
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        info.has_avx2        = (ebx >> 5) & 1;
        info.has_avx512f     = (ebx >> 16) & 1;
        info.has_avx512_vnni = (ecx >> 11) & 1;
    }

    // Check AMX (leaf 7, subleaf 0, edx bits 22-24)
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        info.has_amx_bf16 = (edx >> 22) & 1;
        info.has_amx_int8 = (edx >> 24) & 1;
    }
#endif

    // Cache sizes from sysfs
    auto read_cache = [](const char* path) -> size_t {
        std::ifstream f(path);
        std::string s;
        if (std::getline(f, s)) {
            size_t val = std::stoull(s);
            if (s.back() == 'K' || s.back() == 'k') val *= 1024;
            return val;
        }
        return 0;
    };

    info.l1_cache = read_cache("/sys/devices/system/cpu/cpu0/cache/index0/size");
    info.l2_cache = read_cache("/sys/devices/system/cpu/cpu0/cache/index2/size");
    info.l3_cache = read_cache("/sys/devices/system/cpu/cpu0/cache/index3/size");

    return info;
}

// ============================================================================
// Memory Detection
// ============================================================================

MemoryInfo detect_memory() {
    MemoryInfo info;

    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        info.total_ram = si.totalram * si.mem_unit;
        info.available_ram = si.freeram * si.mem_unit; // Rough; /proc/meminfo is more accurate
        info.swap_total = si.totalswap * si.mem_unit;
    }

    // More accurate available memory from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            size_t kb = 0;
            sscanf(line.c_str(), "MemAvailable: %zu kB", &kb);
            info.available_ram = kb * 1024;
            break;
        }
    }

    // Estimate DDR channels and bandwidth based on CPU type
    // A proper implementation would use dmidecode or read from BIOS
    // EPYC 9004: 12-channel DDR5 @ 4800 MT/s
    // Desktop: 2-channel DDR5 @ 5600 MT/s
    // For now, use a heuristic
    if (info.total_ram > 256ULL * 1024 * 1024 * 1024) {
        // Likely server-class (EPYC, Xeon)
        info.num_channels = 12;
        info.bandwidth = 460.0f; // 12ch DDR5-4800 theoretical
    } else if (info.total_ram > 64ULL * 1024 * 1024 * 1024) {
        info.num_channels = 4;
        info.bandwidth = 153.0f; // 4ch DDR5-4800
    } else {
        info.num_channels = 2;
        info.bandwidth = 89.6f;  // 2ch DDR5-5600
    }

    return info;
}

// ============================================================================
// Storage Detection
// ============================================================================

StorageInfo detect_storage(const std::string& path) {
    StorageInfo info;
    info.path = path;

    // Get filesystem info
    struct statvfs stat;
    if (statvfs(path.c_str(), &stat) == 0) {
        info.total_bytes = stat.f_blocks * stat.f_frsize;
        info.free_bytes = stat.f_bfree * stat.f_frsize;
    }

    // Detect NVMe by checking /sys/block/*/device/
    // This is simplified — a real implementation would use udev or lsblk
    std::ifstream mounts("/proc/mounts");
    std::string line;
    while (std::getline(mounts, line)) {
        if (line.find(path) != std::string::npos || path.find("/") == 0) {
            std::istringstream iss(line);
            iss >> info.device >> info.path >> info.fs_type;
            break;
        }
    }

    // Check if NVMe
    info.is_nvme = (info.device.find("nvme") != std::string::npos);

    // io_uring support check (kernel >= 5.1)
#ifdef TITAN_HAS_IO_URING
    struct utsname uts;
    if (uname(&uts) == 0) {
        int major = 0, minor = 0;
        sscanf(uts.release, "%d.%d", &major, &minor);
        info.supports_io_uring = (major > 5 || (major == 5 && minor >= 1));
    }
#endif

    // Direct I/O support
    info.supports_direct_io = true; // Almost always available on Linux

    // Default bandwidth estimates (should be measured with a quick benchmark)
    if (info.is_nvme) {
        info.seq_read_bw = 7.0f;   // Single NVMe PCIe 4.0 x4
        // Check for RAID by examining md devices
        std::ifstream mdstat("/proc/mdstat");
        if (mdstat.good()) {
            std::string md_line;
            while (std::getline(mdstat, md_line)) {
                if (md_line.find("raid0") != std::string::npos) {
                    // Count drives
                    int drives = 0;
                    for (size_t i = 0; i < md_line.length(); i++) {
                        if (md_line.substr(i, 4) == "nvme") drives++;
                    }
                    if (drives > 1) {
                        info.raid_level = 0;
                        info.raid_disks = drives;
                        info.seq_read_bw = 7.0f * drives; // Linear scaling for RAID 0
                    }
                    break;
                }
            }
        }
    } else {
        info.seq_read_bw = 0.5f; // SATA SSD default
    }

    return info;
}

// ============================================================================
// Full Hardware Detection
// ============================================================================

HardwareProfile detect_hardware() {
    HardwareProfile hw;
    hw.gpus = detect_gpus();
    hw.cpu = detect_cpu();
    hw.memory = detect_memory();
    hw.storage.push_back(detect_storage("/"));
    return hw;
}

// ============================================================================
// Hardware Profile Methods
// ============================================================================

size_t HardwareProfile::optimal_vram_budget() const {
    if (gpus.empty()) return 0;
    // Reserve 500MB for CUDA overhead, use 90% of remaining
    size_t vram = gpus[0].vram_free;
    size_t reserved = 512ULL * 1024 * 1024;
    return vram > reserved ? (size_t)((vram - reserved) * 0.9) : 0;
}

size_t HardwareProfile::optimal_ram_budget() const {
    // Use 80% of available RAM, leave rest for OS and page cache
    return (size_t)(memory.available_ram * 0.8);
}

float HardwareProfile::estimated_nvme_bandwidth() const {
    if (storage.empty()) return 0;
    return storage[0].seq_read_bw;
}

DType HardwareProfile::best_gpu_dtype() const {
    if (gpus.empty()) return DType::FP32;
    if (gpus[0].has_fp4)  return DType::FP4;
    if (gpus[0].has_fp8)  return DType::FP8_E4M3;
    if (gpus[0].has_bf16) return DType::BF16;
    return DType::FP16;
}

bool HardwareProfile::can_use_io_uring() const {
    if (storage.empty()) return false;
    return storage[0].supports_io_uring;
}

// ============================================================================
// Print Summary
// ============================================================================

void print_hardware_summary(const HardwareProfile& hw) {
    LOG_INFO("=== Titan Engine Hardware Profile ===");

    if (!hw.gpus.empty()) {
        for (const auto& gpu : hw.gpus) {
            LOG_INFO("GPU %d: %s", gpu.device_id, gpu.name.c_str());
            LOG_INFO("  VRAM: %.1f GB total, %.1f GB free",
                     gpu.vram_total / 1e9, gpu.vram_free / 1e9);
            LOG_INFO("  Compute: sm_%d%d, %d SMs",
                     gpu.compute_cap_major, gpu.compute_cap_minor, gpu.sm_count);
            LOG_INFO("  PCIe: Gen%d x%d (%.1f GB/s)",
                     gpu.pcie_gen, gpu.pcie_width, gpu.pcie_bandwidth);
            LOG_INFO("  Features: BF16=%d FP8=%d FP4=%d",
                     gpu.has_bf16, gpu.has_fp8, gpu.has_fp4);
        }
    } else {
        LOG_WARN("No GPUs detected — CPU-only mode");
    }

    LOG_INFO("CPU: %s", hw.cpu.model_name.c_str());
    LOG_INFO("  Cores: %d physical, %d logical, %d NUMA nodes",
             hw.cpu.physical_cores, hw.cpu.logical_cores, hw.cpu.numa_nodes);
    LOG_INFO("  Features: AVX2=%d AVX512=%d AMX_BF16=%d AMX_INT8=%d",
             hw.cpu.has_avx2, hw.cpu.has_avx512f,
             hw.cpu.has_amx_bf16, hw.cpu.has_amx_int8);

    LOG_INFO("Memory: %.1f GB total, %.1f GB available (est. %.0f GB/s bandwidth)",
             hw.memory.total_ram / 1e9, hw.memory.available_ram / 1e9,
             hw.memory.bandwidth);

    for (const auto& st : hw.storage) {
        LOG_INFO("Storage: %s (%s)", st.path.c_str(), st.is_nvme ? "NVMe" : "other");
        LOG_INFO("  Capacity: %.1f GB free / %.1f GB total",
                 st.free_bytes / 1e9, st.total_bytes / 1e9);
        LOG_INFO("  Bandwidth: %.1f GB/s seq read%s",
                 st.seq_read_bw,
                 st.raid_level == 0 ?
                     (std::string(" (RAID0 x") + std::to_string(st.raid_disks) + ")").c_str() : "");
        LOG_INFO("  io_uring=%d direct_io=%d",
                 st.supports_io_uring, st.supports_direct_io);
    }

    LOG_INFO("Optimal config: VRAM=%.1f GB, RAM=%.1f GB, dtype=%s",
             hw.optimal_vram_budget() / 1e9,
             hw.optimal_ram_budget() / 1e9,
             dtype_name(hw.best_gpu_dtype()));
}

// ============================================================================
// Execution Planning
// ============================================================================

ExecutionPlan plan_execution(const ModelConfig& model, const HardwareProfile& hw,
                             const RuntimeConfig& runtime) {
    ExecutionPlan plan;

    size_t vram_budget = runtime.vram_budget_mb > 0
        ? runtime.vram_budget_mb * 1024 * 1024
        : hw.optimal_vram_budget();
    size_t ram_budget = runtime.ram_budget_mb > 0
        ? runtime.ram_budget_mb * 1024 * 1024
        : hw.optimal_ram_budget();

    // Estimate per-layer memory requirements
    size_t attn_per_layer = (size_t)model.hidden_dim *
        (model.num_attn_heads + 2 * model.num_kv_heads) * model.head_dim;
    attn_per_layer += (size_t)model.num_attn_heads * model.head_dim * model.hidden_dim;
    // Apply quantization factor
    size_t attn_bytes = model.estimated_weight_bytes(runtime.weight_dtype) / model.total_params() * attn_per_layer;

    // KV cache per layer
    size_t kv_per_layer = 2 * (size_t)model.num_kv_heads * model.head_dim *
                          runtime.max_context_len * dtype_size(runtime.kv_cache_dtype);

    // Embedding + LM head
    size_t embed_bytes = (size_t)model.vocab_size * model.hidden_dim *
                         dtype_size(runtime.weight_dtype);
    if (dtype_size(runtime.weight_dtype) == 0) {
        embed_bytes = (size_t)model.vocab_size * model.hidden_dim / 2; // Assume INT4
    }

    size_t vram_used = 0;

    // Always put embedding in VRAM if it fits
    if (vram_used + embed_bytes < vram_budget) {
        plan.embedding_tier = MemoryTier::VRAM;
        plan.lm_head_tier = MemoryTier::VRAM;
        vram_used += embed_bytes * 2; // embed + lm_head
    } else {
        plan.embedding_tier = MemoryTier::RAM;
        plan.lm_head_tier = MemoryTier::RAM;
    }

    size_t ram_used = 0;

    // Place layers
    for (uint32_t l = 0; l < model.num_layers; l++) {
        ExecutionPlan::LayerPlacement lp;
        lp.layer_id = l;
        lp.gpu_id = hw.gpus.empty() ? -1 : 0;

        // Attention weights: prefer VRAM
        if (vram_used + attn_bytes < vram_budget) {
            lp.attention_weights = MemoryTier::VRAM;
            vram_used += attn_bytes;
        } else if (ram_used + attn_bytes < ram_budget) {
            lp.attention_weights = MemoryTier::RAM;
            ram_used += attn_bytes;
        } else {
            lp.attention_weights = MemoryTier::NVME;
        }

        // FFN/Expert weights: MoE experts go to RAM/NVMe, dense FFN to VRAM if space
        bool is_moe = false;
        if (!model.layer_configs.empty() && l < model.layer_configs.size()) {
            is_moe = model.layer_configs[l].is_moe;
        } else {
            is_moe = (model.model_type == ModelType::MOE || model.model_type == ModelType::HYBRID_MOE);
        }

        if (is_moe) {
            // MoE expert weights are too large for VRAM — always RAM/NVMe
            // Only shared experts go in VRAM
            lp.ffn_weights = MemoryTier::RAM; // Experts cached in RAM, cold on NVMe
        } else {
            size_t ffn_bytes = (size_t)3 * model.hidden_dim * model.intermediate_dim / 2;
            if (vram_used + ffn_bytes < vram_budget) {
                lp.ffn_weights = MemoryTier::VRAM;
                vram_used += ffn_bytes;
            } else {
                lp.ffn_weights = MemoryTier::RAM;
                ram_used += ffn_bytes;
            }
        }

        // KV cache: VRAM if space, else RAM
        if (vram_used + kv_per_layer < vram_budget) {
            lp.kv_cache = MemoryTier::VRAM;
            vram_used += kv_per_layer;
        } else {
            lp.kv_cache = MemoryTier::RAM;
            ram_used += kv_per_layer;
        }

        plan.layers.push_back(lp);
    }

    plan.vram_used = vram_used;
    plan.ram_used = ram_used;

    // Expert caching budget (MoE only)
    if (model.model_type == ModelType::MOE || model.model_type == ModelType::HYBRID_MOE) {
        size_t remaining_ram = ram_budget > ram_used ? ram_budget - ram_used : 0;
        plan.expert_cache_ram_mb = remaining_ram / (1024 * 1024);

        size_t remaining_vram = vram_budget > vram_used ? vram_budget - vram_used : 0;
        plan.expert_cache_vram_mb = remaining_vram / (1024 * 1024);

        plan.expert_prefetch_layers = runtime.prefetch_layers;
    }

    return plan;
}

} // namespace titan
