#include "core/hardware.h"
#include "core/logging.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/mount.h>
#include <mach/mach.h>
#endif

#include <unistd.h>

#ifdef __x86_64__
#ifdef __GNUC__
#include <cpuid.h>
#endif
#endif

// CUDA headers MUST be included OUTSIDE of namespace titan
// to avoid pulling CUDA runtime symbols into our namespace
#include <cuda_runtime.h>

namespace titan {

// ============================================================================
// GPU Detection (CUDA)
// ============================================================================

std::vector<GpuInfo> detect_gpus() {
    std::vector<GpuInfo> gpus;

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        LOG_WARN("No CUDA GPUs detected");
        return gpus;
    }

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaSetDevice(i);
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;

        GpuInfo gpu;
        gpu.device_id = i;
        gpu.name = prop.name;
        gpu.vram_total = prop.totalGlobalMem;
        gpu.compute_cap_major = prop.major;
        gpu.compute_cap_minor = prop.minor;
        gpu.sm_count = prop.multiProcessorCount;
        gpu.l2_cache_size = prop.l2CacheSize;

        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            gpu.vram_free = free_mem;
        }

        float cc = prop.major + prop.minor * 0.1f;
        gpu.has_bf16     = (cc >= 8.0f);
        gpu.has_int8_tc  = (cc >= 7.5f);
        gpu.has_fp8      = (cc >= 8.9f);
        gpu.has_fp4      = (cc >= 10.0f);

        if (cc >= 10.0f) {
            gpu.pcie_gen = 5; gpu.pcie_width = 16; gpu.pcie_bandwidth = 63.0f;
        } else if (cc >= 8.0f) {
            gpu.pcie_gen = 4; gpu.pcie_width = 16; gpu.pcie_bandwidth = 31.5f;
        } else {
            gpu.pcie_gen = 3; gpu.pcie_width = 16; gpu.pcie_bandwidth = 15.75f;
        }

        gpus.push_back(gpu);
    }

    return gpus;
}

// ============================================================================
// CPU Detection (cross-platform)
// ============================================================================

CpuInfo detect_cpu() {
    CpuInfo info;

#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos) info.model_name = line.substr(pos + 2);
            break;
        }
    }
#elif defined(__APPLE__)
    char buf[256] = {};
    size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) {
        info.model_name = buf;
    }
#endif

    info.logical_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    info.physical_cores = std::max(1, info.logical_cores / 2);

#ifdef __linux__
    info.numa_nodes = 1;
    std::ifstream numa_file("/sys/devices/system/node/online");
    if (numa_file.good()) {
        std::string s;
        std::getline(numa_file, s);
        auto dash = s.find('-');
        if (dash != std::string::npos)
            info.numa_nodes = std::stoi(s.substr(dash + 1)) + 1;
    }
#else
    info.numa_nodes = 1;
#endif

#if defined(__x86_64__) && defined(__GNUC__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        info.has_avx2        = (ebx >> 5) & 1;
        info.has_avx512f     = (ebx >> 16) & 1;
        info.has_avx512_vnni = (ecx >> 11) & 1;
        info.has_amx_bf16    = (edx >> 22) & 1;
        info.has_amx_int8    = (edx >> 24) & 1;
    }
#endif

    return info;
}

// ============================================================================
// Memory Detection
// ============================================================================

MemoryInfo detect_memory() {
    MemoryInfo info;

#ifdef __linux__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        info.total_ram = si.totalram * si.mem_unit;
        info.available_ram = si.freeram * si.mem_unit;
        info.swap_total = si.totalswap * si.mem_unit;
    }
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
#elif defined(__APPLE__)
    int64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
        info.total_ram = (size_t)mem;
    }
    // Available RAM via mach
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vmstat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vmstat, &count) == KERN_SUCCESS) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        info.available_ram = ((size_t)vmstat.free_count + vmstat.inactive_count) * page_size;
    }
#endif

    // Estimate bandwidth heuristic
    if (info.total_ram > 256ULL * 1024 * 1024 * 1024) {
        info.num_channels = 12; info.bandwidth = 460.0f;
    } else if (info.total_ram > 64ULL * 1024 * 1024 * 1024) {
        info.num_channels = 4; info.bandwidth = 153.0f;
    } else {
        info.num_channels = 2; info.bandwidth = 89.6f;
    }

    return info;
}

// ============================================================================
// Storage Detection
// ============================================================================

StorageInfo detect_storage(const std::string& path) {
    StorageInfo info;
    info.path = path;

#ifdef __linux__
    struct statvfs stat;
    if (statvfs(path.c_str(), &stat) == 0) {
        info.total_bytes = stat.f_blocks * stat.f_frsize;
        info.free_bytes = stat.f_bfree * stat.f_frsize;
    }

    // Detect NVMe from /proc/mounts
    std::ifstream mounts("/proc/mounts");
    std::string line;
    while (std::getline(mounts, line)) {
        if (line.find(path) != std::string::npos || path == "/") {
            std::istringstream iss(line);
            iss >> info.device >> info.path >> info.fs_type;
            break;
        }
    }
    info.is_nvme = (info.device.find("nvme") != std::string::npos);

    // io_uring support
    info.supports_io_uring = false;
#ifdef TITAN_HAS_IO_URING
    info.supports_io_uring = true;
#endif
    info.supports_direct_io = true;

    // Check RAID
    info.seq_read_bw = info.is_nvme ? 7.0f : 0.5f;
    std::ifstream mdstat("/proc/mdstat");
    if (mdstat.good()) {
        std::string md_line;
        while (std::getline(mdstat, md_line)) {
            if (md_line.find("raid0") != std::string::npos) {
                int drives = 0;
                for (size_t i = 0; i + 3 < md_line.length(); i++) {
                    if (md_line.substr(i, 4) == "nvme") drives++;
                }
                if (drives > 1) {
                    info.raid_level = 0;
                    info.raid_disks = drives;
                    info.seq_read_bw = 7.0f * drives;
                }
                break;
            }
        }
    }
#elif defined(__APPLE__)
    struct statfs stat;
    if (statfs(path.c_str(), &stat) == 0) {
        info.total_bytes = (size_t)stat.f_blocks * stat.f_bsize;
        info.free_bytes = (size_t)stat.f_bfree * stat.f_bsize;
        info.device = stat.f_mntfromname;
        info.fs_type = stat.f_fstypename;
    }
    info.is_nvme = true; // Modern Macs all use NVMe
    info.supports_io_uring = false;
    info.supports_direct_io = false; // macOS doesn't support O_DIRECT
    info.seq_read_bw = 7.0f; // Approximate for Apple NVMe
#endif

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
// HardwareProfile Methods
// ============================================================================

size_t HardwareProfile::optimal_vram_budget() const {
    if (gpus.empty()) return 0;
    size_t vram = gpus[0].vram_free;
    size_t reserved = 512ULL * 1024 * 1024;
    return vram > reserved ? (size_t)((vram - reserved) * 0.9) : 0;
}

size_t HardwareProfile::optimal_ram_budget() const {
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
            LOG_INFO("  Features: BF16=%d FP8=%d FP4=%d",
                     gpu.has_bf16, gpu.has_fp8, gpu.has_fp4);
        }
    } else {
        LOG_WARN("No GPUs detected — CPU-only mode");
    }

    LOG_INFO("CPU: %s", hw.cpu.model_name.c_str());
    LOG_INFO("  Cores: %d physical, %d logical",
             hw.cpu.physical_cores, hw.cpu.logical_cores);
    LOG_INFO("  Features: AVX2=%d AVX512=%d",
             hw.cpu.has_avx2, hw.cpu.has_avx512f);

    LOG_INFO("Memory: %.1f GB total, %.1f GB available",
             hw.memory.total_ram / 1e9, hw.memory.available_ram / 1e9);

    for (const auto& st : hw.storage) {
        LOG_INFO("Storage: %s (%s, %.1f GB free, %.1f GB/s read)",
                 st.path.c_str(), st.is_nvme ? "NVMe" : "other",
                 st.free_bytes / 1e9, st.seq_read_bw);
    }
}

// ============================================================================
// Execution Planning
// ============================================================================

ExecutionPlan plan_execution(const ModelConfig& model, const HardwareProfile& hw,
                             const RuntimeConfig& runtime) {
    ExecutionPlan plan;

    size_t vram_budget = runtime.vram_budget_mb > 0
        ? runtime.vram_budget_mb * 1024ULL * 1024
        : hw.optimal_vram_budget();
    size_t ram_budget = runtime.ram_budget_mb > 0
        ? runtime.ram_budget_mb * 1024ULL * 1024
        : hw.optimal_ram_budget();

    // Simplified planning: try to fit everything in VRAM, overflow to RAM, then NVMe
    size_t vram_used = 0;
    size_t ram_used = 0;

    // Estimate per-layer sizes
    size_t bytes_per_param = 2; // FP16 default
    if (runtime.weight_dtype == DType::INT4 || runtime.weight_dtype == DType::Q4_K)
        bytes_per_param = 1; // ~0.5 bytes but with scales

    size_t embed_bytes = (size_t)model.vocab_size * model.hidden_dim * bytes_per_param;
    if (vram_used + embed_bytes * 2 < vram_budget) {
        plan.embedding_tier = MemoryTier::VRAM;
        plan.lm_head_tier = MemoryTier::VRAM;
        vram_used += embed_bytes * 2;
    } else {
        plan.embedding_tier = MemoryTier::RAM;
        plan.lm_head_tier = MemoryTier::RAM;
        ram_used += embed_bytes * 2;
    }

    for (uint32_t l = 0; l < model.num_layers; l++) {
        ExecutionPlan::LayerPlacement lp;
        lp.layer_id = l;
        lp.gpu_id = hw.gpus.empty() ? -1 : 0;

        size_t attn_bytes = (size_t)(model.num_attn_heads + 2 * model.num_kv_heads) *
                            model.head_dim * model.hidden_dim * bytes_per_param;
        size_t ffn_bytes = (size_t)3 * model.hidden_dim * model.intermediate_dim * bytes_per_param;

        if (vram_used + attn_bytes + ffn_bytes < vram_budget) {
            lp.attention_weights = MemoryTier::VRAM;
            lp.ffn_weights = MemoryTier::VRAM;
            lp.kv_cache = MemoryTier::VRAM;
            vram_used += attn_bytes + ffn_bytes;
        } else if (ram_used + attn_bytes + ffn_bytes < ram_budget) {
            lp.attention_weights = MemoryTier::RAM;
            lp.ffn_weights = MemoryTier::RAM;
            lp.kv_cache = MemoryTier::VRAM;
            ram_used += attn_bytes + ffn_bytes;
        } else {
            lp.attention_weights = MemoryTier::NVME;
            lp.ffn_weights = MemoryTier::NVME;
            lp.kv_cache = MemoryTier::RAM;
        }

        plan.layers.push_back(lp);
    }

    plan.vram_used = vram_used;
    plan.ram_used = ram_used;

    return plan;
}

} // namespace titan
