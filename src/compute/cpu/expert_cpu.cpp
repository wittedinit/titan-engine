#include "core/types.h"
#include "core/logging.h"
#include <thread>
#include <vector>
#include <functional>

namespace titan {
namespace cpu {

// Forward declarations from matmul_avx.cpp
void expert_forward_int4_cpu(
    const float* input, const uint32_t* gate_w, const uint16_t* gate_s,
    const uint16_t* gate_b, const uint32_t* up_w, const uint16_t* up_s,
    const uint16_t* up_b, const uint32_t* down_w, const uint16_t* down_s,
    const uint16_t* down_b, float* output, float* scratch,
    int hidden_dim, int inter_dim, int group_size
);

// ============================================================================
// Parallel Expert Execution on CPU
//
// When experts are cached in RAM, execute them in parallel across CPU cores.
// This is the KTransformers approach: attention on GPU, experts on CPU.
//
// For EPYC 64-core: can execute 16 experts simultaneously with 4 threads each,
// or all experts serially with maximum parallelism per expert.
// ============================================================================

struct ExpertTask {
    const float*    input;
    const void*     expert_data;    // Pointer to packed expert weights in RAM
    float*          output;
    float*          scratch;
    int             hidden_dim;
    int             inter_dim;
    int             group_size;
};

void execute_experts_parallel(
    const std::vector<ExpertTask>& tasks,
    int num_threads_per_expert
) {
    if (tasks.empty()) return;

    std::vector<std::thread> threads;
    threads.reserve(tasks.size());

    for (const auto& task : tasks) {
        threads.emplace_back([&task]() {
            // Parse the packed expert data layout
            // Layout (from flash-moe):
            //   gate_proj weights:  [inter_dim, hidden_dim/8] uint32
            //   gate_proj scales:   [inter_dim, hidden_dim/group_size] fp16
            //   gate_proj biases:   [inter_dim, hidden_dim/group_size] fp16
            //   up_proj weights:    [inter_dim, hidden_dim/8] uint32
            //   up_proj scales:     [inter_dim, hidden_dim/group_size] fp16
            //   up_proj biases:     [inter_dim, hidden_dim/group_size] fp16
            //   down_proj weights:  [hidden_dim, inter_dim/8] uint32
            //   down_proj scales:   [hidden_dim, inter_dim/group_size] fp16
            //   down_proj biases:   [hidden_dim, inter_dim/group_size] fp16

            int hd = task.hidden_dim;
            int id = task.inter_dim;
            int gs = task.group_size;

            size_t gate_w_size = (size_t)id * (hd / 8) * 4;
            size_t gate_s_size = (size_t)id * (hd / gs) * 2;
            size_t gate_b_size = gate_s_size;
            size_t up_w_size   = gate_w_size;
            size_t up_s_size   = gate_s_size;
            size_t up_b_size   = gate_s_size;
            size_t down_w_size = (size_t)hd * (id / 8) * 4;
            size_t down_s_size = (size_t)hd * (id / gs) * 2;
            // down_b_size = down_s_size;

            const uint8_t* p = (const uint8_t*)task.expert_data;

            const uint32_t* gate_w = (const uint32_t*)p; p += gate_w_size;
            const uint16_t* gate_s = (const uint16_t*)p; p += gate_s_size;
            const uint16_t* gate_b = (const uint16_t*)p; p += gate_b_size;
            const uint32_t* up_w   = (const uint32_t*)p; p += up_w_size;
            const uint16_t* up_s   = (const uint16_t*)p; p += up_s_size;
            const uint16_t* up_b   = (const uint16_t*)p; p += up_b_size;
            const uint32_t* down_w = (const uint32_t*)p; p += down_w_size;
            const uint16_t* down_s = (const uint16_t*)p; p += down_s_size;
            const uint16_t* down_b = (const uint16_t*)p;

            expert_forward_int4_cpu(
                task.input,
                gate_w, gate_s, gate_b,
                up_w, up_s, up_b,
                down_w, down_s, down_b,
                task.output, task.scratch,
                hd, id, gs
            );
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

} // namespace cpu
} // namespace titan
