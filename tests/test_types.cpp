#include "core/types.h"
#include <cassert>
#include <cstdio>

int main() {
    using namespace titan;

    // Test dtype_size
    assert(dtype_size(DType::FP32) == 4);
    assert(dtype_size(DType::FP16) == 2);
    assert(dtype_size(DType::INT8) == 1);
    assert(dtype_size(DType::INT4) == 0); // Sub-byte

    // Test ModelConfig
    ModelConfig cfg;
    cfg.hidden_dim = 4096;
    cfg.num_layers = 32;
    cfg.num_attn_heads = 32;
    cfg.num_kv_heads = 8;
    cfg.head_dim = 128;
    cfg.intermediate_dim = 14336;
    cfg.vocab_size = 128256;
    cfg.model_type = ModelType::DENSE;

    size_t total = cfg.total_params();
    assert(total > 0);
    printf("Dense 8B model: %.1fB params\n", total / 1e9);

    size_t active = cfg.active_params_per_token();
    assert(active == total); // Dense model: all params active
    printf("Active per token: %.1fB\n", active / 1e9);

    // Test MoE config
    ModelConfig moe_cfg;
    moe_cfg.hidden_dim = 4096;
    moe_cfg.num_layers = 60;
    moe_cfg.num_attn_heads = 32;
    moe_cfg.num_kv_heads = 2;
    moe_cfg.head_dim = 256;
    moe_cfg.intermediate_dim = 4096;
    moe_cfg.vocab_size = 248320;
    moe_cfg.model_type = ModelType::MOE;
    moe_cfg.num_experts = 512;
    moe_cfg.experts_per_tok = 4;
    moe_cfg.num_shared_experts = 1;
    moe_cfg.moe_intermediate_dim = 1024;

    size_t moe_total = moe_cfg.total_params();
    size_t moe_active = moe_cfg.active_params_per_token();
    printf("MoE 397B model: %.1fB total, %.1fB active\n",
           moe_total / 1e9, moe_active / 1e9);
    assert(moe_active < moe_total); // MoE should have fewer active params

    // Test weight size estimation
    size_t fp16_bytes = cfg.estimated_weight_bytes(DType::FP16);
    size_t int4_bytes = cfg.estimated_weight_bytes(DType::INT4);
    printf("8B model: FP16=%.1f GB, INT4=%.1f GB\n",
           fp16_bytes / 1e9, int4_bytes / 1e9);
    assert(int4_bytes < fp16_bytes);

    printf("All tests passed!\n");
    return 0;
}
