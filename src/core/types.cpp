#include "core/types.h"

namespace titan {

size_t ModelConfig::total_params() const {
    size_t params = 0;

    // Embedding + LM head
    params += (size_t)vocab_size * hidden_dim;  // embedding
    params += (size_t)vocab_size * hidden_dim;  // lm_head (often tied)

    for (uint32_t l = 0; l < num_layers; l++) {
        // Attention projections: Q, K, V, O
        params += (size_t)hidden_dim * num_attn_heads * head_dim;   // Q
        params += (size_t)hidden_dim * num_kv_heads * head_dim;     // K
        params += (size_t)hidden_dim * num_kv_heads * head_dim;     // V
        params += (size_t)num_attn_heads * head_dim * hidden_dim;   // O

        // Norms (small)
        params += hidden_dim * 2;  // attn_norm + ffn_norm

        bool layer_is_moe = false;
        uint32_t layer_experts = 0;
        uint32_t layer_shared = 0;
        uint32_t layer_intermediate = intermediate_dim;

        if (!layer_configs.empty() && l < layer_configs.size()) {
            layer_is_moe = layer_configs[l].is_moe;
            layer_experts = layer_configs[l].num_experts;
            layer_shared = layer_configs[l].num_shared_experts;
        } else if (model_type == ModelType::MOE || model_type == ModelType::HYBRID_MOE) {
            layer_is_moe = true;
            layer_experts = num_experts;
            layer_shared = num_shared_experts;
            layer_intermediate = moe_intermediate_dim;
        }

        if (layer_is_moe && layer_experts > 0) {
            // Routing gate
            params += (size_t)hidden_dim * layer_experts;
            // Expert MLPs: gate + up + down (SwiGLU)
            params += (size_t)layer_experts * 3 * hidden_dim * layer_intermediate;
            // Shared experts
            params += (size_t)layer_shared * 3 * hidden_dim * layer_intermediate;
        } else {
            // Dense FFN: gate + up + down (SwiGLU)
            params += (size_t)3 * hidden_dim * layer_intermediate;
        }
    }

    return params;
}

size_t ModelConfig::active_params_per_token() const {
    size_t params = 0;

    // Embedding + LM head always active
    params += (size_t)vocab_size * hidden_dim * 2;

    for (uint32_t l = 0; l < num_layers; l++) {
        // Attention always active
        params += (size_t)hidden_dim * (num_attn_heads + 2 * num_kv_heads) * head_dim;
        params += (size_t)num_attn_heads * head_dim * hidden_dim;
        params += hidden_dim * 2;

        bool layer_is_moe = false;
        uint32_t layer_k = experts_per_tok;
        uint32_t layer_shared = 0;
        uint32_t layer_intermediate = intermediate_dim;

        if (!layer_configs.empty() && l < layer_configs.size()) {
            layer_is_moe = layer_configs[l].is_moe;
            layer_k = layer_configs[l].experts_per_tok;
            layer_shared = layer_configs[l].num_shared_experts;
        } else if (model_type == ModelType::MOE || model_type == ModelType::HYBRID_MOE) {
            layer_is_moe = true;
            layer_shared = num_shared_experts;
            layer_intermediate = moe_intermediate_dim;
        }

        if (layer_is_moe) {
            // Only K active experts + shared
            params += (size_t)(layer_k + layer_shared) * 3 * hidden_dim * layer_intermediate;
            params += (size_t)hidden_dim * num_experts; // routing gate
        } else {
            params += (size_t)3 * hidden_dim * layer_intermediate;
        }
    }

    return params;
}

size_t ModelConfig::estimated_weight_bytes(DType quant) const {
    size_t total = total_params();
    switch (quant) {
        case DType::FP32:      return total * 4;
        case DType::FP16:
        case DType::BF16:      return total * 2;
        case DType::FP8_E4M3:
        case DType::FP8_E5M2:
        case DType::INT8:
        case DType::Q8_0:      return total;
        case DType::INT4:
        case DType::Q4_K:
        case DType::FP4:       return total / 2 + total / 64 * 4; // weights + scales/biases
        case DType::Q3_K:      return total * 3 / 8 + total / 64 * 4;
        case DType::INT2:
        case DType::Q2_K:      return total / 4 + total / 64 * 4;
        case DType::Q5_K:      return total * 5 / 8 + total / 64 * 4;
        case DType::Q6_K:      return total * 6 / 8 + total / 64 * 4;
    }
    return total * 2; // default FP16
}

} // namespace titan
