#include "core/types.h"
#include "core/logging.h"
#include <cstring>
#include <cmath>

#ifdef TITAN_HAS_AVX512
#include <immintrin.h>
#endif

namespace titan {
namespace cpu {

// ============================================================================
// AVX-512 Matrix-Vector Multiply (for CPU expert execution)
//
// When experts are cached in RAM, we can execute them on CPU using AVX-512
// instead of transferring to GPU. This is beneficial when:
// 1. GPU is busy with attention computation
// 2. Expert weights are already in RAM (avoiding PCIe transfer)
// 3. CPU has high memory bandwidth (EPYC 12-ch DDR5 = ~460 GB/s)
// ============================================================================

#ifdef TITAN_HAS_AVX512

void matvec_fp32_avx512(
    const float* __restrict__ weight,  // [rows, cols]
    const float* __restrict__ input,   // [cols]
    float*       __restrict__ output,  // [rows]
    int rows, int cols
) {
    for (int row = 0; row < rows; row++) {
        __m512 acc = _mm512_setzero_ps();
        const float* w = weight + (size_t)row * cols;

        int col = 0;
        for (; col + 16 <= cols; col += 16) {
            __m512 wv = _mm512_loadu_ps(w + col);
            __m512 xv = _mm512_loadu_ps(input + col);
            acc = _mm512_fmadd_ps(wv, xv, acc);
        }

        float sum = _mm512_reduce_add_ps(acc);

        // Handle remainder
        for (; col < cols; col++) {
            sum += w[col] * input[col];
        }

        output[row] = sum;
    }
}

// INT4 dequant matvec on CPU (AVX-512)
void dequant_matvec_int4_avx512(
    const uint32_t* __restrict__ weights,  // Packed INT4 [rows, cols/8]
    const uint16_t* __restrict__ scales,   // FP16 scales [rows, cols/group_size]
    const uint16_t* __restrict__ biases,   // FP16 biases [rows, cols/group_size]
    const float*    __restrict__ input,    // [cols]
    float*          __restrict__ output,   // [rows]
    int rows, int cols, int group_size
) {
    int packed_per_row = cols / 8;
    int groups_per_row = cols / group_size;

    for (int row = 0; row < rows; row++) {
        float acc = 0.0f;
        const uint32_t* rw = weights + (size_t)row * packed_per_row;
        const uint16_t* rs = scales + (size_t)row * groups_per_row;
        const uint16_t* rb = biases + (size_t)row * groups_per_row;

        for (int p = 0; p < packed_per_row; p++) {
            uint32_t packed = rw[p];
            int base_col = p * 8;
            int group = base_col / group_size;

            // Convert FP16 scale/bias to FP32
            // Use _cvtsh_ss if available, otherwise manual conversion
            float scale, bias;
            {
                __m128i s16 = _mm_set1_epi16(rs[group]);
                __m128 s32 = _mm_cvtph_ps(s16);
                scale = _mm_cvtss_f32(s32);
            }
            {
                __m128i b16 = _mm_set1_epi16(rb[group]);
                __m128 b32 = _mm_cvtph_ps(b16);
                bias = _mm_cvtss_f32(b32);
            }

            for (int n = 0; n < 8; n++) {
                int col = base_col + n;
                if (col >= cols) break;
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                float x = input[col];
                acc += ((float)nibble * scale + bias) * x;
            }
        }
        output[row] = acc;
    }
}

#else // No AVX-512

void matvec_fp32_avx512(
    const float* weight, const float* input, float* output,
    int rows, int cols
) {
    // Scalar fallback
    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        const float* w = weight + (size_t)row * cols;
        for (int col = 0; col < cols; col++) {
            sum += w[col] * input[col];
        }
        output[row] = sum;
    }
}

void dequant_matvec_int4_avx512(
    const uint32_t* weights, const uint16_t* scales, const uint16_t* biases,
    const float* input, float* output,
    int rows, int cols, int group_size
) {
    // Scalar fallback
    int packed_per_row = cols / 8;
    int groups_per_row = cols / group_size;

    for (int row = 0; row < rows; row++) {
        float acc = 0.0f;
        const uint32_t* rw = weights + (size_t)row * packed_per_row;

        for (int p = 0; p < packed_per_row; p++) {
            uint32_t packed = rw[p];
            int base_col = p * 8;
            int group = base_col / group_size;

            // Manual FP16 to FP32 (simplified — assumes IEEE 754 half)
            auto fp16_to_fp32 = [](uint16_t h) -> float {
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                if (exp == 0) return sign ? -0.0f : 0.0f;
                if (exp == 31) return sign ? -INFINITY : INFINITY;
                float result = ldexpf((float)(mant | 0x400) / 1024.0f, (int)exp - 15);
                return sign ? -result : result;
            };

            float scale = fp16_to_fp32(scales[row * groups_per_row + group]);
            float bias  = fp16_to_fp32(biases[row * groups_per_row + group]);

            for (int n = 0; n < 8 && base_col + n < cols; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                acc += ((float)nibble * scale + bias) * input[base_col + n];
            }
        }
        output[row] = acc;
    }
}

#endif

// ============================================================================
// SwiGLU on CPU
// ============================================================================

void swiglu_cpu(float* output, const float* gate, const float* up, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        output[i] = (g * sigmoid_g) * up[i];
    }
}

// ============================================================================
// Expert Forward on CPU (INT4 quantized)
// ============================================================================

void expert_forward_int4_cpu(
    const float*    input,       // [hidden_dim]
    const uint32_t* gate_w,     // Packed INT4
    const uint16_t* gate_s,
    const uint16_t* gate_b,
    const uint32_t* up_w,
    const uint16_t* up_s,
    const uint16_t* up_b,
    const uint32_t* down_w,
    const uint16_t* down_s,
    const uint16_t* down_b,
    float*          output,      // [hidden_dim]
    float*          scratch,     // [inter_dim * 2]
    int hidden_dim, int inter_dim, int group_size
) {
    float* gate_out = scratch;
    float* up_out = scratch + inter_dim;

    // Gate projection
    dequant_matvec_int4_avx512(gate_w, gate_s, gate_b, input, gate_out,
                                inter_dim, hidden_dim, group_size);

    // Up projection
    dequant_matvec_int4_avx512(up_w, up_s, up_b, input, up_out,
                                inter_dim, hidden_dim, group_size);

    // SwiGLU activation
    swiglu_cpu(gate_out, gate_out, up_out, inter_dim);

    // Down projection
    dequant_matvec_int4_avx512(down_w, down_s, down_b, gate_out, output,
                                hidden_dim, inter_dim, group_size);
}

} // namespace cpu
} // namespace titan
