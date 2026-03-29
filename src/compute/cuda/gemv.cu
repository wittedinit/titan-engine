#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <mutex>

namespace titan {
namespace cuda {

// ============================================================================
// cuBLAS-based Matrix-Vector Multiply
//
// For FP32/FP16 (non-quantized) weights, cuBLAS sgemv is the fastest path.
// Used by the dense executor for attention and FFN projections.
// ============================================================================

static cublasHandle_t g_cublas_handle = nullptr;
static std::once_flag g_cublas_init_flag;

void init_cublas() {
    std::call_once(g_cublas_init_flag, []() {
        cublasCreate(&g_cublas_handle);
        // Use Tensor Cores when available
        cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH);
    });
}

void destroy_cublas() {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

// y = A @ x
// A: [rows, cols] row-major -> cuBLAS treats as col-major [cols, rows]
// x: [cols]
// y: [rows]
void gemv_fp32(
    const float* A, const float* x, float* y,
    int rows, int cols,
    cudaStream_t stream
) {
    init_cublas();
    // Always set stream — if stream is nullptr, this sets the default stream,
    // which is correct. Skipping this when stream==nullptr would leave whatever
    // stream was set by a previous call, causing execution on the wrong stream.
    cublasSetStream(g_cublas_handle, stream);

    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is col-major, our weights are row-major
    // A[rows, cols] row-major = A^T[cols, rows] col-major
    // y = A @ x in row-major = A^T @ x in col-major with CUBLAS_OP_T
    cublasSgemv(g_cublas_handle,
                CUBLAS_OP_T,  // Transpose because row-major -> col-major
                cols, rows,   // Transposed dimensions
                &alpha,
                A, cols,      // lda = cols (row stride in col-major view)
                x, 1,
                &beta,
                y, 1);
}

// Batched: y_i = A_i @ x for multiple independent matvecs
// Useful for computing Q, K, V projections in a single call
void gemv_fp32_batched(
    const float* A, const float* x, float* y,
    int rows, int cols, int batch,
    cudaStream_t stream
) {
    init_cublas();
    // Always set stream (see gemv_fp32 comment above)
    cublasSetStream(g_cublas_handle, stream);

    float alpha = 1.0f, beta = 0.0f;

    // Just call sgemm with N=1 (matrix x vector = matrix x 1-col matrix)
    // For batch: treat as [rows*batch, cols] @ [cols, 1] -> [rows*batch, 1]
    // But that only works if all projections share the same input.
    // Since Q, K, V all take the same hidden state as input, this works!
    cublasSgemv(g_cublas_handle,
                CUBLAS_OP_T,
                cols, rows * batch,
                &alpha,
                A, cols,
                x, 1,
                &beta,
                y, 1);
}

// BF16 matrix × FP32 vector → FP32 output
// A: [rows, cols] row-major BF16  |  x: [cols] FP32  |  y: [rows] FP32
// Uses cublasGemmEx: same CUBLAS_OP_T trick as gemv_fp32.
void gemv_bf16_to_fp32(
    const void* A, const float* x, float* y,
    int rows, int cols,
    cudaStream_t stream
) {
    init_cublas();
    cublasSetStream(g_cublas_handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    // Mirror gemv_fp32 layout: A[rows,cols] row-major = [cols,rows] col-major
    // CUBLAS_OP_T transposes the stored [cols,rows] back to [rows,cols]
    cublasGemmEx(g_cublas_handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 rows, 1, cols,
                 &alpha,
                 A, CUDA_R_16BF, cols,
                 x, CUDA_R_32F, cols,
                 &beta,
                 y, CUDA_R_32F, rows,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// BF16 embedding lookup: copy and convert row token_id to FP32
__global__ void bf16_embed_kernel(
    float* __restrict__ out, const uint16_t* __restrict__ emb,
    int token_id, int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    uint32_t v = (uint32_t)emb[(size_t)token_id * dim + i] << 16;
    float f;
    __builtin_memcpy(&f, &v, 4);
    out[i] = f;
}

void embed_token_bf16(float* out, const void* emb, int token_id, int dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    bf16_embed_kernel<<<blocks, threads, 0, stream>>>(out, (const uint16_t*)emb, token_id, dim);
    CUDA_CHECK_LAUNCH();
}

// Add two vectors: y = a + b (element-wise)
__global__ void vector_add_kernel(float* y, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

void vector_add(float* y, const float* a, const float* b, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads, 0, stream>>>(y, a, b, n);
    CUDA_CHECK_LAUNCH();
}

// Copy: dst = src
__global__ void vector_copy_kernel(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

void vector_copy(float* dst, const float* src, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_copy_kernel<<<blocks, threads, 0, stream>>>(dst, src, n);
    CUDA_CHECK_LAUNCH();
}

} // namespace cuda
} // namespace titan
