#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cassert>

// Test CUDA kernel correctness against CPU reference implementations

namespace titan { namespace cuda {
    void rmsnorm(float* output, const float* input, const float* weight,
                 int dim, float eps, cudaStream_t stream);
    void swiglu(float* output, const float* gate, const float* up,
                int dim, cudaStream_t stream);
}}

// CPU reference: RMSNorm
void rmsnorm_ref(float* output, const float* input, const float* weight,
                 int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += input[i] * input[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) output[i] = input[i] * inv_rms * weight[i];
}

// CPU reference: SwiGLU
void swiglu_ref(float* output, const float* gate, const float* up, int dim) {
    for (int i = 0; i < dim; i++) {
        float sigmoid_g = 1.0f / (1.0f + expf(-gate[i]));
        output[i] = (gate[i] * sigmoid_g) * up[i];
    }
}

bool test_rmsnorm() {
    const int dim = 4096;
    const float eps = 1e-6f;

    // Allocate host memory
    float* h_input = new float[dim];
    float* h_weight = new float[dim];
    float* h_output_ref = new float[dim];
    float* h_output_gpu = new float[dim];

    // Initialize with random-ish values
    for (int i = 0; i < dim; i++) {
        h_input[i] = sinf(i * 0.01f) * 2.0f;
        h_weight[i] = 1.0f + cosf(i * 0.007f) * 0.1f;
    }

    // CPU reference
    rmsnorm_ref(h_output_ref, h_input, h_weight, dim, eps);

    // GPU
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, dim * sizeof(float));
    cudaMalloc(&d_weight, dim * sizeof(float));
    cudaMalloc(&d_output, dim * sizeof(float));
    cudaMemcpy(d_input, h_input, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, dim * sizeof(float), cudaMemcpyHostToDevice);

    titan::cuda::rmsnorm(d_output, d_input, d_weight, dim, eps, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_gpu, d_output, dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float max_error = 0.0f;
    for (int i = 0; i < dim; i++) {
        float err = fabsf(h_output_ref[i] - h_output_gpu[i]);
        if (err > max_error) max_error = err;
    }

    printf("RMSNorm: max error = %e %s\n", max_error, max_error < 1e-4 ? "PASS" : "FAIL");

    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_output);
    delete[] h_input; delete[] h_weight; delete[] h_output_ref; delete[] h_output_gpu;

    return max_error < 1e-4;
}

bool test_swiglu() {
    const int dim = 4096;

    float* h_gate = new float[dim];
    float* h_up = new float[dim];
    float* h_output_ref = new float[dim];
    float* h_output_gpu = new float[dim];

    for (int i = 0; i < dim; i++) {
        h_gate[i] = sinf(i * 0.013f);
        h_up[i] = cosf(i * 0.017f);
    }

    swiglu_ref(h_output_ref, h_gate, h_up, dim);

    float *d_gate, *d_up, *d_output;
    cudaMalloc(&d_gate, dim * sizeof(float));
    cudaMalloc(&d_up, dim * sizeof(float));
    cudaMalloc(&d_output, dim * sizeof(float));
    cudaMemcpy(d_gate, h_gate, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up, dim * sizeof(float), cudaMemcpyHostToDevice);

    titan::cuda::swiglu(d_output, d_gate, d_up, dim, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_gpu, d_output, dim * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < dim; i++) {
        float err = fabsf(h_output_ref[i] - h_output_gpu[i]);
        if (err > max_error) max_error = err;
    }

    printf("SwiGLU: max error = %e %s\n", max_error, max_error < 1e-4 ? "PASS" : "FAIL");

    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_output);
    delete[] h_gate; delete[] h_up; delete[] h_output_ref; delete[] h_output_gpu;

    return max_error < 1e-4;
}

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found. Skipping GPU kernel tests.\n");
        return 0;
    }

    printf("Testing CUDA kernels on device 0...\n");
    bool pass = true;
    pass &= test_rmsnorm();
    pass &= test_swiglu();

    printf("\n%s\n", pass ? "All kernel tests PASSED" : "Some tests FAILED");
    return pass ? 0 : 1;
}
