#include "core/types.h"
#include <cassert>
#include <cstdio>

// Memory manager tests require CUDA runtime
// These are placeholder tests for CPU-only components

int main() {
    using namespace titan;

    printf("Memory manager tests (CPU-only placeholders)\n");

    // Test TensorDesc
    TensorDesc td;
    td.name = "test_tensor";
    td.dtype = DType::FP16;
    td.shape = {4096, 4096};
    td.byte_size = 4096 * 4096 * 2;
    td.tier = MemoryTier::NONE;

    assert(td.numel() == 4096 * 4096);
    assert(td.byte_size == td.numel() * 2);

    printf("All tests passed!\n");
    return 0;
}
