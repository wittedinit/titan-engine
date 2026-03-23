# Titan Engine

Universal LLM inference engine — run models up to 1T parameters on consumer hardware.

## Build & Run

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./titan -m /path/to/model -q q4_k
```

Supports HuggingFace safetensors directories and GGUF files (auto-detected).

## Architecture

- **Language**: C++17 with CUDA kernels, cuBLAS for FP32 matvec
- **GPU**: NVIDIA CUDA (sm_80 Ampere, sm_89 Ada, sm_90a Blackwell)
- **Build**: CMake 3.24+, CUDA 12.0+
- **I/O**: io_uring for async NVMe reads, pread fallback on macOS/older Linux

## Key Design Principles

1. **3-tier memory**: VRAM (hot) → RAM (warm, LRU expert cache) → NVMe (cold, io_uring)
2. **MoE-first**: Optimized for sparse MoE models where only K of N experts are active per token
3. **Discrete GPU advantage**: NVMe reads don't compete with GPU compute (unlike unified memory)
4. **Pre-allocated buffers**: MoE executor uses double-buffered expert staging — zero per-token mallocs
5. **Dual compute path**: cuBLAS sgemv for FP32, custom dequant kernels for INT4/INT2
6. **Auto-detection**: Model format (safetensors vs GGUF), model type (dense vs MoE), hardware capabilities

## Directory Layout

- `src/core/` — Types, config (HF config.json parser), hardware detection, logging
- `src/memory/` — 3-tier memory manager (VRAM pool, pinned RAM pool, NVMe io_uring pool)
- `src/compute/cuda/` — 8 CUDA kernel files (dequant, gemv, attention, MoE, norms, activation, sampling, sparse)
- `src/compute/cpu/` — AVX-512 CPU kernels for RAM-resident expert execution
- `src/model/` — Safetensors loader, GGUF loader, BPE tokenizer, dense executor, MoE executor, sparsity system
- `src/inference/` — Engine orchestrator (prefill + decode), KV cache, scheduler stubs
- `src/api/` — Interactive CLI chat with colored streaming output
- `tools/` — moeify.py (dense→MoE conversion + sparsity profiling), convert.py, benchmark.py
- `tests/` — Kernel correctness tests (CUDA vs CPU reference), type tests

## Weight Loading Flow

1. `ModelLoader::load()` parses safetensors headers (or `GGUFLoader::load()` for GGUF)
2. `read_tensor_gpu()` reads from disk via pread → pinned staging buffer → cudaMemcpy to GPU
3. FP16/BF16 weights auto-converted to FP32 during load (IEEE 754 half→float)
4. Dense executor loads by tensor name: `model.layers.N.self_attn.q_proj.weight`

## Forward Pass Dispatch

`matvec_dispatch()` in dense.cpp routes based on `weight_format_`:
- `DType::FP32` → `cuda::gemv_fp32()` (cuBLAS sgemv)
- `DType::INT4` / `DType::Q4_K` → `cuda::dequant_matvec_int4()` (custom kernel with FMA trick)

## Conventions

- Quantization: group_size=64 for INT4, scale+bias per group (affine quantization)
- Expert layout: [gate_proj | up_proj | down_proj] contiguous in memory
- Memory alignment: 256 bytes for VRAM, 64 bytes for RAM (AVX-512)
- RMSNorm epsilon: 1e-5 (matches Llama default)
- All CUDA kernels use shared memory + warp-level reduction
