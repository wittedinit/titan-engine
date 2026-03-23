# Titan Engine

Universal LLM inference engine targeting models up to 1T parameters on consumer/prosumer hardware.

## Architecture

- **Language**: C++17 with CUDA kernels
- **GPU**: NVIDIA CUDA (RTX 5090 primary target)
- **Build**: CMake 3.24+, CUDA 12.0+
- **I/O**: io_uring for async NVMe reads, pread fallback

## Key Design Principles

1. **3-tier memory**: VRAM (hot) → RAM (warm) → NVMe (cold)
2. **MoE-first**: Optimized for sparse MoE models where only a fraction of params are active per token
3. **Discrete GPU advantage**: NVMe reads don't compete with GPU compute (unlike Apple Silicon unified memory)
4. **Trust RAM cache**: Unlike flash-moe which trusts OS page cache, we explicitly manage RAM↔VRAM transfers
5. **Fused kernels**: Minimize memory round-trips (combine+residual+norm in one kernel)

## Directory Layout

- `src/core/` — Types, config, hardware detection, logging
- `src/memory/` — 3-tier memory manager (VRAM pool, RAM pool, NVMe pool with io_uring)
- `src/compute/cuda/` — CUDA kernels (dequant, attention, MoE, norms, activation, sampling)
- `src/compute/cpu/` — AVX-512 CPU kernels for expert execution
- `src/model/` — Model loader (safetensors), tokenizer, architecture executors
- `src/inference/` — Inference engine, scheduler, speculative decoding, KV cache
- `src/api/` — CLI chat, HTTP server, Python bindings
- `tools/` — Model conversion and quantization utilities
- `tests/` — Unit tests (kernel correctness vs CPU reference)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Conventions

- All CUDA kernels have CPU reference implementations in tests for correctness verification
- Quantization: group_size=64 for INT4, scale+bias per group (affine quantization)
- Expert layout follows flash-moe format: [gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b]
- Memory alignment: 256 bytes for VRAM, 64 bytes for RAM (AVX-512)
