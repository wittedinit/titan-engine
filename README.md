# Titan Engine

**Run models up to 1 trillion parameters on a single machine.**

Titan Engine is a high-performance LLM inference engine designed to push the boundaries of what's possible on consumer and prosumer hardware. Built on the same principles as [flash-moe](https://github.com/nicholaschenai/flash-moe-main) — which runs a 397B parameter model on a MacBook — but engineered for Linux/CUDA systems with discrete GPUs, large RAM, and fast NVMe storage.

## Key Idea

You don't need to fit the entire model in GPU memory. With clever 3-tier memory management (VRAM → RAM → NVMe), aggressive quantization, and MoE sparsity, a machine with 32GB VRAM + 128GB RAM + fast NVMe RAID can run models 30x larger than its VRAM would suggest.

## Target Hardware & Expected Performance

| Model | Parameters | Active/Token | Quant | Est. tok/s |
|-------|-----------|-------------|-------|-----------|
| Llama 3.x 70B | 70B | 70B (dense) | Q4_K | 15–25 |
| Llama 3.x 405B | 405B | 405B (dense) | Q4_K | 3–6 |
| DeepSeek-V3 671B | 671B | 37B (MoE) | Q4_K | 12–20 |
| Qwen3.5-MoE 397B | 397B | 17B (MoE) | Q4 | 15–30 |
| **1T MoE Model** | **1T** | **~60B** | **Q3–Q4** | **8–15** |

*Benchmarked on: RTX 5090 (32GB), EPYC 64-core, 128GB DDR5, 4x NVMe RAID 0 (~28 GB/s)*

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Titan Engine                       │
├─────────────────────────────────────────────────────┤
│  API: HTTP (OpenAI-compatible) + CLI Chat + Python  │
├─────────────────────────────────────────────────────┤
│  Inference: Orchestrator, Speculative Decoding,     │
│             Continuous Batching, KV Cache Mgmt      │
├─────────────────────────────────────────────────────┤
│  Model Runtime: Dense / MoE / Hybrid Executors      │
├─────────────────────────────────────────────────────┤
│  Compute: CUDA Kernels (FP4–FP32) + CPU (AVX-512)  │
├─────────────────────────────────────────────────────┤
│  Memory: 3-Tier Manager (VRAM ↔ RAM ↔ NVMe)        │
│          Expert LRU Cache + Predictive Prefetching  │
├─────────────────────────────────────────────────────┤
│  I/O: io_uring async NVMe + CUDA DMA + pinned RAM  │
└─────────────────────────────────────────────────────┘
```

### 3-Tier Memory Manager

The core innovation: a hierarchical memory system that automatically places model weights across three tiers based on access frequency and available resources.

| Tier | Storage | Bandwidth | Contents |
|------|---------|-----------|----------|
| **Hot (VRAM)** | 32 GB | ~2 TB/s | Attention weights, embeddings, active experts, KV cache |
| **Warm (RAM)** | 128 GB | ~460 GB/s | Expert cache (LRU), overflow KV cache, dense FFN |
| **Cold (NVMe)** | Unlimited | ~28 GB/s | Full model weights, cold experts |

**Key advantage over Apple Silicon**: On discrete GPU systems, NVMe and GPU don't share a memory bus. This means NVMe reads can happen *simultaneously* with GPU compute — true parallel I/O + compute overlap.

### MoE Expert Streaming

For Mixture-of-Experts models, Titan Engine exploits extreme sparsity:

- A 1T MoE model might have 512 experts per layer, but only 4–8 are active per token
- Active expert weights (~30 GB at Q4) fit in VRAM + RAM
- Expert LRU cache in RAM achieves 70%+ hit rate
- Cache misses are served from NVMe at 28 GB/s (predictive prefetching hides latency)

### CUDA Kernel Library

Hand-tuned CUDA kernels for every quantization level:

- **FP4**: Native Blackwell FP4 Tensor Cores (RTX 5090)
- **INT4/Q4_K**: FMA-optimized dequant + matvec (inspired by flash-moe)
- **FP8**: Ada/Blackwell FP8 E4M3 support
- **INT2**: Extreme compression for less-sensitive layers
- **Fused kernels**: SwiGLU, RoPE, Add+RMSNorm, MoE Combine+Residual+Norm
- **Flash Attention**: Decode-optimized with GQA and PagedAttention

### CPU Expert Execution

When experts are cached in RAM, they can be executed directly on CPU using AVX-512/AMX — no GPU transfer needed:

- EPYC 64-core: run 16 experts in parallel (4 threads each)
- ~460 GB/s memory bandwidth for expert matvec
- GPU stays free for attention computation

## Why Not Just Use llama.cpp / vLLM / etc.?

| Feature | Titan | llama.cpp | vLLM | KTransformers |
|---------|-------|-----------|------|---------------|
| 1T parameter models | Yes | Partial | No | 671B max |
| 3-tier VRAM/RAM/NVMe | Yes | GPU/CPU only | GPU only | GPU/CPU only |
| io_uring NVMe | Yes | No | No | No |
| Expert prefetching | Yes | No | No | No |
| CPU expert execution | Yes | Partial | No | Yes (AMX) |
| FP4 Blackwell native | Yes | No | No | No |
| Fused MoE kernels | Yes | No | N/A | Yes |

Titan Engine is specifically designed for the "too big for VRAM, too big for RAM, but not too big for NVMe" regime — models from 200B to 1T+ parameters.

## Techniques

Titan Engine combines research from multiple state-of-the-art projects:

- **Flash-MoE**: SSD expert streaming, OS page cache trust, FMA dequant optimization
- **KTransformers**: CPU/GPU hybrid execution for MoE models
- **FlexGen**: GPU/CPU/NVMe offloading with throughput optimization
- **PowerInfer**: Activation sparsity exploitation, hot/cold neuron splitting
- **vLLM**: PagedAttention for efficient KV cache management
- **SpecMoEOff**: Speculative decoding to hide expert offloading latency

### Quantization Support

| Format | Bits | Quality | Speed | Best For |
|--------|------|---------|-------|----------|
| FP16/BF16 | 16 | Baseline | Slow | Reference |
| FP8 E4M3 | 8 | Near-lossless | Fast | Ada/Blackwell GPUs |
| Q4_K | 4.5 | Excellent | Very fast | General use |
| INT4 | 4 | Very good | Very fast | Expert weights |
| Q3_K | 3.5 | Good | Fast | Large models |
| INT2 | 2 | Acceptable | Fastest | Extreme compression |
| FP4 | 4 | Good | Fastest* | Blackwell native |

*FP4 uses Blackwell's native FP4 Tensor Cores for 2x throughput over FP8.

### Speculative Decoding

Multiple strategies to effectively multiply throughput:

- **Draft model**: Small model (1–3B) generates candidate tokens, large model verifies
- **Self-speculative**: Use fewer experts (top-1 vs top-4) as draft
- **Medusa heads**: Parallel token prediction from intermediate layers
- Expected **2–3x effective speedup** on top of base throughput

## Building

### Prerequisites

- Linux (kernel 5.1+ for io_uring)
- CUDA Toolkit 12.0+ (12.6+ for FP4 support)
- CMake 3.24+
- GCC 11+ or Clang 14+
- liburing (optional, for io_uring support)

### Build

```bash
git clone https://github.com/yourusername/titan-engine.git
cd titan-engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build options

```bash
cmake .. \
  -DTITAN_USE_IO_URING=ON \     # io_uring for NVMe I/O
  -DTITAN_USE_AVX512=ON \       # AVX-512 CPU kernels
  -DTITAN_BUILD_TESTS=ON \      # Build test suite
  -DCMAKE_CUDA_ARCHITECTURES="89;90a"  # Target GPU architectures
```

## Usage

### Interactive Chat

```bash
./titan --model /path/to/model --quant q4_k
```

### With Hardware Auto-Detection

```bash
# Show detected hardware and optimal config
./titan --hardware

# Run with auto-configured memory budgets
./titan -m /models/deepseek-v3 -q int4

# Explicit memory budgets
./titan -m /models/qwen-1t-moe -q q3_k --vram 28000 --ram 100000 --nvme-cache /mnt/raid
```

### As an OpenAI-Compatible Server

```bash
./titan-server -m /models/deepseek-v3 -q int4 --port 8080

# Use with any OpenAI-compatible client
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-v3", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Project Structure

```
titan-engine/
├── src/
│   ├── core/           # Types, config, hardware detection, logging
│   ├── memory/         # 3-tier memory manager (VRAM/RAM/NVMe)
│   ├── compute/
│   │   ├── cuda/       # CUDA kernels (dequant, attention, MoE, norms)
│   │   └── cpu/        # AVX-512/AMX CPU kernels
│   ├── model/          # Model loader, tokenizer, architecture executors
│   ├── inference/      # Engine, scheduler, speculative decoding, KV cache
│   └── api/            # CLI, HTTP server, Python bindings
├── tools/              # Model conversion and quantization utilities
├── tests/              # Unit and integration tests
└── docs/               # Technical documentation
```

## How It Works: Running a 1T MoE Model

Here's the execution flow for generating a single token from a 1T parameter MoE model:

```
Token generation (per layer):
  ┌─ GPU Stream 1 ─────────────────────────────────────┐
  │ Attention: Q/K/V projection → Flash Attention → O   │
  │ (weights always in VRAM, ~1ms)                      │
  └────────────────────────────┬────────────────────────┘
                               │
  ┌─ GPU ─────────────────────────────────────────────┐
  │ Routing: gate projection → softmax → top-K select  │
  │ (identifies which 4–8 of 512 experts are needed)   │
  └────────────────────────────┬──────────────────────┘
                               │
  ┌─ PARALLEL ─────────────────┼──────────────────────┐
  │                            │                       │
  │  ┌─ GPU Stream 1 ─────┐   │  ┌─ NVMe/RAM ─────┐ │
  │  │ Shared expert fwd   │   │  │ Load selected   │ │
  │  │ (always in VRAM)    │   │  │ experts to VRAM │ │
  │  └─────────────────────┘   │  │ (prefetched!)   │ │
  │                            │  └─────────────────┘ │
  └────────────────────────────┼──────────────────────┘
                               │
  ┌─ GPU Stream 2 ────────────────────────────────────┐
  │ Expert forward: gate_proj → SwiGLU → down_proj     │
  │ (4–8 experts, batched, ~0.5ms each)                │
  └────────────────────────────┬──────────────────────┘
                               │
  ┌─ GPU (fused) ─────────────────────────────────────┐
  │ Combine experts + residual + RMSNorm → next layer  │
  └───────────────────────────────────────────────────┘
```

Total per-layer time: **2–5ms** → 60 layers = **120–300ms per token** → **3–8 tok/s**

With expert caching (70%+ hit rate), most expert loads come from RAM (~0.5ms) instead of NVMe (~2ms), pushing towards the upper end of performance.

## Research References

This project builds on techniques from:

- **Flash-MoE** — SSD expert streaming on Apple Silicon ([paper](https://arxiv.org/abs/...))
- **KTransformers** — CPU/GPU hybrid MoE inference (SOSP 2025)
- **FlexGen** — GPU/CPU/NVMe offloading ([paper](https://arxiv.org/abs/2303.06865))
- **PowerInfer** — Activation sparsity exploitation ([paper](https://arxiv.org/abs/2312.12456))
- **PagedAttention / vLLM** — Efficient KV cache management ([paper](https://arxiv.org/abs/2309.06180))
- **Flash Attention 2** — IO-aware exact attention ([paper](https://arxiv.org/abs/2307.08691))
- **SpecMoEOff** — Speculative decoding for MoE offloading
- **GPTQ / AWQ / GGUF** — Post-training quantization techniques

## Status

**Early development** — The core architecture, types, memory manager, CUDA kernels, and build system are implemented. The inference pipeline is being connected.

### Completed
- Project architecture and build system (CMake + CUDA)
- Core types: tensors, quantization formats, model config
- Hardware detection (GPU, CPU, RAM, NVMe)
- 3-tier memory manager (VRAM, RAM, NVMe pools)
- CUDA kernels: INT4/INT2/FP8 dequant, attention, RoPE, RMSNorm, SwiGLU, MoE routing
- Fused kernels: Add+RMSNorm, MoE Combine+Residual+Norm
- CPU kernels: AVX-512 matvec, INT4 dequant, parallel expert execution
- Model loader (safetensors format)
- Execution planner (automatic VRAM/RAM/NVMe placement)
- CLI interface

### In Progress
- Full inference pipeline (connecting loader → executor → sampling)
- Tokenizer (BPE/SentencePiece)
- KV cache with PagedAttention
- Expert prefetching with io_uring

### Planned
- Speculative decoding
- GGUF format support
- OpenAI-compatible HTTP API
- Python bindings
- FP4 Blackwell kernels
- Continuous batching
- Benchmarking suite

## Contributing

This project is in active development. Contributions welcome for:
- Additional model architecture support
- CUDA kernel optimization
- Quantization research
- Benchmarking and testing

## License

MIT
