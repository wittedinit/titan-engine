# Titan Engine

**Run models up to 1 trillion parameters on a single machine.**

Titan Engine is a high-performance C++/CUDA LLM inference engine that pushes the boundaries of what's possible on consumer and prosumer hardware. Built on the same principles as [flash-moe](https://github.com/nicholaschenai/flash-moe-main) — which runs a 397B parameter model on a MacBook — but engineered for Linux/CUDA systems with discrete GPUs, large RAM, and fast NVMe storage.

## Quick Start

### Prerequisites

- **Linux** (kernel 5.1+ for io_uring, also compiles on macOS for development)
- **CUDA Toolkit 12.0+** (12.6+ for FP4/Blackwell support)
- **CMake 3.24+**
- **GCC 11+** or **Clang 14+**
- **liburing** (optional, for io_uring NVMe I/O — falls back to pread without it)

### Build

```bash
git clone https://github.com/wittedinit/titan-engine.git
cd titan-engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Build options:
```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTITAN_USE_IO_URING=ON \                    # io_uring for NVMe I/O
  -DTITAN_USE_AVX512=ON \                      # AVX-512 CPU expert execution
  -DTITAN_BUILD_TESTS=ON \                     # Build test suite
  -DCMAKE_CUDA_ARCHITECTURES="80;89;90a"       # Target GPU architectures
```

### Run

```bash
# Run with a HuggingFace model directory (safetensors format)
./titan -m /path/to/llama-3.1-8b-instruct -q q4_k

# Run with a GGUF file (auto-detected)
./titan -m /path/to/llama-3.1-8b.Q4_K_M.gguf

# Run a MoE model (auto-detected from config.json)
./titan -m /path/to/deepseek-v3 -q int4

# Show detected hardware and exit
./titan --hardware
```

### Full CLI Options

```
Usage: titan [options]

Options:
  -m, --model PATH       Model directory (HuggingFace) or .gguf file
  -q, --quant TYPE       Quantization: fp16, fp8, int4, q4_k, q3_k, int2
  -c, --context N        Max context length (default: 8192)
  --vram N               VRAM budget in MB (default: auto-detect)
  --ram N                RAM budget in MB (default: auto-detect)
  --threads N            I/O threads (default: 4)
  --temp T               Temperature (default: 0.7)
  --top-p P              Top-p / nucleus sampling (default: 0.9)
  --top-k K              Top-k sampling (default: 40)
  --max-tokens N         Max tokens to generate (default: 2048)
  --no-prefetch          Disable expert prefetching
  --hardware             Print hardware info and exit
  -v, --verbose          Verbose logging (per-layer timing)
  -h, --help             Show help

In-chat commands:
  /stats                 Show memory usage and cache hit rates
  /help                  Show available commands
  exit                   Quit
```

### Example Session

```
$ ./titan -m ./Meta-Llama-3.1-8B-Instruct -q q4_k

  ████████╗██╗████████╗ █████╗ ███╗   ██╗
  ╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║
     ██║   ██║   ██║   ███████║██╔██╗ ██║
     ██║   ██║   ██║   ██╔══██║██║╚██╗██║
     ██║   ██║   ██║   ██║  ██║██║ ╚████║
     ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝
  Universal LLM Inference Engine

[  0.001] INFO  === Titan Engine Hardware Profile ===
[  0.002] INFO  GPU 0: NVIDIA GeForce RTX 5090
[  0.002] INFO    VRAM: 32.0 GB total, 31.2 GB free
[  0.003] INFO  CPU: AMD EPYC 9654 96-Core Processor
[  0.004] INFO  Memory: 128.0 GB total, 112.3 GB available
[  0.005] INFO  Storage: / (NVMe, 28.0 GB/s read)
[  0.120] INFO  Tokenizer loaded: 128256 tokens, BOS=128000 EOS=128001
[  2.450] INFO  All weights loaded to GPU
[  2.451] INFO  Dense executor ready: llama (32L, h=4096, heads=32/8, inter=14336)

Model: llama (8.0B params)
Quant: q4_k | Context: 8192 | Temp: 0.7

> Hello! What can you help me with?
I'm an AI assistant. I can help with a wide variety of tasks...
```

---

## Key Idea

You don't need to fit the entire model in GPU memory. With 3-tier memory management (VRAM → RAM → NVMe), aggressive quantization, and MoE sparsity, a machine with 32GB VRAM + 128GB RAM + fast NVMe RAID can run models **30x larger than its VRAM**.

## Target Hardware & Expected Performance

| Model | Parameters | Active/Token | Quant | Est. tok/s |
|-------|-----------|-------------|-------|-----------|
| Llama 3.x 8B | 8B | 8B (dense) | Q4_K | 80–120 |
| Llama 3.x 70B | 70B | 70B (dense) | Q4_K | 15–25 |
| Llama 3.x 405B | 405B | 405B (dense) | Q4_K | 3–6 |
| DeepSeek-V3 671B | 671B | 37B (MoE) | Q4_K | 12–20 |
| Qwen3.5-MoE 397B | 397B | 17B (MoE) | Q4 | 15–30 |
| Kimi K2.5 ~1T | ~1T | ~60B (MoE) | Q3–Q4 | 8–15 |

*Target hardware: RTX 5090 (32GB), EPYC 64-core, 128GB DDR5, 4x NVMe RAID 0 (~28 GB/s)*

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Titan Engine                       │
├─────────────────────────────────────────────────────┤
│  API: CLI Chat + HTTP (OpenAI-compatible) + Python  │
├─────────────────────────────────────────────────────┤
│  Inference: Engine, KV Cache, Speculative Decoding  │
├─────────────────────────────────────────────────────┤
│  Model: Dense Executor │ MoE Executor │ Hybrid      │
├─────────────────────────────────────────────────────┤
│  Compute: CUDA Kernels (FP4–FP32, INT2–INT8)       │
│           + cuBLAS gemv + CPU AVX-512/AMX           │
├─────────────────────────────────────────────────────┤
│  Memory: 3-Tier Manager (VRAM ↔ RAM ↔ NVMe)        │
│          Expert LRU Cache + Predictive Prefetching  │
├─────────────────────────────────────────────────────┤
│  Loaders: Safetensors + GGUF + Titan format         │
│  I/O: io_uring async NVMe + CUDA DMA + pinned RAM  │
└─────────────────────────────────────────────────────┘
```

### 3-Tier Memory Manager

| Tier | Storage | Bandwidth | Contents |
|------|---------|-----------|----------|
| **Hot (VRAM)** | 32 GB | ~2 TB/s | Attention weights, embeddings, active experts, KV cache |
| **Warm (RAM)** | 128 GB | ~460 GB/s | Expert LRU cache, overflow KV, dense FFN weights |
| **Cold (NVMe)** | Unlimited | ~28 GB/s | Full model weights, cold experts |

**Key advantage over Apple Silicon**: On discrete GPU systems, NVMe and GPU don't share a memory bus — true parallel I/O + compute overlap.

### Model Format Support

| Format | Status | Notes |
|--------|--------|-------|
| **HuggingFace safetensors** | Full | Single-file and sharded, FP16/BF16/FP32 with auto-conversion |
| **GGUF** | Full | v3 format, all GGML quant types (Q4_K, Q3_K, Q8_0, Q2_K, etc.) |
| **Titan format** | Planned | Pre-quantized, layout-optimized for streaming |

### Quantization Support

| Format | Bits | Best For |
|--------|------|----------|
| FP16/BF16 | 16 | Reference quality |
| FP8 E4M3 | 8 | Ada/Blackwell GPUs |
| Q4_K | 4.5 | General use (best quality/speed tradeoff) |
| INT4 | 4 | Expert weights |
| Q3_K | 3.5 | Large models (1T range) |
| INT2 | 2 | Extreme compression |
| FP4 | 4 | Blackwell native Tensor Cores |

The dense executor auto-dispatches between **cuBLAS sgemv** (FP32/FP16) and **custom dequant kernels** (INT4/INT2) based on weight format.

### Running Dense Models Faster

Two approaches for dense models that don't have a native MoE variant:

**1. Activation Sparsity** (no conversion needed, ~2-3x speedup):
```bash
# Profile which neurons are active
python tools/moeify.py --model ./llama-70b --profile-sparsity

# Titan uses the sparsity profile automatically at inference
./titan -m ./llama-70b -q q4_k
```

**2. MoE-ification** (convert dense → MoE, enables expert streaming):
```bash
# Convert dense FFN layers into 16 experts, top-4 routing
python tools/moeify.py --model ./llama-70b --num-experts 16 --top-k 4 \
                       --method cluster --output ./llama-70b-moe

# Now runs with expert streaming (75% params skipped per token)
./titan -m ./llama-70b-moe -q q4_k
```

Splitting methods: `random` (fast), `cluster` (k-means, good quality), `svd` (best quality, slow).

## How It Works: Token Generation Pipeline

```
Per-layer execution (decode phase):

  ┌─ GPU ─────────────────────────────────────────────┐
  │ 1. RMSNorm(hidden, attn_norm)                      │
  │ 2. Q = q_proj @ norm    (cuBLAS or INT4 dequant)   │
  │    K = k_proj @ norm                                │
  │    V = v_proj @ norm                                │
  │ 3. RoPE(Q, K, position)                             │
  │ 4. KV_cache[layer][pos] = (K, V)                   │
  │ 5. attn_out = FlashAttention(Q, K_cache, V_cache)  │
  │ 6. hidden += o_proj @ attn_out                      │
  └─────────────────────────────────────────────────────┘
                          │
  ┌─ GPU + NVMe (parallel for MoE) ────────────────────┐
  │ 7. RMSNorm(hidden, ffn_norm)                        │
  │ 8. [MoE] route = softmax(gate @ hidden) → topK     │
  │ 9. [MoE] Load K experts from RAM/NVMe → GPU        │
  │    [Dense] gate = gate_proj @ hidden                │
  │            up   = up_proj @ hidden                  │
  │ 10. act = SwiGLU(gate, up)                          │
  │ 11. hidden += down_proj @ act                       │
  │ 12. [MoE] Combine experts + shared + residual       │
  └─────────────────────────────────────────────────────┘
```

## Project Structure

```
titan-engine/
├── CMakeLists.txt                  # Build system (CUDA + C++17)
├── src/
│   ├── core/                       # Types, config, hardware detection, logging
│   │   ├── types.h/cpp             # DType, Tensor, ModelConfig, RuntimeConfig
│   │   ├── config.h/cpp            # HuggingFace config.json parser, CLI args
│   │   ├── hardware.h/cpp          # GPU/CPU/RAM/NVMe detection + execution planning
│   │   └── logging.h/cpp           # Timestamped logging
│   ├── memory/                     # 3-tier memory manager
│   │   ├── memory_manager.h/cpp    # Orchestrator + expert LRU cache
│   │   ├── vram_pool.cpp           # CUDA device memory with sub-allocation
│   │   ├── ram_pool.cpp            # Pinned system RAM for fast DMA
│   │   ├── nvme_pool.cpp           # io_uring async I/O with thread pool fallback
│   │   └── prefetcher.cpp          # Predictive expert prefetching
│   ├── compute/
│   │   ├── cuda/                   # GPU kernels
│   │   │   ├── dequant.cu          # INT4/INT2/FP8 dequant + matvec (FMA-optimized)
│   │   │   ├── gemv.cu             # cuBLAS FP32 sgemv + vector_add
│   │   │   ├── attention.cu        # Flash Attention decode + RoPE
│   │   │   ├── moe.cu             # Expert routing (gate → softmax → topK)
│   │   │   ├── norm.cu            # RMSNorm, LayerNorm
│   │   │   ├── activation.cu      # SwiGLU, GELU, fused Add+RMSNorm, fused MoE combine
│   │   │   ├── sampling.cu        # GPU-accelerated temperature/top-p/top-k/argmax
│   │   │   └── sparse.cu          # Activation sparsity kernels (predictor, sparse matvec)
│   │   ├── cpu/                    # CPU kernels for expert execution
│   │   │   ├── matmul_avx.cpp     # AVX-512 FP32 + INT4 dequant matvec
│   │   │   └── expert_cpu.cpp     # Parallel expert execution across CPU cores
│   │   └── dispatch.h             # Route compute to GPU/CPU based on weight location
│   ├── model/                      # Model loading and execution
│   │   ├── loader.h/cpp           # Safetensors reader with GPU staging
│   │   ├── gguf_loader.h/cpp      # GGUF v3 format reader (all quant types)
│   │   ├── tokenizer.h/cpp        # BPE tokenizer (HuggingFace tokenizer.json)
│   │   ├── architecture.h         # Abstract model interface
│   │   ├── dense.h/cpp            # Dense transformer executor (FP32 + INT4 dispatch)
│   │   ├── moe.h/cpp              # MoE executor (expert routing + 3-tier streaming)
│   │   ├── sparsity.h/cpp         # Activation sparsity profiler + sparse FFN executor
│   │   └── quantizer.cpp          # Weight quantization pipeline
│   ├── inference/                  # Inference orchestration
│   │   ├── engine.h/cpp           # Main engine (prefill + decode + auto model detection)
│   │   ├── kv_cache.h/cpp         # GPU-resident KV cache with per-position update
│   │   ├── scheduler.cpp          # Layer execution scheduler
│   │   ├── speculative.cpp        # Speculative decoding (draft model)
│   │   └── batch.cpp              # Continuous batching
│   └── api/
│       ├── cli.cpp                # Interactive chat with colored streaming output
│       └── server.cpp             # OpenAI-compatible HTTP server (planned)
├── tools/
│   ├── moeify.py                  # Dense → MoE conversion + sparsity profiling
│   ├── convert.py                 # HF → Titan format conversion
│   └── benchmark.py               # Performance benchmarking suite
└── tests/
    ├── test_types.cpp             # Core type tests
    ├── test_memory.cpp            # Memory manager tests
    └── test_kernels.cu            # CUDA kernel correctness (vs CPU reference)
```

**54 source files, 9,115 lines of C++/CUDA/Python.**

## Research References

- **Flash-MoE** — SSD expert streaming on Apple Silicon
- **KTransformers** — CPU/GPU hybrid MoE inference (SOSP 2025)
- **FlexGen** — GPU/CPU/NVMe offloading ([paper](https://arxiv.org/abs/2303.06865))
- **PowerInfer** — Activation sparsity exploitation ([paper](https://arxiv.org/abs/2312.12456))
- **PagedAttention / vLLM** — Efficient KV cache management ([paper](https://arxiv.org/abs/2309.06180))
- **Flash Attention 2** — IO-aware exact attention ([paper](https://arxiv.org/abs/2307.08691))
- **SpecMoEOff** — Speculative decoding for MoE offloading
- **GPTQ / AWQ / GGUF** — Post-training quantization techniques

## Status

**v0.1.0** — Complete inference pipeline with all core components implemented and connected.

### What Works
- Full forward pass: embedding → 32+ layers → logits → sampling → text output
- Dense executor with cuBLAS FP32 and INT4 dequant dispatch
- MoE executor with pre-allocated buffers, expert routing, shared experts, 3-tier memory
- BPE tokenizer (HuggingFace tokenizer.json)
- Safetensors loader (single + sharded, FP16→FP32 auto-conversion)
- GGUF loader (v3, all quant types, auto-detection)
- KV cache (GPU-resident, per-position update)
- Hardware auto-detection (GPU, CPU, RAM, NVMe, RAID)
- Execution planner (automatic VRAM/RAM/NVMe weight placement)
- 8 CUDA kernel files (dequant, attention, norms, activations, MoE, sampling, sparse, gemv)
- CPU AVX-512 expert execution path
- Activation sparsity system (profiler + predictor + sparse kernels)
- MoE-ification tool (dense → MoE conversion)
- Interactive CLI with streaming output

### Roadmap
- [ ] Speculative decoding (draft model + self-speculative)
- [ ] OpenAI-compatible HTTP API server
- [ ] Python bindings (pybind11)
- [ ] FP4 Blackwell native Tensor Core kernels
- [ ] Continuous batching for multi-user serving
- [ ] Expert prefetching with io_uring pipelining
- [ ] Benchmarking suite with automated regression testing
- [ ] Titan native format (pre-quantized, layout-optimized)

## Contributing

Contributions welcome! Priority areas:
- CUDA kernel optimization and benchmarking
- Additional model architecture support
- GGUF tokenizer extraction
- Real-hardware testing and performance tuning

## License

MIT
