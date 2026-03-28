# Titan Engine

**Run models up to 1 trillion parameters on a single machine.**

Titan Engine is a high-performance C++/CUDA LLM inference engine designed for consumer and prosumer hardware. With 3-tier memory management (VRAM → RAM → NVMe), aggressive quantization, and MoE-aware expert streaming, a single RTX 5090 + EPYC + 128GB RAM + NVMe RAID can run models **30x larger than its VRAM**.

## Quick Start

### Prerequisites

- **Linux** (kernel 5.1+ for io_uring; Ubuntu 24.04 tested)
- **CUDA Toolkit 12.8+** — required for sm_100 Blackwell (RTX 5090). sm_89 Ada (RTX 4090) works with 12.0+.
- **CMake 3.24+**
- **GCC 11+** or **Clang 14+**
- **liburing** (optional — falls back to pread without it)

### Build (Ubuntu 24.04 — Native, Recommended)

```bash
# 1. Install CUDA 12.8 (skip if already installed — verify with: nvcc --version)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8 cmake build-essential

# 2. Set PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# 3. Clone and build
git clone https://github.com/wittedinit/titan-engine.git
cd titan-engine && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100
make -j$(nproc)
```

> **Ubuntu 24.04 known issue:** If `nvidia-cuda-toolkit` (the Ubuntu system package, version 12.0) is also installed, it will shadow the 12.8 nvcc and cause build failures (`_Float32` errors, `compute_100` unsupported). Fix: `sudo apt-get remove --purge nvidia-cuda-toolkit` then rebuild.

Build options:
```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="89;100" \   # 89=Ada (RTX 4090), 100=Blackwell (RTX 5090)
  -DTITAN_USE_IO_URING=ON \               # io_uring for NVMe I/O
  -DTITAN_USE_AVX512=ON \                 # AVX-512 CPU expert execution
  -DTITAN_BUILD_TESTS=ON \                # Build test suite
  -DTITAN_BUILD_PYTHON=ON                 # Python bindings (requires pybind11)
```

### Build in Docker

```bash
# Requires CUDA 12.8 image for sm_100 Blackwell support
docker run --gpus all -it --rm \
  -v /path/to/titan-engine:/workspace \
  -v /path/to/models:/models \
  -p 8080:8080 \
  nvidia/cuda:12.8.1-devel-ubuntu22.04 bash

# Inside container:
apt-get update && apt-get install -y cmake build-essential
cd /workspace && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100
make -j$(nproc)
```

### Run (Interactive Chat)

```bash
# HuggingFace safetensors directory
./titan -m /path/to/llama-3.1-8b-instruct -q q4_k

# GGUF file (auto-detected)
./titan -m /path/to/llama-3.1-8b.Q4_K_M.gguf

# MoE model (auto-detected from config.json)
./titan -m /path/to/deepseek-v3 -q int4

# Kimi K2.5 (~1T params) with FP4 quantization
./titan -m /path/to/Kimi-K2.5-NVFP4 -q fp4

# Show detected hardware and exit
./titan --hardware
```

### Run (HTTP API Server — OpenAI-Compatible)

```bash
# Start the server
./titan -m /path/to/model -q q4_k --serve --port 8080

# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"titan","messages":[{"role":"user","content":"Hello!"}]}'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"titan","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

**API Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (`{"status":"ok"}`) |
| `GET` | `/v1/models` | List loaded model |
| `POST` | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| `POST` | `/v1/completions` | Text completion |

Works with: **OpenAI Python SDK**, **Open WebUI**, **LM Studio**, **curl**, and any OpenAI-compatible client.

```python
# OpenAI Python SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="titan",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Run (Python Bindings)

```python
import titan

# Detect hardware
print(titan.hardware())

# Load and generate
engine = titan.Engine()
engine.load("/path/to/model", quant="q4_k", context=8192)

# Non-streaming
result = engine.generate("Hello!", temperature=0.7, max_tokens=100)
print(result)

# Streaming
engine.generate("Hello!", callback=lambda t: print(t, end="", flush=True))

# Chat-style
response = engine.chat([
    ("system", "You are a helpful assistant."),
    ("user", "What is the capital of France?")
])

# Tokenize
tokens = engine.encode("Hello world")
text = engine.decode(tokens)
```

Build with: `cmake .. -DTITAN_BUILD_PYTHON=ON` (requires `pip install pybind11`)

### Full CLI Options

```
Usage: titan [options]

Options:
  -m, --model PATH       Model directory (HuggingFace) or .gguf file
  -q, --quant TYPE       Quantization: fp16, fp8, fp4, int4, q4_k, q3_k, int2
  -c, --context N        Max context length (default: 8192)
  --vram N               VRAM budget in MB (default: auto-detect)
  --ram N                RAM budget in MB (default: auto-detect)
  --threads N            I/O threads (default: 4)
  --temp T               Temperature (default: 0.7)
  --top-p P              Top-p / nucleus sampling (default: 0.9)
  --top-k K              Top-k sampling (default: 40)
  --max-tokens N         Max tokens to generate (default: 2048)
  --no-prefetch          Disable expert prefetching
  --serve                Start as HTTP API server (OpenAI-compatible)
  --host HOST            Server bind address (default: 0.0.0.0)
  --port PORT            Server port (default: 8080)
  --hardware             Print hardware info and exit
  -v, --verbose          Verbose logging (per-layer timing)
  -h, --help             Show help

In-chat commands:
  /stats                 Show memory usage and cache hit rates
  /help                  Show available commands
  exit                   Quit
```

---

## Tools

### Model Conversion

**NVIDIA FP4 Conversion** (BF16 → NVFP4 for Blackwell GPUs):
```bash
pip install nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com

python tools/convert_nvfp4.py \
  --model /path/to/Kimi-K2.5 \
  --output /path/to/Kimi-K2.5-nvfp4 \
  --format nvfp4 \
  --calib-data wikitext
```

Or use NVIDIA's pre-quantized checkpoints:
```bash
huggingface-cli download nvidia/Kimi-K2-Thinking-NVFP4 --local-dir ./Kimi-K2-NVFP4
```

**Titan Native Format** (pre-quantized, layout-optimized for NVMe streaming):
```bash
python tools/convert_titan.py \
  --model /path/to/model \
  --quant q4_k \
  --output /path/to/model.titan
```

### Dense → MoE Conversion

Convert dense FFN layers into expert-routed MoE for expert streaming:
```bash
python tools/moeify.py \
  --model ./llama-70b \
  --num-experts 16 --top-k 4 \
  --method cluster \
  --output ./llama-70b-moe
```

Methods: `random` (fast), `cluster` (k-means, good quality), `svd` (best quality, slow).

### Benchmarking

```bash
# Single benchmark
python tools/benchmark.py --model /path/to/model -q q4_k --tokens 100

# Compare quantization levels
python tools/benchmark.py --model /path/to/model --sweep q3_k,q4_k,int4,fp8

# Regression testing
python tools/benchmark.py --model /path/to/model --regression baseline.json
```

Measures: TTFT, decode tok/s, VRAM usage, expert cache hit rate. JSON output for CI.

---

## Key Idea

You don't need to fit the entire model in GPU memory. With 3-tier memory management, aggressive quantization, and MoE sparsity, a machine with 32GB VRAM + 128GB RAM + fast NVMe RAID can run models **30x larger than its VRAM**.

## Target Hardware & Expected Performance

| Model | Parameters | Active/Token | Quant | Est. tok/s |
|-------|-----------|-------------|-------|-----------|
| Llama 3.x 8B | 8B | 8B (dense) | Q4_K | 80–120 |
| Llama 3.x 70B | 70B | 70B (dense) | Q4_K | 15–25 |
| Llama 3.x 405B | 405B | 405B (dense) | Q4_K | 3–6 |
| DeepSeek-V3 671B | 671B | 37B (MoE) | Q4_K | 12–20 |
| Qwen3.5-MoE 397B | 397B | 17B (MoE) | Q4 | 15–30 |
| Kimi K2.5 ~1T | ~1T | ~60B (MoE) | FP4/Q3–Q4 | 8–15 |

*Target hardware: RTX 5090 (32GB), EPYC 64-core, 128GB DDR5, 4x NVMe RAID 0 (~28 GB/s)*

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Titan Engine                       │
├─────────────────────────────────────────────────────┤
│  API: CLI Chat + HTTP (OpenAI-compatible) + Python  │
├─────────────────────────────────────────────────────┤
│  Inference: Engine, KV Cache, Speculative Decoding, │
│             Continuous Batching                      │
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

### Quantization Support

| Format | Bits | Best For |
|--------|------|----------|
| FP16/BF16 | 16 | Reference quality |
| FP8 E4M3 | 8 | Ada/Blackwell GPUs |
| Q4_K | 4.5 | General use (best quality/speed tradeoff) |
| INT4 | 4 | Expert weights |
| Q3_K | 3.5 | Large models (1T range) |
| INT2 | 2 | Extreme compression |
| FP4 E2M1 | 4 | Blackwell native Tensor Cores (~2x over FP8) |

### Model Format Support

| Format | Status | Notes |
|--------|--------|-------|
| **HuggingFace safetensors** | Full | Single-file and sharded, FP16/BF16/FP32 auto-conversion |
| **GGUF** | Full | v3 format, all GGML quant types |
| **Titan native** | Full | Pre-quantized, layout-optimized for NVMe streaming |
| **NVIDIA NVFP4** | Full | Via convert_nvfp4.py or pre-quantized HF checkpoints |

## Project Structure

```
titan-engine/
├── CMakeLists.txt                  # Build system (CUDA + C++17)
├── CLAUDE.md                       # AI assistant guardrails and conventions
├── src/
│   ├── core/                       # Types, config, hardware detection, logging
│   │   ├── types.h/cpp             # DType, Tensor, ModelConfig, RuntimeConfig
│   │   ├── config.h/cpp            # HuggingFace config.json parser, CLI args
│   │   ├── hardware.h/cpp          # GPU/CPU/RAM/NVMe detection + execution planning
│   │   └── logging.h/cpp           # Timestamped logging with severity levels
│   ├── memory/                     # 3-tier memory manager
│   │   ├── memory_manager.h/cpp    # Orchestrator + expert LRU cache
│   │   ├── vram_pool.cpp           # CUDA device memory with sub-allocation
│   │   ├── ram_pool.cpp            # Pinned system RAM for fast DMA
│   │   ├── nvme_pool.cpp           # io_uring async I/O with thread pool fallback
│   │   └── prefetcher.h/cpp        # Predictive expert prefetching (frequency + temporal)
│   ├── compute/
│   │   ├── cuda/                   # GPU kernels (9 files)
│   │   │   ├── cuda_check.h        # CUDA_CHECK_LAUNCH() debug macro
│   │   │   ├── dequant.cu          # INT4/INT2/FP8 dequant + matvec
│   │   │   ├── fp4.cu              # FP4 E2M1 dequant + quantization (Blackwell)
│   │   │   ├── gemv.cu             # cuBLAS FP32 sgemv + vector ops
│   │   │   ├── attention.cu        # Flash Attention decode + RoPE
│   │   │   ├── moe.cu              # Expert routing (gate → softmax → topK)
│   │   │   ├── norm.cu             # RMSNorm, LayerNorm
│   │   │   ├── activation.cu       # SwiGLU, GELU, fused Add+RMSNorm, fused MoE combine
│   │   │   ├── sampling.cu         # Temperature/top-p/top-k/argmax on GPU
│   │   │   └── sparse.cu           # Activation sparsity (predictor, sparse matvec)
│   │   ├── cpu/                    # CPU fallback kernels
│   │   │   ├── matmul_avx.cpp      # AVX-512 FP32 + INT4 dequant matvec
│   │   │   └── expert_cpu.cpp      # Parallel expert execution across CPU cores
│   │   └── dispatch.h              # Route compute to GPU/CPU based on weight location
│   ├── model/                      # Model loading and execution
│   │   ├── architecture.h          # Abstract model interface (pure virtual)
│   │   ├── loader.h/cpp            # Safetensors reader with GPU staging
│   │   ├── gguf_loader.h/cpp       # GGUF v3 format reader
│   │   ├── tokenizer.h/cpp         # BPE tokenizer (HuggingFace tokenizer.json)
│   │   ├── dense.h/cpp             # Dense transformer executor
│   │   ├── moe.h/cpp               # MoE executor (expert routing + 3-tier streaming)
│   │   ├── sparsity.h/cpp          # Activation sparsity profiler + sparse executor
│   │   └── quantizer.cpp           # Weight quantization pipeline
│   ├── inference/                   # Inference orchestration
│   │   ├── engine.h/cpp            # Main engine (prefill + decode + model detection)
│   │   ├── kv_cache.h/cpp          # GPU-resident KV cache
│   │   ├── speculative.h/cpp       # Speculative decoding (draft + self-speculative)
│   │   ├── batch.h/cpp             # Continuous batching scheduler
│   │   └── scheduler.cpp           # Layer execution scheduler
│   └── api/
│       ├── cli.cpp                 # Interactive chat with streaming output
│       ├── http.h/cpp              # Minimal HTTP server (zero dependencies)
│       ├── server.cpp              # OpenAI-compatible API (SSE streaming)
│       └── python/titan_bindings.cpp  # pybind11 Python module
├── tools/
│   ├── moeify.py                   # Dense → MoE conversion + sparsity profiling
│   ├── convert_nvfp4.py            # BF16 → NVFP4/MXFP4 via NVIDIA Model Optimizer
│   ├── convert_titan.py            # HF → Titan native format
│   ├── convert.py                  # General conversion (placeholder)
│   └── benchmark.py                # Performance benchmarking + regression testing
└── tests/
    ├── test_types.cpp              # Core type tests
    ├── test_memory.cpp             # Memory manager tests
    └── test_kernels.cu             # CUDA kernel correctness (vs CPU reference)
```

**66 source files, 11,600+ lines of C++/CUDA/Python.**

## Status

**v0.3.1** — Verified build on Ubuntu 24.04 + RTX 5090 (sm_100). All features implemented. Full 6-agent audit + real-hardware build validation complete.

### Implemented
- [x] Full forward pass: embedding → N layers → logits → sampling → text output
- [x] Dense executor with cuBLAS FP32 and INT4/FP4 dequant dispatch
- [x] MoE executor with double-buffered expert staging, 3-tier memory
- [x] BPE tokenizer (HuggingFace tokenizer.json, proper UTF-8 byte encoding)
- [x] Safetensors loader (single + sharded, FP16/BF16→FP32 auto-conversion)
- [x] GGUF loader (v3, all GGML quant types, auto-detection)
- [x] KV cache (GPU-resident, per-position update, engine-initialized)
- [x] Hardware auto-detection (GPU, CPU, RAM, NVMe, RAID)
- [x] Execution planner (automatic VRAM/RAM/NVMe weight placement)
- [x] 9 CUDA kernel files with CUDA_CHECK_LAUNCH() debug assertions
- [x] CPU AVX-512 expert execution path with scalar fallback
- [x] Activation sparsity system (profiler + predictor + sparse kernels)
- [x] MoE-ification tool (dense → MoE conversion via cluster/SVD/random)
- [x] Interactive CLI with streaming output and in-chat commands
- [x] OpenAI-compatible HTTP API server (SSE streaming, CORS, SIGPIPE-safe)
- [x] Speculative decoding (draft model + self-speculative, pre-allocated buffers)
- [x] Python bindings (pybind11: Engine.load, generate, chat, encode, decode)
- [x] FP4 Blackwell Tensor Core kernels (E2M1 dequant + quantization)
- [x] Continuous batching (per-request KV slots, indexed buffers, dynamic scheduling)
- [x] Expert prefetcher (frequency + temporal prediction, O_DIRECT aligned I/O)
- [x] Benchmarking suite (TTFT, tok/s, VRAM, quant sweeps, regression testing)
- [x] Titan native format (pre-quantized INT4, layout-optimized, manifest.json)
- [x] NVIDIA FP4 converter (BF16 → NVFP4/MXFP4 via Model Optimizer)
- [x] Proper VRAM lifecycle (destructors free all model weights, no leaks)
- [x] Thread-safe cuBLAS init, atomic NVMe stats, mutex-guarded prefetcher
- [x] Ubuntu 24.04 + GCC 13 build validated (sm_100, CUDA 12.8, AVX-512)

## Research References

- **Flash-MoE** — SSD expert streaming on Apple Silicon
- **KTransformers** — CPU/GPU hybrid MoE inference (SOSP 2025)
- **FlexGen** — GPU/CPU/NVMe offloading ([paper](https://arxiv.org/abs/2303.06865))
- **PowerInfer** — Activation sparsity exploitation ([paper](https://arxiv.org/abs/2312.12456))
- **PagedAttention / vLLM** — Efficient KV cache management ([paper](https://arxiv.org/abs/2309.06180))
- **Flash Attention 2** — IO-aware exact attention ([paper](https://arxiv.org/abs/2307.08691))
- **SpecMoEOff** — Speculative decoding for MoE offloading
- **GPTQ / AWQ / GGUF** — Post-training quantization techniques

## Contributing

Contributions welcome! Priority areas:
- CUDA kernel optimization and benchmarking on real hardware
- Additional model architecture support (Qwen3, Gemma3, etc.)
- Real-hardware performance tuning and regression baselines
- io_uring expert prefetching with true async completion

## License

MIT
