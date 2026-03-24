# Titan Engine

Universal LLM inference engine — run models up to 1T parameters on consumer hardware.

## Project Overview

Primary languages: C++17, CUDA, Python. Tools include convert_nvfp4.py, convert_titan.py, benchmark.py, moeify.py. Always verify README accuracy against actual installation steps before committing.

## Architectural Invariants

### NEVER

- NEVER suggest external inference engines (vLLM, llama.cpp, TGI, SGLang, etc.) — we built Titan Engine; use it.
- NEVER assume public infrastructure, relay servers, or TURN servers unless explicitly approved.
- NEVER allocate memory per-token in hot paths — all buffers must be pre-allocated.
- NEVER use dynamic dispatch (virtual calls) in the inner inference loop — use compile-time dispatch or function pointers set at init.
- NEVER load an entire 1T model into RAM — always use the 3-tier memory hierarchy.
- NEVER bypass the execution planner's weight placement decisions.
- NEVER use `cudaMalloc`/`cudaFree` during generation — only during model load.
- NEVER commit code that compiles on only one platform without `#ifdef` guards (CUDA code must have CPU fallback stubs).
- NEVER modify the OpenAI-compatible API response format — clients depend on exact field names.
- NEVER use `git push --force` on main.

### ALWAYS

- ALWAYS use Titan's own HTTP server (`--serve`) for API endpoints, not external serving frameworks.
- ALWAYS run through the 3-tier memory path: VRAM → RAM → NVMe, with the execution planner deciding placement.
- ALWAYS use pre-allocated double-buffered expert staging for MoE models.
- ALWAYS use `pread`/`io_uring` for weight loading — never `mmap` (unpredictable page faults kill latency).
- ALWAYS quantize with per-group scale factors (never per-tensor — quality is unacceptable).
- ALWAYS include CPU reference implementations alongside CUDA kernels for testing.
- ALWAYS verify tensor shapes match expected dimensions before dispatching to kernels.
- ALWAYS check `cudaGetLastError()` after kernel launches in debug builds.
- ALWAYS write tests before implementing new kernel operations.

## Common Mistakes

These approaches seem reasonable but violate our constraints:

| Mistake | Why It's Wrong |
|---------|---------------|
| "Just use vLLM for serving" | We built Titan's HTTP server specifically for this. Using vLLM defeats the entire project. |
| "mmap the model file" | Page faults during inference cause unpredictable latency spikes. We use explicit `pread` with prefetching. |
| "cudaMalloc a temporary buffer" | CUDA allocator is slow and fragments VRAM. All buffers are pre-allocated at model load time. |
| "Use FP16 for everything" | A 1T model in FP16 = 2TB. Our whole point is aggressive quantization (FP4/INT4) with quality preservation. |
| "Per-tensor quantization is fine" | Per-tensor loses too much quality on outlier-heavy layers. Per-group (group_size=64) is mandatory. |
| "Just add more GPU memory" | Our target is consumer hardware (single RTX 5090 + EPYC + NVMe). We optimize for what exists, not what we wish existed. |
| "Load all experts into RAM" | A 1T MoE has ~500GB of experts. 128GB RAM can't hold them all. NVMe streaming with prefetching is the design. |
| "Use Python for the inference loop" | C++/CUDA only in the hot path. Python is for tools, bindings, and benchmarks. |

## Sub-Agent Instructions

Any spawned sub-agent working on this codebase MUST:

1. **Read this CLAUDE.md first** before making any changes.
2. **Check file ownership** — before modifying a file, check the shared TODO to see if another agent is working on it. If so, coordinate or choose a different file.
3. **Pull and rebase before committing** — run `git pull --rebase origin main` before every commit.
4. **Run tests after every change** — `cd build && make -j$(nproc) && ctest --output-on-failure`. Do not commit if tests fail.
5. **Never modify these files without explicit approval**: `CMakeLists.txt` (root), `src/core/types.h`, `src/inference/engine.h`, `src/api/http.h`. These are shared interfaces.
6. **Use feature branches** — never commit directly to main. Branch naming: `feat/<description>`, `fix/<description>`.
7. **If a test fails that you didn't write**, stop immediately. Check if the failure is in your changes or pre-existing. If pre-existing, report it. If yours, fix before proceeding.
8. **Scope your changes** — do not refactor, rename, or "improve" code outside your assigned task.
9. **Respect the NEVER/ALWAYS rules above** — violations will be rejected at review.

## Parallel Agent Coordination Protocol

When multiple agents work simultaneously:

1. **Shared TODO** — maintain a `TodoWrite` checklist with format: `[agent-name] file.cpp — description`. Check before touching any file.
2. **Conflict detection** — before staging files, run `git diff --name-only origin/main..HEAD` and compare against other agents' modified files. If overlap, coordinate.
3. **Merge order** — agents merge in dependency order: core → compute → model → inference → api. Never merge api changes before the inference changes they depend on.
4. **Cascading failure protocol** — if agent A's test failure could affect agent B's branch:
   - Agent A pauses and reports the failing test
   - Agent B checks if the failure is in shared code
   - If shared, both agents pause until the fix is verified
   - If isolated to A's branch, B continues

## TDD Protocol

For new features, enforce strict red-green-refactor:

1. **Red**: Write failing tests first. Run them. Confirm they fail for the right reason.
2. **Green**: Write minimum code to pass. Run full suite after every change.
3. **Refactor**: Clean up while keeping tests green.
4. **Self-review**: Check for type mismatches, off-by-one errors, incorrect assumptions, missing error handling.
5. **Mutation check**: For CUDA kernels, verify tests catch incorrect implementations (e.g., does the test fail if you zero-out the output?).

## General Rules

- After building a tool or engine in this project, use it — don't suggest external alternatives.
- Do not make UI changes, rename buttons, or modify user-facing labels unless explicitly requested. Keep changes scoped to what was asked.

## Git & PR Workflow

- When working on PRs, never push commits to already-merged branches. Always check PR/branch status before pushing.
- Feature branches only — never commit directly to main.
- Run full test suite before opening PR.
- PR title format: `feat: description` or `fix: description`.

## Build & Run

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100
make -j$(nproc)

# Interactive chat
./titan -m /path/to/model -q q4_k

# API server
./titan -m /path/to/model -q q4_k --serve --port 8080

# Docker (Unraid/headless)
docker run --gpus all -v /mnt/user/temp/models:/models -p 8080:8080 \
  titan-engine -m /models/Kimi-K2.5-NVFP4 -q fp4 --serve --port 8080
```

Supports HuggingFace safetensors directories, GGUF files, and Titan native format (auto-detected).

## Architecture

- **Language**: C++17 with CUDA kernels, cuBLAS for FP32 matvec
- **GPU**: NVIDIA CUDA (sm_80 Ampere, sm_89 Ada, sm_100 Blackwell)
- **Build**: CMake 3.24+, CUDA 12.0+
- **I/O**: io_uring for async NVMe reads, pread fallback on macOS/older Linux

## Key Design Principles

1. **3-tier memory**: VRAM (hot) → RAM (warm, LRU expert cache) → NVMe (cold, io_uring)
2. **MoE-first**: Optimized for sparse MoE models where only K of N experts are active per token
3. **Discrete GPU advantage**: NVMe reads don't compete with GPU compute (unlike unified memory)
4. **Pre-allocated buffers**: MoE executor uses double-buffered expert staging — zero per-token mallocs
5. **Dual compute path**: cuBLAS sgemv for FP32, custom dequant kernels for INT4/INT2/FP4
6. **Auto-detection**: Model format (safetensors vs GGUF vs Titan), model type (dense vs MoE), hardware capabilities

## Directory Layout

- `src/core/` — Types, config (HF config.json parser), hardware detection, logging
- `src/memory/` — 3-tier memory manager (VRAM pool, pinned RAM pool, NVMe io_uring pool), expert prefetcher
- `src/compute/cuda/` — 9 CUDA kernel files (dequant, gemv, fp4, attention, MoE, norms, activation, sampling, sparse)
- `src/compute/cpu/` — AVX-512 CPU kernels for RAM-resident expert execution
- `src/model/` — Safetensors loader, GGUF loader, BPE tokenizer, dense executor, MoE executor, sparsity system
- `src/inference/` — Engine orchestrator (prefill + decode), KV cache, speculative decoding, continuous batching
- `src/api/` — Interactive CLI, OpenAI-compatible HTTP server (SSE streaming), Python bindings (pybind11)
- `tools/` — moeify.py, convert.py, convert_nvfp4.py, convert_titan.py, benchmark.py
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
- `DType::FP4` → `cuda::dequant_matvec_fp4()` (FP4 E2M1 with constant LUT, Blackwell native)

## Conventions

- Quantization: group_size=64 for INT4, scale+bias per group (affine quantization)
- Expert layout: [gate_proj | up_proj | down_proj] contiguous in memory
- Memory alignment: 256 bytes for VRAM, 64 bytes for RAM (AVX-512)
- RMSNorm epsilon: 1e-5 (matches Llama default)
- All CUDA kernels use shared memory + warp-level reduction
