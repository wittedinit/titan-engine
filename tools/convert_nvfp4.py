#!/usr/bin/env python3
"""
Titan Engine — NVIDIA FP4 Conversion Tool

Converts BF16 HuggingFace checkpoints to NVFP4/MXFP4 format using
NVIDIA's TensorRT Model Optimizer, then produces a Titan-compatible
deployment package.

This is the recommended path for running models like Kimi K2.5 on
RTX 5090 (Blackwell) with native FP4 Tensor Cores (~2x throughput
over FP8).

Prerequisites:
    pip install nvidia-modelopt
    # OR
    git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
    cd TensorRT-Model-Optimizer && pip install -e .

Usage:
    # Convert BF16 model to NVFP4
    python convert_nvfp4.py --model /path/to/Kimi-K2.5 --output /path/to/Kimi-K2.5-nvfp4

    # Convert to MXFP4 (OpenAI-compatible format)
    python convert_nvfp4.py --model /path/to/Kimi-K2.5 --format mxfp4 --output /path/to/Kimi-K2.5-mxfp4

    # Convert with custom calibration dataset
    python convert_nvfp4.py --model /path/to/Kimi-K2.5 --calib-data wikitext --calib-size 512

    # Build TensorRT-LLM engine (optional, for max performance)
    python convert_nvfp4.py --model /path/to/Kimi-K2.5 --output /path/to/Kimi-K2.5-nvfp4 --build-engine

Flow:
    BF16 checkpoint
    → NVIDIA Model Optimizer PTQ (Post-Training Quantization)
    → NVFP4 or MXFP4 quantized checkpoint
    → (Optional) TensorRT-LLM engine build
    → Titan Engine loads the quantized checkpoint
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required NVIDIA tools are installed."""
    deps = {}

    # Check for modelopt
    try:
        import modelopt  # noqa
        deps["modelopt"] = True
    except ImportError:
        deps["modelopt"] = False

    # Check for torch
    try:
        import torch
        deps["torch"] = True
        deps["torch_bf16"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if torch.cuda.is_available():
            deps["gpu_name"] = torch.cuda.get_device_name(0)
            deps["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        deps["torch"] = False

    # Check for transformers
    try:
        import transformers  # noqa
        deps["transformers"] = True
    except ImportError:
        deps["transformers"] = False

    return deps


def install_modelopt():
    """Install NVIDIA Model Optimizer."""
    print("Installing NVIDIA Model Optimizer...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "nvidia-modelopt[all]", "--extra-index-url",
            "https://pypi.nvidia.com"
        ])
        return True
    except subprocess.CalledProcessError:
        print("pip install failed. Trying from source...")
        try:
            subprocess.check_call([
                "git", "clone",
                "https://github.com/NVIDIA/TensorRT-Model-Optimizer.git",
                "/tmp/TensorRT-Model-Optimizer"
            ])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-e",
                "/tmp/TensorRT-Model-Optimizer"
            ])
            return True
        except subprocess.CalledProcessError:
            return False


def convert_nvfp4(args):
    """Convert BF16 model to NVFP4 using NVIDIA Model Optimizer."""
    import torch
    import modelopt.torch.quantization as mtq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model
    output_path = args.output

    print(f"Loading BF16 model from {model_path}...")
    print(f"  This may require significant GPU memory for large models.")
    print(f"  For Kimi K2.5 (~1T params), use multiple GPUs or offloading.")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Model loaded: {model.config.model_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    # Prepare calibration data
    print(f"\nPreparing calibration data ({args.calib_size} samples)...")
    if args.calib_data == "wikitext":
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        calib_texts = [t for t in dataset["text"] if len(t) > 100][:args.calib_size]
    else:
        # Use simple prompts as calibration data
        calib_texts = [
            "The meaning of life is",
            "In computer science, a hash function is",
            "The capital of France is Paris, which is known for",
            "Machine learning algorithms can be categorized into",
            "The quick brown fox jumps over the lazy dog",
        ] * (args.calib_size // 5 + 1)
        calib_texts = calib_texts[:args.calib_size]

    def calibrate_loop(model):
        """Run calibration data through the model."""
        for text in calib_texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=512,
                              truncation=True).to(model.device)
            with torch.no_grad():
                model(**inputs)

    # Quantize
    if args.format == "nvfp4":
        print(f"\nQuantizing to NVFP4 (FP4 E2M1 with per-group scaling)...")
        quant_config = mtq.FP8_DEFAULT_CFG.copy()
        # Override to FP4 for Blackwell
        quant_config["quant_cfg"]["*weight_quantizer"] = {
            "num_bits": 4, "axis": None, "enable": True
        }
        mtq.quantize(model, quant_config, forward_loop=calibrate_loop)
    elif args.format == "mxfp4":
        print(f"\nQuantizing to MXFP4 (Microscaling FP4)...")
        quant_config = {
            "quant_cfg": {
                "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
                "*input_quantizer": {"enable": False},
            },
            "algorithm": "max",
        }
        mtq.quantize(model, quant_config, forward_loop=calibrate_loop)
    else:
        print(f"Unknown format: {args.format}")
        return False

    # Save quantized model
    print(f"\nSaving quantized model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Add Titan metadata
    titan_meta = {
        "titan_version": "0.1.0",
        "source_model": model_path,
        "quantization": args.format,
        "calibration_samples": args.calib_size,
        "calibration_dataset": args.calib_data,
        "target_gpu": "blackwell",
    }
    with open(os.path.join(output_path, "titan_meta.json"), 'w') as f:
        json.dump(titan_meta, f, indent=2)

    # Calculate size reduction
    original_size = sum(
        os.path.getsize(os.path.join(model_path, f))
        for f in os.listdir(model_path)
        if f.endswith('.safetensors') or f.endswith('.bin')
    )
    output_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith('.safetensors') or f.endswith('.bin')
    )

    print(f"\nConversion complete!")
    print(f"  Original size: {original_size / 1e9:.1f} GB (BF16)")
    print(f"  Output size:   {output_size / 1e9:.1f} GB ({args.format})")
    print(f"  Compression:   {original_size / max(output_size, 1):.1f}x")
    print(f"\nRun with Titan:")
    print(f"  ./titan -m {output_path} -q fp4")

    return True


def build_trt_engine(args):
    """Build a TensorRT-LLM engine from the quantized checkpoint."""
    print("\nBuilding TensorRT-LLM engine...")
    print("  This requires trtllm-build from the TensorRT-LLM package.")

    # Check if trtllm-build is available
    trtllm = shutil.which("trtllm-build")
    if not trtllm:
        print("ERROR: trtllm-build not found.")
        print("Install TensorRT-LLM: pip install tensorrt-llm")
        return False

    engine_dir = args.output + "_engine"
    os.makedirs(engine_dir, exist_ok=True)

    cmd = [
        "trtllm-build",
        "--checkpoint_dir", args.output,
        "--output_dir", engine_dir,
        "--gemm_plugin", "auto",
        "--max_batch_size", "1",
        "--max_input_len", "4096",
        "--max_seq_len", "8192",
    ]

    print(f"  Command: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"\nEngine built at: {engine_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Engine build failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Titan Engine — NVIDIA FP4 Conversion for Blackwell GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Kimi K2.5 BF16 to NVFP4
  python convert_nvfp4.py --model ./Kimi-K2.5 --output ./Kimi-K2.5-nvfp4

  # Convert to MXFP4 format
  python convert_nvfp4.py --model ./Kimi-K2.5 --format mxfp4 --output ./Kimi-K2.5-mxfp4

  # Convert and build TensorRT engine
  python convert_nvfp4.py --model ./Kimi-K2.5 --output ./Kimi-K2.5-nvfp4 --build-engine

  # Use NVIDIA's pre-quantized Kimi instead
  huggingface-cli download nvidia/Kimi-K2-Thinking-NVFP4 --local-dir ./Kimi-K2-NVFP4

Reference:
  - nvidia/Kimi-K2-Thinking-NVFP4 on HuggingFace (pre-quantized reference)
  - NVIDIA TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer
  - FP4 is supported on Blackwell-class GPUs (RTX 5090, B100, B200, GB200)
        """
    )

    parser.add_argument("--model", required=True, help="Path to BF16 model directory")
    parser.add_argument("--output", required=True, help="Output path for quantized model")
    parser.add_argument("--format", choices=["nvfp4", "mxfp4"], default="nvfp4",
                        help="Quantization format (default: nvfp4)")
    parser.add_argument("--calib-data", default="simple",
                        choices=["simple", "wikitext"],
                        help="Calibration dataset (default: simple)")
    parser.add_argument("--calib-size", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--build-engine", action="store_true",
                        help="Also build TensorRT-LLM engine")
    parser.add_argument("--install-deps", action="store_true",
                        help="Install NVIDIA Model Optimizer if not found")

    args = parser.parse_args()

    # Check dependencies
    deps = check_dependencies()

    print("Titan Engine — NVIDIA FP4 Conversion")
    print(f"  PyTorch:       {'OK' if deps.get('torch') else 'MISSING'}")
    print(f"  Model Optimizer: {'OK' if deps.get('modelopt') else 'MISSING'}")
    print(f"  Transformers:  {'OK' if deps.get('transformers') else 'MISSING'}")
    if deps.get('gpu_name'):
        print(f"  GPU:           {deps['gpu_name']} ({deps.get('gpu_memory_gb', 0):.0f} GB)")
    print()

    if not deps.get("torch"):
        print("ERROR: PyTorch is required. Install: pip install torch")
        sys.exit(1)

    if not deps.get("modelopt"):
        if args.install_deps:
            if not install_modelopt():
                print("ERROR: Failed to install NVIDIA Model Optimizer")
                sys.exit(1)
        else:
            print("ERROR: NVIDIA Model Optimizer not found.")
            print("Install: pip install nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com")
            print("Or run with --install-deps to auto-install")
            sys.exit(1)

    if not deps.get("transformers"):
        print("ERROR: transformers is required. Install: pip install transformers")
        sys.exit(1)

    # Run conversion
    success = convert_nvfp4(args)
    if not success:
        sys.exit(1)

    # Optionally build engine
    if args.build_engine:
        build_trt_engine(args)


if __name__ == "__main__":
    main()
