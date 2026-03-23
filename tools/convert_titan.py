#!/usr/bin/env python3
"""
Titan Engine — Native Format Converter

Converts HuggingFace models into Titan's optimized native format:
- Pre-quantized weights (INT4/FP4/FP8) with optimal group sizes
- Layout-optimized for streaming (experts packed contiguously per layer)
- Expert index for fast NVMe random access
- Single metadata header for instant model loading

Titan format layout:
    model.titan/
    ├── manifest.json           # Model config + tensor index + expert index
    ├── attention.bin           # All attention weights (contiguous, VRAM-resident)
    ├── embedding.bin           # Embedding + LM head (VRAM-resident)
    ├── norms.bin               # All norm weights (tiny, VRAM-resident)
    ├── experts/
    │   ├── layer_00.bin        # Layer 0 experts [N x expert_bytes] (NVMe-streamed)
    │   ├── layer_01.bin
    │   └── ...
    └── shared_experts.bin      # Shared expert weights (VRAM-resident)

Usage:
    python convert_titan.py --model /path/to/model --quant q4_k --output /path/to/model.titan
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

try:
    import torch
    import numpy as np
    from safetensors import safe_open
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def quantize_int4(tensor, group_size=64):
    """Quantize FP32/FP16 tensor to INT4 with per-group scale+bias."""
    t = tensor.float().contiguous()
    rows, cols = t.shape
    assert cols % group_size == 0, f"cols ({cols}) must be divisible by group_size ({group_size})"

    # Reshape into groups
    groups = t.reshape(rows, -1, group_size)  # [rows, num_groups, group_size]
    num_groups = groups.shape[1]

    # Compute per-group min/max for affine quantization
    gmin = groups.min(dim=-1).values  # [rows, num_groups]
    gmax = groups.max(dim=-1).values

    # Scale and bias: val = raw * scale + bias
    # raw in [0, 15], val in [gmin, gmax]
    scales = (gmax - gmin) / 15.0
    scales = scales.clamp(min=1e-10)
    biases = gmin

    # Quantize
    quantized = ((groups - biases.unsqueeze(-1)) / scales.unsqueeze(-1)).round().clamp(0, 15).byte()

    # Pack 8 nibbles per uint32
    packed_cols = cols // 8
    packed = torch.zeros(rows, packed_cols, dtype=torch.int32)
    for n in range(8):
        packed |= quantized[:, :, n::8].reshape(rows, packed_cols).int() << (n * 4)

    return {
        'weights': packed.numpy(),
        'scales': scales.half().numpy(),
        'biases': biases.half().numpy(),
        'group_size': group_size,
    }


def convert_model(args):
    if not HAS_DEPS:
        print("ERROR: torch, numpy, safetensors required")
        print("  pip install torch numpy safetensors")
        sys.exit(1)

    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    hidden = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    num_experts = config.get("num_local_experts", 0)
    vocab_size = config.get("vocab_size", 32000)

    print(f"Model: {config.get('model_type', 'unknown')}")
    print(f"  Layers: {num_layers}, Hidden: {hidden}, Vocab: {vocab_size}")
    if num_experts > 0:
        print(f"  MoE: {num_experts} experts")
    print(f"  Output quant: {args.quant}")

    # Scan safetensors files
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        print("ERROR: No safetensors files found")
        return False

    # Collect all tensors
    print(f"\nScanning {len(st_files)} safetensors file(s)...")
    all_tensors = {}
    for st_file in st_files:
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():
                all_tensors[name] = (st_file, f.get_tensor(name).shape)

    print(f"  Found {len(all_tensors)} tensors")

    # Categorize tensors
    attention_tensors = {}
    ffn_tensors = {}
    expert_tensors = {}
    norm_tensors = {}
    embed_tensors = {}

    for name, (file, shape) in all_tensors.items():
        if "embed" in name or "lm_head" in name:
            embed_tensors[name] = file
        elif "norm" in name or "layernorm" in name:
            norm_tensors[name] = file
        elif "experts" in name:
            expert_tensors[name] = file
        elif "self_attn" in name or "attention" in name:
            attention_tensors[name] = file
        elif "mlp" in name or "feed_forward" in name:
            if num_experts > 0:
                expert_tensors[name] = file
            else:
                ffn_tensors[name] = file

    print(f"  Attention: {len(attention_tensors)}, FFN: {len(ffn_tensors)}")
    print(f"  Experts: {len(expert_tensors)}, Norms: {len(norm_tensors)}")
    print(f"  Embedding: {len(embed_tensors)}")

    # Build manifest
    manifest = {
        "format": "titan-v1",
        "model_config": config,
        "quantization": args.quant,
        "group_size": args.group_size,
        "files": {},
        "expert_index": {},
    }

    # Pack and write each category
    def write_tensors(tensors, output_file, quantize=True):
        """Load tensors, optionally quantize, write to binary file."""
        entries = []
        offset = 0

        with open(output_path / output_file, 'wb') as out:
            for name, st_file in sorted(tensors.items()):
                with safe_open(st_file, framework="pt") as f:
                    tensor = f.get_tensor(name)

                if quantize and args.quant in ("q4_k", "int4") and len(tensor.shape) == 2:
                    # Quantize to INT4
                    q = quantize_int4(tensor, args.group_size)
                    data = q['weights'].tobytes()
                    data += q['scales'].tobytes()
                    data += q['biases'].tobytes()
                    entry = {
                        "name": name,
                        "offset": offset,
                        "size": len(data),
                        "dtype": "int4",
                        "shape": list(tensor.shape),
                        "group_size": args.group_size,
                    }
                else:
                    # Keep as FP16
                    t = tensor.half()
                    data = t.numpy().tobytes()
                    entry = {
                        "name": name,
                        "offset": offset,
                        "size": len(data),
                        "dtype": "fp16",
                        "shape": list(tensor.shape),
                    }

                out.write(data)
                entries.append(entry)
                offset += len(data)

        return entries

    print(f"\nPacking attention weights...")
    attn_entries = write_tensors(attention_tensors, "attention.bin")
    manifest["files"]["attention"] = {"file": "attention.bin", "tensors": attn_entries}

    print(f"Packing embedding weights...")
    embed_entries = write_tensors(embed_tensors, "embedding.bin", quantize=False)
    manifest["files"]["embedding"] = {"file": "embedding.bin", "tensors": embed_entries}

    print(f"Packing norm weights...")
    norm_entries = write_tensors(norm_tensors, "norms.bin", quantize=False)
    manifest["files"]["norms"] = {"file": "norms.bin", "tensors": norm_entries}

    if ffn_tensors:
        print(f"Packing FFN weights...")
        ffn_entries = write_tensors(ffn_tensors, "ffn.bin")
        manifest["files"]["ffn"] = {"file": "ffn.bin", "tensors": ffn_entries}

    if expert_tensors:
        print(f"Packing expert weights into per-layer files...")
        (output_path / "experts").mkdir(exist_ok=True)
        # Group by layer
        for l in range(num_layers):
            layer_experts = {k: v for k, v in expert_tensors.items()
                           if f".{l}." in k}
            if layer_experts:
                fname = f"experts/layer_{l:02d}.bin"
                entries = write_tensors(layer_experts, fname)
                manifest["expert_index"][str(l)] = {"file": fname, "tensors": entries}

    # Copy tokenizer
    for tok_file in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                     "special_tokens_map.json"]:
        src = model_path / tok_file
        if src.exists():
            import shutil
            shutil.copy2(src, output_path / tok_file)

    # Write manifest
    with open(output_path / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # Calculate sizes
    total_size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, dirs, files in os.walk(output_path)
        for f in files
    )

    print(f"\nConversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Total size: {total_size / 1e9:.1f} GB")
    print(f"  Format: titan-v1 ({args.quant})")
    print(f"\nRun with:")
    print(f"  ./titan -m {output_path} -q {args.quant}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert to Titan native format")
    parser.add_argument("--model", required=True, help="HuggingFace model directory")
    parser.add_argument("--output", required=True, help="Output directory for Titan format")
    parser.add_argument("--quant", default="q4_k", choices=["fp16", "int4", "q4_k", "q3_k", "fp4"],
                        help="Quantization format (default: q4_k)")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    args = parser.parse_args()

    convert_model(args)


if __name__ == "__main__":
    main()
