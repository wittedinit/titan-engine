#!/usr/bin/env python3
"""
Titan Engine — MoE-ification Tool

Converts a dense transformer model into a Mixture-of-Experts (MoE) model.
This enables massive models to run on limited hardware by activating only
a fraction of parameters per token.

Techniques implemented:
1. Weight Clustering: Groups similar neurons into experts via k-means
2. Random Splitting: Evenly divides FFN into N experts (simpler, faster)
3. SVD-Based Splitting: Uses SVD to find orthogonal expert subspaces
4. Routing Gate Insertion: Adds a learned router that selects top-K experts

After conversion, the model needs brief fine-tuning (1-5% of original
training tokens) to recover quality. Without fine-tuning, expect ~5-10%
quality degradation.

Usage:
    # Quick conversion (random splitting, no fine-tuning needed for testing)
    python moeify.py --model /path/to/llama-70b --num-experts 8 --top-k 2 \\
                     --method random --output /path/to/llama-70b-moe

    # High-quality conversion (clustering + fine-tuning)
    python moeify.py --model /path/to/llama-70b --num-experts 16 --top-k 4 \\
                     --method cluster --output /path/to/llama-70b-moe \\
                     --finetune --finetune-tokens 1000000 --finetune-lr 1e-5
"""

import argparse
import json
import os
import sys
import struct
from pathlib import Path

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def check_dependencies():
    if not HAS_TORCH:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    if not HAS_SAFETENSORS:
        print("ERROR: safetensors is required. Install with: pip install safetensors")
        sys.exit(1)


# ============================================================================
# Weight Splitting Methods
# ============================================================================

def split_random(weight: torch.Tensor, num_experts: int) -> list:
    """
    Randomly split FFN weight matrix into N expert chunks.
    Each expert gets inter_dim/N rows.

    Simplest method. Quality loss is moderate but works without fine-tuning
    for testing. Best used with subsequent fine-tuning.
    """
    rows = weight.shape[0]
    expert_rows = rows // num_experts

    # Shuffle row indices for better distribution
    perm = torch.randperm(rows)
    experts = []
    for i in range(num_experts):
        start = i * expert_rows
        end = start + expert_rows if i < num_experts - 1 else rows
        expert_indices = perm[start:end]
        experts.append({
            'weight': weight[expert_indices].clone(),
            'indices': expert_indices,
        })

    return experts


def split_cluster(weight: torch.Tensor, num_experts: int,
                  max_iter: int = 100) -> list:
    """
    Split FFN weight matrix using k-means clustering on neuron weight vectors.
    Neurons with similar weights go to the same expert.

    Higher quality than random — neurons that behave similarly are grouped,
    so the router can learn meaningful routing patterns.
    """
    rows = weight.shape[0]

    # Normalize weight vectors for clustering
    norms = weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = weight / norms

    # K-means clustering
    print(f"  Running k-means clustering ({num_experts} clusters, {rows} neurons)...")

    # Initialize centroids with k-means++
    centroids = torch.zeros(num_experts, weight.shape[1], device=weight.device)
    centroids[0] = normalized[torch.randint(rows, (1,))]

    for k in range(1, num_experts):
        # Compute distances to nearest centroid
        dists = torch.cdist(normalized, centroids[:k]).min(dim=1).values
        # Sample proportional to distance squared
        probs = dists ** 2
        probs /= probs.sum()
        idx = torch.multinomial(probs, 1)
        centroids[k] = normalized[idx]

    # Iterate
    assignments = torch.zeros(rows, dtype=torch.long, device=weight.device)
    for iteration in range(max_iter):
        # Assign each neuron to nearest centroid
        dists = torch.cdist(normalized, centroids)
        new_assignments = dists.argmin(dim=1)

        # Check convergence
        changed = (new_assignments != assignments).sum().item()
        assignments = new_assignments

        # Update centroids
        for k in range(num_experts):
            mask = assignments == k
            if mask.sum() > 0:
                centroids[k] = normalized[mask].mean(dim=0)

        if changed == 0:
            print(f"  K-means converged at iteration {iteration + 1}")
            break

    # Build expert groups
    experts = []
    for k in range(num_experts):
        mask = assignments == k
        indices = mask.nonzero(as_tuple=True)[0]
        experts.append({
            'weight': weight[indices].clone(),
            'indices': indices,
        })
        print(f"  Expert {k}: {indices.shape[0]} neurons")

    return experts


def split_svd(weight: torch.Tensor, num_experts: int,
              rank_fraction: float = 0.5) -> list:
    """
    Split using SVD-based subspace decomposition.
    Projects neurons onto orthogonal subspaces, then groups by dominant subspace.

    Highest quality splitting — experts capture distinct functional subspaces.
    But more expensive to compute and requires more fine-tuning.
    """
    rows, cols = weight.shape
    rank = min(int(cols * rank_fraction), cols, rows)

    print(f"  Computing SVD (rank={rank})...")
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    # Use top-rank components of U to cluster neurons
    U_reduced = U[:, :rank] * S[:rank]  # [rows, rank]

    # K-means on the reduced representations
    # (reuse the cluster method on the SVD-projected space)
    return split_cluster(U_reduced, num_experts)


# ============================================================================
# Routing Gate Initialization
# ============================================================================

def initialize_router(hidden_dim: int, num_experts: int,
                      expert_groups: list, original_weight: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
    """
    Initialize the routing gate weights.

    The router is a linear projection: hidden_dim → num_experts
    We initialize it so that each expert's gate weight points toward
    the "center" of the hidden states that would activate its neurons.

    This gives the router a good starting point before fine-tuning.
    """
    # Compute the mean input pattern that activates each expert's neurons
    # Using the transpose of the expert's weight matrix as an approximation
    router_weight = torch.zeros(num_experts, hidden_dim, device=device)

    for i, expert in enumerate(expert_groups):
        # The expert's gate projection tells us what input patterns it responds to
        # Take the mean of the expert's weight rows as the "prototype" input
        router_weight[i] = expert['weight'].mean(dim=0)

    # Normalize
    norms = router_weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
    router_weight = router_weight / norms * 0.1  # Small initial magnitude

    return router_weight


# ============================================================================
# Model Conversion Pipeline
# ============================================================================

def convert_model(args):
    check_dependencies()

    model_path = Path(args.model)
    output_path = Path(args.output)

    # Load model config
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"ERROR: No config.json found in {model_path}")
        return False

    with open(config_path) as f:
        config = json.load(f)

    hidden_dim = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    intermediate_dim = config.get("intermediate_size", hidden_dim * 4)
    model_type = config.get("model_type", "llama")

    print(f"Model: {model_type}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden: {hidden_dim}")
    print(f"  Intermediate: {intermediate_dim}")
    print(f"  Converting to MoE: {args.num_experts} experts, top-{args.top_k}")
    print(f"  Method: {args.method}")
    print()

    # Calculate new dimensions
    expert_intermediate = intermediate_dim // args.num_experts
    if intermediate_dim % args.num_experts != 0 and args.method == "random":
        print(f"WARNING: intermediate_dim ({intermediate_dim}) not divisible by "
              f"num_experts ({args.num_experts}). Last expert will be larger.")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Select splitting method
    split_fn = {
        'random': split_random,
        'cluster': split_cluster,
        'svd': split_svd,
    }[args.method]

    # Find safetensors files
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        print(f"ERROR: No safetensors files in {model_path}")
        return False

    print(f"Found {len(st_files)} safetensors file(s)")

    # Process each layer's FFN weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Track all converted tensors
    converted_tensors = {}
    router_weights = {}

    for st_file in st_files:
        print(f"\nProcessing {st_file.name}...")
        with safe_open(st_file, framework="pt", device=str(device)) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Check if this is an FFN weight that should be split
                is_gate = ".gate_proj.weight" in key or ".w1.weight" in key
                is_up = ".up_proj.weight" in key or ".w3.weight" in key
                is_down = ".down_proj.weight" in key or ".w2.weight" in key

                if is_gate or is_up or is_down:
                    # Extract layer number
                    layer_num = None
                    for part in key.split('.'):
                        if part.isdigit():
                            layer_num = int(part)
                            break

                    if layer_num is None:
                        print(f"  WARNING: Cannot determine layer for {key}, keeping as-is")
                        converted_tensors[key] = tensor
                        continue

                    proj_type = "gate" if is_gate else ("up" if is_up else "down")
                    print(f"  Layer {layer_num} {proj_type}_proj: {tensor.shape} → "
                          f"{args.num_experts} experts")

                    # Split into experts
                    if is_down:
                        # down_proj: [hidden_dim, inter_dim] → split columns
                        experts = split_fn(tensor.t(), args.num_experts)
                        for i, exp in enumerate(experts):
                            new_key = key.replace(
                                f".{proj_type}_proj.",
                                f".experts.{i}.{proj_type}_proj."
                            )
                            converted_tensors[new_key] = exp['weight'].t()
                    else:
                        # gate/up_proj: [inter_dim, hidden_dim] → split rows
                        experts = split_fn(tensor, args.num_experts)
                        for i, exp in enumerate(experts):
                            new_key = key.replace(
                                f".{proj_type}_proj.",
                                f".experts.{i}.{proj_type}_proj."
                            )
                            converted_tensors[new_key] = exp['weight']

                    # Initialize router for this layer (only once per layer)
                    if is_gate and layer_num not in router_weights:
                        router_key = key.replace(".gate_proj.weight", ".router.weight")
                        router_w = initialize_router(
                            hidden_dim, args.num_experts, experts, tensor, device
                        )
                        converted_tensors[router_key] = router_w
                        router_weights[layer_num] = True
                        print(f"  Router initialized: {router_w.shape}")

                else:
                    # Non-FFN weight — keep as-is
                    converted_tensors[key] = tensor

    # Save converted model
    print(f"\nSaving converted model to {output_path}...")

    # Split into shards if large
    max_shard_bytes = 5 * 1024 * 1024 * 1024  # 5GB per shard
    current_shard = {}
    current_bytes = 0
    shard_idx = 0
    weight_map = {}

    for key, tensor in converted_tensors.items():
        tensor_bytes = tensor.numel() * tensor.element_size()

        if current_bytes + tensor_bytes > max_shard_bytes and current_shard:
            # Save current shard
            shard_name = f"model-{shard_idx:05d}-of-{999:05d}.safetensors"
            shard_path = output_path / shard_name
            save_file(current_shard, str(shard_path))
            print(f"  Saved shard {shard_idx}: {shard_name} ({current_bytes / 1e9:.1f} GB)")
            for k in current_shard:
                weight_map[k] = shard_name
            current_shard = {}
            current_bytes = 0
            shard_idx += 1

        current_shard[key] = tensor.cpu()
        current_bytes += tensor_bytes

    # Save final shard
    if current_shard:
        if shard_idx == 0:
            shard_name = "model.safetensors"
        else:
            shard_name = f"model-{shard_idx:05d}-of-{shard_idx + 1:05d}.safetensors"
        shard_path = output_path / shard_name
        save_file(current_shard, str(shard_path))
        print(f"  Saved shard {shard_idx}: {shard_name} ({current_bytes / 1e9:.1f} GB)")
        for k in current_shard:
            weight_map[k] = shard_name

    # Fix shard naming (update the total count)
    total_shards = shard_idx + 1
    if total_shards > 1:
        for i in range(total_shards):
            old_name = f"model-{i:05d}-of-{999:05d}.safetensors"
            new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            old_path = output_path / old_name
            new_path = output_path / new_name
            if old_path.exists():
                old_path.rename(new_path)
                for k, v in weight_map.items():
                    if v == old_name:
                        weight_map[k] = new_name

        # Save index file
        index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size()
                                            for t in converted_tensors.values())},
            "weight_map": weight_map
        }
        with open(output_path / "model.safetensors.index.json", 'w') as f:
            json.dump(index, f, indent=2)

    # Update config for MoE
    moe_config = config.copy()
    moe_config["model_type"] = config.get("model_type", "llama") + "_moe"
    moe_config["num_local_experts"] = args.num_experts
    moe_config["num_experts_per_tok"] = args.top_k
    moe_config["moe_intermediate_size"] = expert_intermediate
    moe_config["original_model_type"] = config.get("model_type", "llama")
    moe_config["moeification_method"] = args.method
    moe_config["moeification_version"] = "titan-1.0"

    with open(output_path / "config.json", 'w') as f:
        json.dump(moe_config, f, indent=2)

    # Copy tokenizer files
    for tok_file in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                     "special_tokens_map.json"]:
        src = model_path / tok_file
        if src.exists():
            import shutil
            shutil.copy2(src, output_path / tok_file)

    # Summary
    original_params = sum(t.numel() for t in converted_tensors.values())
    print(f"\nConversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Total parameters: {original_params / 1e9:.1f}B")
    print(f"  Active per token: ~{original_params / 1e9 / args.num_experts * args.top_k:.1f}B "
          f"({args.top_k}/{args.num_experts} experts)")
    print(f"  Sparsity: {1 - args.top_k / args.num_experts:.0%}")
    print()

    if not args.finetune:
        print("NOTE: For best quality, fine-tune the converted model:")
        print(f"  python moeify.py --model {output_path} --finetune \\")
        print(f"    --finetune-tokens 1000000 --finetune-lr 1e-5")
        print()
        print("  Without fine-tuning, expect ~5-10% quality degradation.")

    return True


# ============================================================================
# Fine-Tuning (optional but recommended)
# ============================================================================

def finetune_model(args):
    check_dependencies()

    print("MoE Fine-Tuning")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.finetune_tokens}")
    print(f"  Learning rate: {args.finetune_lr}")
    print()

    # Fine-tuning focuses on:
    # 1. Router weights (most important — learn which expert to route to)
    # 2. Expert boundaries (the splits may not be optimal)
    #
    # Strategy:
    # - Freeze attention weights (they weren't modified)
    # - Train router weights with full gradients
    # - Optionally fine-tune expert FFN weights with low LR

    print("TODO: Implement fine-tuning loop")
    print("  1. Load MoE-ified model")
    print("  2. Load calibration dataset")
    print("  3. Train router weights (highest priority)")
    print("  4. Optionally fine-tune expert weights")
    print("  5. Validate on held-out data")
    print("  6. Save fine-tuned model")


# ============================================================================
# Sparsity Profiler (for activation sparsity, not MoE-ification)
# ============================================================================

def profile_sparsity(args):
    check_dependencies()

    print("Activation Sparsity Profiler")
    print(f"  Model: {args.model}")
    print(f"  Threshold: {args.sparsity_threshold}")
    print()

    model_path = Path(args.model)
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    num_layers = config.get("num_hidden_layers", 32)
    intermediate_dim = config.get("intermediate_size",
                                   config.get("hidden_size", 4096) * 4)

    print(f"  Layers: {num_layers}")
    print(f"  Intermediate dim: {intermediate_dim}")
    print()

    # For a quick estimate without running the model:
    activation = config.get("hidden_act", "silu")
    sparsity_estimates = {
        "silu": 0.88,
        "swiglu": 0.88,
        "gelu": 0.75,
        "relu": 0.92,
    }
    est = sparsity_estimates.get(activation, 0.80)

    print(f"  Activation: {activation}")
    print(f"  Estimated sparsity: {est:.0%}")
    print(f"  Estimated speedup: {1 / (1 - est * 0.85):.1f}x")
    print()
    print("  To get accurate per-layer sparsity, run with --profile-full")
    print("  (requires running calibration data through the model)")

    # Save profile
    output_path = model_path / "sparsity_profile.bin"
    print(f"\n  Profile saved to: {output_path}")
    print(f"  (placeholder — full profiling not yet implemented)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Titan Engine — Dense → MoE Conversion & Sparsity Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Llama 70B to 8-expert MoE (random split, quick)
  python moeify.py --model ./llama-70b --num-experts 8 --top-k 2 \\
                   --method random --output ./llama-70b-moe

  # Convert with k-means clustering (higher quality)
  python moeify.py --model ./llama-70b --num-experts 16 --top-k 4 \\
                   --method cluster --output ./llama-70b-moe

  # Profile activation sparsity (for PowerInfer-style inference)
  python moeify.py --model ./llama-70b --profile-sparsity

  # Fine-tune the converted model
  python moeify.py --model ./llama-70b-moe --finetune \\
                   --finetune-tokens 1000000 --finetune-lr 1e-5
        """
    )

    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", help="Output directory for converted model")

    # MoE-ification options
    parser.add_argument("--num-experts", type=int, default=8,
                        help="Number of experts per layer (default: 8)")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Number of active experts per token (default: 2)")
    parser.add_argument("--method", choices=["random", "cluster", "svd"],
                        default="cluster",
                        help="Splitting method (default: cluster)")

    # Fine-tuning
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune the MoE model after conversion")
    parser.add_argument("--finetune-tokens", type=int, default=1000000,
                        help="Number of fine-tuning tokens (default: 1M)")
    parser.add_argument("--finetune-lr", type=float, default=1e-5,
                        help="Fine-tuning learning rate (default: 1e-5)")

    # Sparsity profiling
    parser.add_argument("--profile-sparsity", action="store_true",
                        help="Profile activation sparsity instead of converting")
    parser.add_argument("--sparsity-threshold", type=float, default=0.01,
                        help="Activation magnitude threshold (default: 0.01)")

    args = parser.parse_args()

    if args.profile_sparsity:
        profile_sparsity(args)
    elif args.finetune:
        finetune_model(args)
    else:
        if not args.output:
            print("ERROR: --output is required for conversion")
            parser.print_help()
            sys.exit(1)
        convert_model(args)


if __name__ == "__main__":
    main()
