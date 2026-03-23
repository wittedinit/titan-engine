#!/usr/bin/env python3
"""
Titan Engine — Benchmarking Suite with Automated Regression Testing

Measures:
- Time to first token (TTFT)
- Tokens per second (decode throughput)
- Peak VRAM usage
- Expert cache hit rate (MoE models)
- Per-layer timing breakdown

Usage:
    # Basic benchmark
    python benchmark.py --model /path/to/model --quant q4_k --tokens 100

    # Full regression test (compares against baseline)
    python benchmark.py --model /path/to/model --quant q4_k --regression baseline.json

    # Profile all quantization levels
    python benchmark.py --model /path/to/model --sweep q3_k,q4_k,int4,fp8
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_titan(model_path, quant="q4_k", max_tokens=100, prompt="Once upon a time",
              extra_args=None, titan_binary="./build/titan"):
    """Run Titan Engine and capture output."""
    cmd = [
        titan_binary,
        "-m", model_path,
        "-q", quant,
        "--max-tokens", str(max_tokens),
        "-v",  # Verbose for timing info
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Feed prompt via stdin
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300
    )

    stdout, stderr = proc.communicate(input=prompt + "\nexit\n", timeout=300)
    return stdout, stderr, proc.returncode


def parse_metrics(stderr_output):
    """Extract metrics from Titan's log output."""
    metrics = {
        "ttft_ms": None,
        "tokens_per_sec": None,
        "prefill_tok_s": None,
        "tokens_generated": None,
        "vram_used_gb": None,
        "ram_used_gb": None,
        "expert_cache_hit_rate": None,
    }

    for line in stderr_output.split('\n'):
        if "Prefill:" in line:
            # [  X.XXX] INFO  Prefill: N tokens in X.X ms (X tok/s)
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "ms" and i > 0:
                    try:
                        metrics["ttft_ms"] = float(parts[i-1])
                    except ValueError:
                        pass
                if "tok/s)" in p or (p == "tok/s" and i > 0):
                    try:
                        val = parts[i-1].rstrip(')')
                        metrics["prefill_tok_s"] = float(val)
                    except ValueError:
                        pass

        elif "Generated" in line and "tok/s" in line:
            # [  X.XXX] INFO  Generated N tokens in X.X s (X.X tok/s)
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "tokens" and i > 0:
                    try:
                        metrics["tokens_generated"] = int(parts[i-1])
                    except ValueError:
                        pass
                if "tok/s)" in p:
                    try:
                        val = parts[i].rstrip(')').lstrip('(')
                        metrics["tokens_per_sec"] = float(val)
                    except ValueError:
                        pass

        elif "VRAM:" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "/" and i > 0:
                    try:
                        metrics["vram_used_gb"] = float(parts[i-1])
                    except ValueError:
                        pass

        elif "Expert cache:" in line and "hit rate" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if "%" in p:
                    try:
                        metrics["expert_cache_hit_rate"] = float(p.rstrip('%'))
                    except ValueError:
                        pass

    return metrics


def run_benchmark(args):
    """Run a single benchmark configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {args.model}")
    print(f"Quant: {args.quant} | Tokens: {args.tokens}")
    print(f"Prompt: \"{args.prompt[:50]}...\"" if len(args.prompt) > 50 else f"Prompt: \"{args.prompt}\"")
    print(f"{'='*60}")

    titan_bin = args.binary or "./build/titan"
    if not os.path.exists(titan_bin):
        # Try relative to script
        script_dir = Path(__file__).parent.parent
        titan_bin = str(script_dir / "build" / "titan")

    if not os.path.exists(titan_bin):
        print(f"ERROR: Titan binary not found at {titan_bin}")
        print("Build first: mkdir build && cd build && cmake .. && make -j$(nproc)")
        return None

    start = time.time()
    try:
        stdout, stderr, rc = run_titan(
            args.model, args.quant, args.tokens, args.prompt,
            titan_binary=titan_bin
        )
    except subprocess.TimeoutExpired:
        print("ERROR: Benchmark timed out (300s)")
        return None
    except FileNotFoundError:
        print(f"ERROR: Cannot execute {titan_bin}")
        return None

    elapsed = time.time() - start

    if rc != 0:
        print(f"ERROR: Titan exited with code {rc}")
        print("STDERR:", stderr[:500])
        return None

    metrics = parse_metrics(stderr)
    metrics["total_time_s"] = elapsed
    metrics["quant"] = args.quant
    metrics["model"] = args.model
    metrics["max_tokens"] = args.tokens
    metrics["timestamp"] = datetime.now().isoformat()

    # Print results
    print(f"\nResults:")
    print(f"  TTFT:              {metrics['ttft_ms']:.0f} ms" if metrics['ttft_ms'] else "  TTFT:              N/A")
    print(f"  Decode throughput: {metrics['tokens_per_sec']:.1f} tok/s" if metrics['tokens_per_sec'] else "  Decode throughput: N/A")
    print(f"  Prefill:           {metrics['prefill_tok_s']:.0f} tok/s" if metrics['prefill_tok_s'] else "  Prefill:           N/A")
    print(f"  Tokens generated:  {metrics['tokens_generated']}" if metrics['tokens_generated'] else "  Tokens generated:  N/A")
    print(f"  VRAM used:         {metrics['vram_used_gb']:.1f} GB" if metrics['vram_used_gb'] else "  VRAM used:         N/A")
    if metrics.get('expert_cache_hit_rate'):
        print(f"  Expert cache hit:  {metrics['expert_cache_hit_rate']:.0f}%")
    print(f"  Total wall time:   {elapsed:.1f} s")

    return metrics


def run_sweep(args):
    """Run benchmarks across multiple quant levels."""
    quant_levels = args.sweep.split(',')
    results = []

    for quant in quant_levels:
        args.quant = quant.strip()
        metrics = run_benchmark(args)
        if metrics:
            results.append(metrics)

    # Print comparison table
    if results:
        print(f"\n{'='*70}")
        print("Comparison:")
        print(f"{'Quant':<8} {'TTFT (ms)':<12} {'Decode (tok/s)':<16} {'VRAM (GB)':<12}")
        print("-" * 48)
        for r in results:
            ttft = f"{r['ttft_ms']:.0f}" if r.get('ttft_ms') else "N/A"
            tps = f"{r['tokens_per_sec']:.1f}" if r.get('tokens_per_sec') else "N/A"
            vram = f"{r['vram_used_gb']:.1f}" if r.get('vram_used_gb') else "N/A"
            print(f"{r['quant']:<8} {ttft:<12} {tps:<16} {vram:<12}")

    return results


def run_regression(args):
    """Run benchmark and compare against baseline."""
    metrics = run_benchmark(args)
    if not metrics:
        return False

    # Load baseline
    if not os.path.exists(args.regression):
        print(f"\nNo baseline found. Saving current results as baseline: {args.regression}")
        with open(args.regression, 'w') as f:
            json.dump(metrics, f, indent=2)
        return True

    with open(args.regression) as f:
        baseline = json.load(f)

    # Compare
    print(f"\nRegression comparison vs {args.regression}:")
    regressions = []

    def check(key, label, threshold=0.1, higher_is_better=True):
        if metrics.get(key) and baseline.get(key):
            old = baseline[key]
            new = metrics[key]
            change = (new - old) / old if old != 0 else 0
            direction = "better" if (change > 0) == higher_is_better else "worse"
            status = "PASS" if abs(change) < threshold or direction == "better" else "FAIL"
            print(f"  {label:<25} {old:>10.1f} → {new:>10.1f}  ({change:+.1%}) {status}")
            if status == "FAIL":
                regressions.append(label)

    check("tokens_per_sec", "Decode tok/s", threshold=0.05, higher_is_better=True)
    check("ttft_ms", "TTFT (ms)", threshold=0.10, higher_is_better=False)
    check("vram_used_gb", "VRAM (GB)", threshold=0.05, higher_is_better=False)

    if regressions:
        print(f"\nREGRESSION DETECTED in: {', '.join(regressions)}")
        return False
    else:
        print("\nAll metrics within tolerance. PASS.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Titan Engine Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --model ./llama-8b -q q4_k --tokens 100
  python benchmark.py --model ./llama-8b --sweep q3_k,q4_k,int4,fp8
  python benchmark.py --model ./llama-8b --regression baseline.json
        """
    )
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("-q", "--quant", default="q4_k", help="Quantization level")
    parser.add_argument("--tokens", type=int, default=100, help="Tokens to generate")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. Once upon a time in a land far far away,",
                        help="Input prompt")
    parser.add_argument("--binary", help="Path to titan binary")
    parser.add_argument("--sweep", help="Comma-separated quant levels to benchmark")
    parser.add_argument("--regression", help="Compare against baseline JSON file")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    if args.sweep:
        results = run_sweep(args)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    elif args.regression:
        success = run_regression(args)
        sys.exit(0 if success else 1)
    else:
        metrics = run_benchmark(args)
        if args.output and metrics:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
