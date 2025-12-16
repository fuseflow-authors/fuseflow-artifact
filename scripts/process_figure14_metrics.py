#!/usr/bin/env python3
"""
Process simulator output logs for Figure 14 metrics extraction.

Reads simulator output files saved by run_figure12_benchmarks.py with --output-dir
and extracts FLOPs and memory access counts for each configuration.

Figure 14 shows GCN fusion ablation with 5 configurations:
1. Fully Fused (gcn_sparse.mlir)
2. Only First Layer Fused (one_layer_gcn.mlir + unfused layer 2 ops)
3. Only Second Layer Fused (unfused layer 1 ops + one_layer_gcn2.mlir)
4. First and Second Layer Fused (one_layer_gcn.mlir + one_layer_gcn2.mlir) - this is "partially_fused"
5. End-to-end Unfused (all 8 unfused ops)

Usage:
    python3 scripts/process_figure14_metrics.py --input-dir <simulator_outputs_dir> --output figure14_metrics.json
"""

import argparse
import json
import os
from pathlib import Path


def extract_metrics_from_file(filepath):
    """Extract memory and compute counts from a simulator output file.

    Based on get_metrics.py logic.
    """
    mem_ops = ["Crd read count", "Crd write count", "Num reads", "Write count"]
    compute_ops = ["Reduce count", "Op count"]

    mem_count = 0
    op_count = 0

    if not os.path.exists(filepath):
        return None, None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith(tuple(mem_ops + compute_ops)):
            continue
        try:
            count = int(line.split(": ")[1])
            if line.startswith(tuple(mem_ops)):
                mem_count += count
            elif line.startswith(tuple(compute_ops)):
                op_count += count
        except (IndexError, ValueError):
            continue

    return mem_count, op_count


def sum_metrics_from_files(filepaths):
    """Sum metrics from multiple simulator output files."""
    total_mem = 0
    total_flops = 0

    for filepath in filepaths:
        mem, flops = extract_metrics_from_file(filepath)
        if mem is not None and flops is not None:
            total_mem += mem
            total_flops += flops
        else:
            print(f"  Warning: Could not extract metrics from {filepath}")
            return None, None

    return total_mem, total_flops


def process_gcn_metrics(input_dir, dataset):
    """Process GCN metrics for a single dataset.

    Returns dict with metrics for all 5 Figure 14 configurations.
    """
    base_dir = Path(input_dir) / 'gcn' / dataset

    results = {}

    # Configuration 1: Fully Fused
    fused_file = base_dir / 'fully_fused' / 'gcn_unfused_gcn_sparse.txt'
    mem, flops = extract_metrics_from_file(fused_file)
    results['fully_fused'] = {'mem': mem, 'flops': flops}
    print(f"  Fully Fused: mem={mem}, flops={flops}")

    # Configuration 5: End-to-end Unfused (all 8 ops)
    unfused_ops = [
        'gcn_adj_x1.txt',
        'gcn_linear1_mul.txt',
        'gcn_linear1_bias.txt',
        'gcn_relu.txt',
        'gcn_adj_x2.txt',
        'gcn_linear2_mul.txt',
        'gcn_linear2_bias.txt',
        'gcn_softmax.txt',
    ]
    unfused_files = [base_dir / 'unfused' / op for op in unfused_ops]
    mem, flops = sum_metrics_from_files(unfused_files)
    results['unfused'] = {'mem': mem, 'flops': flops}
    print(f"  Unfused: mem={mem}, flops={flops}")

    # Configuration 4: First and Second Layer Fused (partially fused)
    # This is one_layer_gcn.mlir + one_layer_gcn2.mlir
    partial_ops = [
        'one_layer_gcn.txt',
        'one_layer_gcn2.txt',
    ]
    partial_files = [base_dir / 'partially_fused' / op for op in partial_ops]
    mem, flops = sum_metrics_from_files(partial_files)
    results['partially_fused'] = {'mem': mem, 'flops': flops}
    print(f"  Partially Fused (Layer 1+2): mem={mem}, flops={flops}")

    # Configuration 2: Only First Layer Fused
    # one_layer_gcn.mlir (layer 1 fused) + unfused layer 2 ops
    layer1_fused_files = [base_dir / 'partially_fused' / 'one_layer_gcn.txt']
    layer2_unfused_ops = [
        'gcn_adj_x2.txt',
        'gcn_linear2_mul.txt',
        'gcn_linear2_bias.txt',
        'gcn_softmax.txt',
    ]
    layer2_unfused_files = [base_dir / 'unfused' / op for op in layer2_unfused_ops]
    mem, flops = sum_metrics_from_files(layer1_fused_files + layer2_unfused_files)
    results['only_first_layer_fused'] = {'mem': mem, 'flops': flops}
    print(f"  Only First Layer Fused: mem={mem}, flops={flops}")

    # Configuration 3: Only Second Layer Fused
    # unfused layer 1 ops + one_layer_gcn2.mlir (layer 2 fused)
    layer1_unfused_ops = [
        'gcn_adj_x1.txt',
        'gcn_linear1_mul.txt',
        'gcn_linear1_bias.txt',
        'gcn_relu.txt',
    ]
    layer1_unfused_files = [base_dir / 'unfused' / op for op in layer1_unfused_ops]
    layer2_fused_files = [base_dir / 'partially_fused' / 'one_layer_gcn2.txt']
    mem, flops = sum_metrics_from_files(layer1_unfused_files + layer2_fused_files)
    results['only_second_layer_fused'] = {'mem': mem, 'flops': flops}
    print(f"  Only Second Layer Fused: mem={mem}, flops={flops}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Process simulator outputs for Figure 14 metrics')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='Directory containing simulator output files (from run_figure12_benchmarks.py --output-dir)')
    parser.add_argument('--output', '-o', type=str, default='figure14_metrics.json',
                        help='Output JSON file for metrics (default: figure14_metrics.json)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['cora_ml', 'dblp', 'collab'],
                        help='GCN datasets to process (default: cora_ml dblp collab)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory {input_dir} does not exist")
        return 1

    all_metrics = {}

    for dataset in args.datasets:
        print(f"\nProcessing GCN dataset: {dataset}")
        metrics = process_gcn_metrics(input_dir, dataset)
        all_metrics[dataset] = metrics

    # Save to JSON
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_file}")

    # Print summary table
    print("\n" + "="*80)
    print("FIGURE 14 METRICS SUMMARY")
    print("="*80)

    config_names = [
        ('fully_fused', 'Fully Fused'),
        ('only_first_layer_fused', 'Only First Layer Fused'),
        ('only_second_layer_fused', 'Only Second Layer Fused'),
        ('partially_fused', 'First and Second Layer Fused'),
        ('unfused', 'End-to-end Unfused'),
    ]

    for dataset in args.datasets:
        print(f"\n{dataset.upper()}:")
        print(f"{'Configuration':<35} {'FLOPs':>15} {'Memory':>15} {'Normalized FLOPs':>18} {'Normalized Mem':>16}")
        print("-"*100)

        unfused_flops = all_metrics[dataset]['unfused']['flops']
        unfused_mem = all_metrics[dataset]['unfused']['mem']

        for config_key, config_name in config_names:
            flops = all_metrics[dataset][config_key]['flops']
            mem = all_metrics[dataset][config_key]['mem']

            if flops is not None and unfused_flops is not None and unfused_flops > 0:
                norm_flops = f"{flops / unfused_flops:.4f}"
            else:
                norm_flops = "N/A"

            if mem is not None and unfused_mem is not None and unfused_mem > 0:
                norm_mem = f"{mem / unfused_mem:.4f}"
            else:
                norm_mem = "N/A"

            flops_str = f"{flops:,}" if flops is not None else "N/A"
            mem_str = f"{mem:,}" if mem is not None else "N/A"

            print(f"{config_name:<35} {flops_str:>15} {mem_str:>15} {norm_flops:>18} {norm_mem:>16}")

    # Print in plotting script format
    print("\n" + "="*80)
    print("VALUES FOR PLOTTING SCRIPT (copy-paste ready)")
    print("="*80)

    for dataset in args.datasets:
        print(f"\n# {dataset}")

        # Order for plotting: fully_fused, only_first, only_second, partial, unfused
        flops_list = []
        mem_list = []
        for config_key, _ in config_names:
            flops_list.append(all_metrics[dataset][config_key]['flops'] or 0)
            mem_list.append(all_metrics[dataset][config_key]['mem'] or 0)

        print(f"flops_{dataset} = {flops_list}")
        print(f"mem_{dataset} = {mem_list}")

    return 0


if __name__ == '__main__':
    exit(main())
