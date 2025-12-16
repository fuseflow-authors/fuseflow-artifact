#!/usr/bin/env python3
"""
Figure 14 Plotting Script for FuseFlow Paper

Plots FLOPs vs Memory Accesses (operational intensity) for GCN fusion ablation.
Shows 5 configurations across multiple datasets with normalized values.

Configurations:
1. Fully Fused
2. Only First Layer Fused
3. Only Second Layer Fused
4. First and Second Layer Fused (Partially Fused)
5. End-to-end Unfused

Usage:
    python3 scripts/plot_figure14.py --json figure14_metrics.json --output figure14.pdf
"""

import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Try to use scienceplots style
USE_LATEX = False

try:
    import scienceplots
    if USE_LATEX:
        plt.style.use('science')
    else:
        plt.style.use(['science', 'no-latex'])
except Exception as e:
    print(f"Note: Could not load 'science' style ({e}). Using default style.")
    plt.style.use('seaborn-v0_8-whitegrid')

matplotlib.rcParams['text.usetex'] = USE_LATEX
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('xtick', labelsize=16)

color_pal = sns.color_palette(palette='tab20')
marker_sel = ("o", "s", "D", "v", "<", ">")


def load_metrics_from_json(json_path):
    """Load metrics from JSON file produced by process_figure14_metrics.py."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = {}
    for dataset, metrics in data.items():
        # Order: fully_fused, only_first_layer_fused, only_second_layer_fused, partially_fused, unfused
        flops = [
            metrics['fully_fused']['flops'],
            metrics['only_first_layer_fused']['flops'],
            metrics['only_second_layer_fused']['flops'],
            metrics['partially_fused']['flops'],
            metrics['unfused']['flops'],
        ]
        mem = [
            metrics['fully_fused']['mem'],
            metrics['only_first_layer_fused']['mem'],
            metrics['only_second_layer_fused']['mem'],
            metrics['partially_fused']['mem'],
            metrics['unfused']['mem'],
        ]
        result[dataset] = {'flops': flops, 'mem': mem}

    return result


def get_default_data():
    """Return default hardcoded data from the original plotting script."""
    return {
        'cora_ml': {
            'flops': [2477724772, 240329975, 292489512, 240409687, 292409800],
            'mem': [3422625009, 305549531, 493297493, 305123333, 493723691],
        },
        'dblp': {
            'flops': [9901253852, 841903789, 844201668, 842146612, 843958845],
            'mem': [13815099378, 1046957403, 1339169398, 1045331819, 1340794982],
        },
        'collab': {
            'flops': [25541873568, 1377303232, 1371642400, 1374944552, 1374001080],
            'mem': [34440972179, 1750798916, 2156020138, 1729099055, 2177719999],
        },
    }


def plot_figure14(data, output_path='figure14.pdf', datasets=None):
    """Generate the Figure 14 plot."""

    if datasets is None:
        datasets = ['cora_ml', 'dblp', 'collab']

    # Configuration names
    operation_names = [
        "Fully Fused",
        "Only First Layer Fused",
        "Only Second Layer Fused",
        "First and Second Layer Fused",
        "End-to-end Unfused"
    ]

    # Assign colors to categories
    colors = {
        "Fully Fused": color_pal[2],
        "Only First Layer Fused": color_pal[4],
        "Only Second Layer Fused": color_pal[5],
        "First and Second Layer Fused": color_pal[1],
        "End-to-end Unfused": color_pal[0],
    }

    markers = {
        "Fully Fused": marker_sel[0],
        "Only First Layer Fused": marker_sel[1],
        "Only Second Layer Fused": marker_sel[2],
        "First and Second Layer Fused": marker_sel[3],
        "End-to-end Unfused": marker_sel[4],
    }

    plt.rcParams['axes.axisbelow'] = True
    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 3), sharey=True, dpi=300)

    # Handle single dataset case
    if len(datasets) == 1:
        axes = [axes]

    # Prepare datasets for plotting
    plot_data = []
    for i, dataset in enumerate(datasets):
        if dataset in data:
            flops = data[dataset]['flops']
            mem = data[dataset]['mem']
            plot_data.append((flops, mem, axes[i], dataset))
        else:
            print(f"Warning: Dataset {dataset} not found in data")

    # Collect all normalized values for axis limits
    all_flops_norm = []
    all_mem_norm = []

    for flops, mem, ax, title in plot_data:
        # Unfused is the last element (index 4)
        unfused_flops = flops[4]
        unfused_mem = mem[4]

        for i, label in enumerate(operation_names):
            # Normalize by unfused values
            x_val = (mem[i] * 4) / (unfused_mem * 4)  # Normalized bytes (×4 for float32)
            y_val = flops[i] / unfused_flops  # Normalized FLOPs

            all_flops_norm.append(y_val)
            all_mem_norm.append(x_val)

            # Calculate slope line for operational intensity
            slope = y_val / x_val

            print(f"Dataset: {title}, operation: {label}, slope: {slope:.4f}")

            # Plot a line representing the slope for each point
            x_vals = np.array([x_val / 2000, x_val * 20000])
            y_vals = slope * x_vals
            ax.plot(x_vals, y_vals, linestyle="--", color=colors[label], linewidth=0.7)

            # Plot scatter point
            ax.scatter(
                x_val, y_val,
                color=colors[label], s=150, edgecolor='black',
                marker=markers[label], label=label if label in colors else ""
            )

        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)
        ax.set_xlabel("Normalized Bytes Transferred", fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.set_axisbelow(True)

    # Set y-axis label and limits
    axes[0].set_ylabel("Normalized # FLOPs", fontsize=18)
    axes[0].set_ylim([min(all_flops_norm) / 1.5, max(all_flops_norm) * 1.4])

    # Set x-axis limits per subplot
    for i, (flops, mem, ax, title) in enumerate(plot_data):
        unfused_mem = mem[4]
        mem_norm = [m / unfused_mem for m in mem]
        ax.set_xlim([min(mem_norm) / 1.2, max(mem_norm) * 1.3])
        ax.set_axisbelow(True)

    # Add legend
    handles = [
        plt.Line2D([0], [0], color=color_pal[2], marker=marker_sel[0], linestyle='', markersize=8, label='Fully Fused'),
        plt.Line2D([0], [0], color=color_pal[4], marker=marker_sel[1], linestyle='', markersize=8, label='Partially Fused (First Layer)'),
        plt.Line2D([0], [0], color=color_pal[5], marker=marker_sel[2], linestyle='', markersize=8, label='Partially Fused (Second Layer)'),
        plt.Line2D([0], [0], color=color_pal[1], marker=marker_sel[3], linestyle='', markersize=8, label='First and Second Layer Fused'),
        plt.Line2D([0], [0], color=color_pal[0], marker=marker_sel[4], linestyle='', markersize=8, label='End-to-End Unfused Kernels')
    ]

    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.51, 1.11),
               ncol=3, fontsize=14, labelspacing=0.0001, borderpad=0.1, handletextpad=0.1)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    return fig


def print_data_summary(data):
    """Print a summary of the data being plotted."""
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)

    operation_names = [
        "Fully Fused",
        "Only First Layer Fused",
        "Only Second Layer Fused",
        "First and Second Layer Fused",
        "End-to-end Unfused"
    ]

    for dataset, metrics in data.items():
        print(f"\n{dataset.upper()}:")
        print(f"  FLOPs: {metrics['flops']}")
        print(f"  Mem:   {metrics['mem']}")

        # Calculate normalized values
        unfused_flops = metrics['flops'][4]
        unfused_mem = metrics['mem'][4]

        print("\n  Normalized (by Unfused):")
        for i, name in enumerate(operation_names):
            norm_flops = metrics['flops'][i] / unfused_flops if unfused_flops else 0
            norm_mem = metrics['mem'][i] / unfused_mem if unfused_mem else 0
            print(f"    {name:<30}: FLOPs={norm_flops:.4f}, Mem={norm_mem:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Plot Figure 14 from benchmark metrics')
    parser.add_argument('--json', '-j', type=str, default=None,
                        help='Path to JSON metrics file from process_figure14_metrics.py')
    parser.add_argument('--output', '-o', type=str, default='results/figure14.pdf',
                        help='Output PDF filename (default: results/figure14.pdf)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['cora_ml', 'dblp', 'collab'],
                        help='Datasets to plot (default: cora_ml dblp collab)')
    parser.add_argument('--use-defaults', action='store_true',
                        help='Use hardcoded default data instead of JSON file')
    parser.add_argument('--print-summary', action='store_true',
                        help='Print data summary before plotting')
    args = parser.parse_args()

    # Load data
    if args.use_defaults or args.json is None:
        if args.json is None and not args.use_defaults:
            print("No JSON file specified, using default hardcoded data.")
            print("Use --json <file> to load from metrics file.")
        data = get_default_data()
    else:
        json_path = Path(args.json)
        if not json_path.exists():
            print(f"ERROR: JSON file not found: {json_path}")
            print("Using default hardcoded data instead.")
            data = get_default_data()
        else:
            print(f"Loading metrics from: {json_path}")
            data = load_metrics_from_json(json_path)

    if args.print_summary:
        print_data_summary(data)

    # Generate plot
    plot_figure14(data, args.output, args.datasets)


if __name__ == '__main__':
    main()
