#!/usr/bin/env python3
"""
Figure 17: Dataflow Order Impact Plot

Plots speedup over worst performance (max_cycles / each_cycles)
for different dataflow orderings.
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Use non-interactive backend for saving figures
matplotlib.use('Agg')

# Labels mapping order indices to loop order names
# These map the dataflow order indices to human-readable loop orderings
ORDER_INDEX_TO_LABEL = {
    0: 'ikjl',
    1: 'iklj',
    2: 'ijkl',
    3: 'ijlk',
    4: 'ilkj',
    5: 'iljk',
    12: 'jikl',
    13: 'jilk',
    14: 'jkil',
    16: 'jlik',
    17: 'jlki',
    18: 'likj',
    19: 'lijk',
    22: 'ljik',
    23: 'ljki'
}


def load_results(json_file: str) -> dict:
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def plot_speedup(results: dict, output_file: str, title: str = None):
    """
    Plot speedup over worst performance.

    Speedup = max_cycles / cycles_for_this_order
    Higher is better (faster execution).
    """
    # Extract successful results
    successful = [r for r in results['results'] if r['success']]

    if not successful:
        print("No successful results to plot!")
        return

    # Sort by order index for consistent plotting
    successful = sorted(successful, key=lambda x: x['order_index'])

    # Get cycles and calculate speedup
    cycles = [r['cycles'] for r in successful]
    max_cycles = max(cycles)

    speedups = [max_cycles / c for c in cycles]
    order_indices = [r['order_index'] for r in successful]

    # Get labels for each order
    labels = [ORDER_INDEX_TO_LABEL.get(idx, f'order_{idx}') for idx in order_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color bars by speedup (gradient from red to green)
    colors = plt.cm.RdYlGn(np.array(speedups) / max(speedups))

    bars = ax.bar(range(len(labels)), speedups, color=colors, edgecolor='black', linewidth=0.5)

    # Customize axes
    ax.set_xlabel('Dataflow Order', fontsize=12)
    ax.set_ylabel('Speedup over Worst', fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.annotate(f'{speedup:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Add horizontal line at 1.0 (worst case baseline)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Worst (baseline)')

    # Add title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        metadata = results.get('metadata', {})
        mlir_name = Path(metadata.get('mlir_file', 'unknown')).stem
        sparsity = metadata.get('sparsity', '?')
        ax.set_title(f'Dataflow Order Impact: {mlir_name} (sparsity={sparsity})', fontsize=14)

    # Add grid
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Add statistics annotation
    min_speedup = min(speedups)
    max_speedup = max(speedups)
    avg_speedup = np.mean(speedups)

    stats_text = f'Min: {min_speedup:.2f}x | Max: {max_speedup:.2f}x | Avg: {avg_speedup:.2f}x'
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")

    # Also print summary
    print(f"\nSpeedup Summary:")
    print(f"  Best order: {labels[speedups.index(max_speedup)]} ({max_speedup:.2f}x)")
    print(f"  Worst order: {labels[speedups.index(min_speedup)]} ({min_speedup:.2f}x)")
    print(f"  Potential improvement: {max_speedup/min_speedup:.2f}x")


def plot_cycles_comparison(results: dict, output_file: str):
    """
    Alternative plot showing raw cycle counts.
    """
    successful = [r for r in results['results'] if r['success']]

    if not successful:
        print("No successful results to plot!")
        return

    successful = sorted(successful, key=lambda x: x['order_index'])

    cycles = [r['cycles'] for r in successful]
    order_indices = [r['order_index'] for r in successful]
    labels = [ORDER_INDEX_TO_LABEL.get(idx, f'order_{idx}') for idx in order_indices]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(labels)), cycles, color='steelblue', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Dataflow Order', fontsize=12)
    ax.set_ylabel('Cycle Count', fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

    metadata = results.get('metadata', {})
    mlir_name = Path(metadata.get('mlir_file', 'unknown')).stem
    ax.set_title(f'Cycle Count by Dataflow Order: {mlir_name}', fontsize=14)

    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Format y-axis with K/M suffixes
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K' if x >= 1e3 else str(int(x))
    ))

    plt.tight_layout()
    output_cycles = output_file.replace('.pdf', '_cycles.pdf').replace('.png', '_cycles.png')
    plt.savefig(output_cycles, dpi=300, bbox_inches='tight')
    print(f"Cycles figure saved to: {output_cycles}")


def main():
    parser = argparse.ArgumentParser(description='Plot Figure 17: Dataflow Order Impact')
    parser.add_argument('--input', type=str, default='figure17_results.json',
                        help='Input JSON file with sweep results')
    parser.add_argument('--output', type=str, default='figure17_speedup.pdf',
                        help='Output figure file (PDF or PNG)')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom title for the plot')
    parser.add_argument('--show-cycles', action='store_true',
                        help='Also generate a raw cycles comparison plot')
    args = parser.parse_args()

    # Load results
    results = load_results(args.input)

    # Plot speedup
    plot_speedup(results, args.output, args.title)

    # Optionally plot raw cycles
    if args.show_cycles:
        plot_cycles_comparison(results, args.output)


if __name__ == '__main__':
    main()
