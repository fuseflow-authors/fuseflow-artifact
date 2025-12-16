#!/usr/bin/env python3
"""
Plot block comparison results from JSON file.
Compares scalar (unstructured) vs true block sparse performance for MHA.
"""

import json
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Disable LaTeX rendering
matplotlib.rcParams['text.usetex'] = False

matplotlib.rc('xtick', labelsize=19)
matplotlib.rc('ytick', labelsize=19)


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_block_comparison(results, output_path):
    """Plot block comparison results."""
    colors = sns.color_palette(palette='tab20')

    configurations = ['16', '32', '64']
    block_sizes = [16, 32, 64]

    # Extract cycles from results - JSON has nested structure:
    # results['16']['scalar']['cycles'] and results['16']['trueblock']['cycles']
    scalar_cycles = []
    blocked_cycles = []

    for bs in block_sizes:
        bs_str = str(bs)
        if bs_str in results:
            bs_data = results[bs_str]
            # Get scalar cycles
            if 'scalar' in bs_data and bs_data['scalar'].get('cycles'):
                scalar_cycles.append(bs_data['scalar']['cycles'])
            else:
                scalar_cycles.append(None)
            # Get trueblock cycles
            if 'trueblock' in bs_data and bs_data['trueblock'].get('cycles'):
                blocked_cycles.append(bs_data['trueblock']['cycles'])
            else:
                blocked_cycles.append(None)
        else:
            scalar_cycles.append(None)
            blocked_cycles.append(None)

    # Check for missing data
    valid_data = []
    for i, bs in enumerate(configurations):
        if scalar_cycles[i] is not None and blocked_cycles[i] is not None:
            valid_data.append(i)
        else:
            print(f"Warning: Missing data for block size {bs}")

    if not valid_data:
        print("Error: No valid data to plot")
        return

    # Filter to valid data only
    configurations = [configurations[i] for i in valid_data]
    scalar_cycles = [scalar_cycles[i] for i in valid_data]
    blocked_cycles = [blocked_cycles[i] for i in valid_data]

    x = np.arange(len(configurations))
    width = 0.2

    # Calculate speedup (scalar / blocked)
    final_naive = [1] * len(configurations)
    final_blocked = [s / b for s, b in zip(scalar_cycles, blocked_cycles)]

    # Plotting
    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width, final_naive, width,
                    label='Unstructured', color=colors[0])
    rects2 = ax.bar(x, final_blocked, width,
                    label='Blocked', color=colors[1])

    ax.set_yscale('log')
    ax.set_ylabel('Speedup', fontsize=21)
    ax.set_xlabel('Block Size', fontsize=21)
    bottom, top = plt.ylim()
    plt.ylim(top=100000)
    ax.set_xticks(x)
    ax.set_xticklabels(configurations)
    ax.legend(loc="upper left", fontsize=18, bbox_to_anchor=(-0.04, 1.07))

    plt.show()
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Print summary
    print("\n=== Block Comparison Results ===")
    for i, bs in enumerate(configurations):
        print(f"Block Size {bs}:")
        print(f"  Scalar cycles:  {scalar_cycles[i]:,}")
        print(f"  Blocked cycles: {blocked_cycles[i]:,}")
        print(f"  Speedup:        {final_blocked[i]:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Plot block comparison results')
    parser.add_argument('--input', '-i', type=str,
                        default='block_comparison_results.json',
                        help='Input JSON file with results')
    parser.add_argument('--output', '-o', type=str,
                        default='results/figure16.pdf',
                        help='Output PDF path')
    args = parser.parse_args()

    data = load_results(args.input)
    # Handle nested 'results' structure from run_block_comparison.py
    results = data.get('results', data)
    plot_block_comparison(results, args.output)


if __name__ == '__main__':
    main()
