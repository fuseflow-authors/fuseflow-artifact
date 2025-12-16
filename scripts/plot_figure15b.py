#!/usr/bin/env python3
"""
Plot Figure 15b - Parallelization Level Sweep

Compares parallelization factors (1, 2, 4) across different stream levels:
- Stream level 1 with par factors 1, 2, 4
- Stream level 2 with par factors 1, 2, 4
- Both stream levels parallelized (par=4 for both)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
import argparse

# Disable LaTeX rendering (scienceplots requires LaTeX)
matplotlib.rcParams['text.usetex'] = False
matplotlib.use('Agg')

matplotlib.rc('xtick', labelsize=19)
matplotlib.rc('ytick', labelsize=19)

colors = sns.color_palette(palette='tab20')
speedup_color = colors[9]


def main():
    parser = argparse.ArgumentParser(description='Plot Figure 15b - Parallelization Level Sweep')
    parser.add_argument('--output', type=str, default='results/figure15b.pdf',
                        help='Output path for the figure')
    args = parser.parse_args()

    # X-axis: Stream Level 1, Stream Level 2
    configurations = ['1', '2', 'All']

    # Cycles for each configuration
    # [stream_level_1_cycles, stream_level_2_cycles] for each par factor
    values_config_1 = [806101226, 806101226]  # par factor 1
    values_config_2 = [403087023, 403054894]  # par factor 2
    values_config_4 = [201560883, 201531826]  # par factor 4
    combined_value = [50390209]  # both levels parallelized with par=4

    # Calculate speedup relative to baseline (par=1, stream level 1)
    baseline = 806101226
    speedup_1 = [baseline / y for y in values_config_1]
    speedup_2 = [baseline / y for y in values_config_2]
    speedup_4 = [baseline / y for y in values_config_4]

    print("Speedups:")
    print(f"  Par=1: {speedup_1}")
    print(f"  Par=2: {speedup_2}")
    print(f"  Par=4: {speedup_4}")
    print(f"  Combined (par=4 both): {baseline/combined_value[0]:.2f}x")

    x = np.arange(len(values_config_1))  # the label locations
    width = 0.2  # width of the bars

    # Plotting
    fig, ax = plt.subplots()

    # Creating bars for par factors 1, 2, and 4
    rects = ax.bar(x - width, speedup_1, width, label='1', color=colors[0])
    rects1 = ax.bar(x, speedup_2, width, label='2', color=colors[1])
    rects2 = ax.bar(x + width, speedup_4, width, label='4', color=colors[2])

    # Adding a separate bar for 'Combined' (both levels parallelized)
    combined_speedup = baseline / combined_value[0]
    rects4 = ax.bar(len(values_config_1), [combined_speedup], width, color=colors[1])

    # Add some labels
    ax.set_ylabel('Normalized Cycles', fontsize=21)
    ax.set_xlabel('Parallelized Level', fontsize=21)
    ax.set_xticks(np.append(x, len(values_config_1)))
    ax.set_xticklabels(configurations)
    ax.legend(title="Par Factor", fontsize=16, title_fontsize=18)

    plt.tight_layout()
    plt.show()

    fig.savefig(args.output, format="pdf", dpi=600, bbox_inches='tight')
    print(f"\nFigure saved to: {args.output}")


if __name__ == '__main__':
    main()
