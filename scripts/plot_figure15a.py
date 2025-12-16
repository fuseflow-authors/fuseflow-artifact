import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, FuncFormatter
import json
import argparse

mpl.rc('ytick', labelsize=18)
mpl.rc('xtick', labelsize=18)

colors = sns.color_palette(palette='tab20')
scaling_color = colors[0]


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description='Plot Figure 15 - Parallelization Factor Sweep')
    parser.add_argument('--json', type=str, default='figure16_results.json',
                        help='Path to JSON file with results')
    parser.add_argument('--output', type=str, default='results/figure15.pdf',
                        help='Output path for the figure')
    args = parser.parse_args()

    # Load results from JSON
    data = load_results(args.json)
    fig16a = data['figure16a']
    results = fig16a['results']

    # Parallelization factors (sorted)
    par_fact = sorted([int(k) for k in results.keys()])
    par_sweep_cycles = [results[str(p)] for p in par_fact]

    # Calculate speedup relative to par=1
    baseline = par_sweep_cycles[0]
    par_sweep_speedup = [baseline / cycles for cycles in par_sweep_cycles]

    # Ideal scaling (linear speedup)
    ideal_speedup = [p for p in par_fact]

    print("Parallelization Factor Sweep Results:")
    print("=" * 50)
    for i, par in enumerate(par_fact):
        print(f"par={par}: {par_sweep_cycles[i]:,} cycles, speedup={par_sweep_speedup[i]:.2f}x (ideal={ideal_speedup[i]}x)")

    print("\nSpeedup efficiency:")
    for i, par in enumerate(par_fact):
        efficiency = par_sweep_speedup[i] / ideal_speedup[i] * 100
        print(f"par={par}: {efficiency:.1f}%")

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(par_fact))

    # Plot ideal scaling line
    ax.plot(x, ideal_speedup, color='gray', linestyle='--', marker='x', label="Ideal Scaling", alpha=0.7)

    # Plot actual speedup
    ax.scatter(x, par_sweep_speedup, color=colors[0], s=60, zorder=5)
    ax.plot(x, par_sweep_speedup, color=colors[0], label="Actual Speedup")

    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_locator(LogLocator(base=2.0, numticks=8))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):d}" if x >= 1 else f"{x:.2f}"))

    ax.set_ylabel("Speedup", fontsize=21)
    ax.set_xlabel("Parallel Factor", fontsize=21)
    ax.set_xticks(x, par_fact)

    ax.legend(fontsize=14, loc='upper left')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    fig.savefig(args.output, format="pdf", dpi=600, bbox_inches='tight')
    print(f"\nFigure saved to: {args.output}")


if __name__ == '__main__':
    main()
