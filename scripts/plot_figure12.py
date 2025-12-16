#!/usr/bin/env python3
"""
Figure 12 Plotting Script for FuseFlow Paper

Reads benchmark results from JSON file and generates fusion ablation plot.
Supports both hardcoded data (for manual entry) and JSON file input.
"""

import argparse
import json
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to use scienceplots style, fall back to default if LaTeX not available
USE_LATEX = False  # Set to True if you have LaTeX installed

try:
    import scienceplots
    if USE_LATEX:
        plt.style.use('science')
    else:
        # Use science style without LaTeX
        plt.style.use(['science', 'no-latex'])
except Exception as e:
    print(f"Note: Could not load 'science' style ({e}). Using default style.")
    plt.style.use('seaborn-v0_8-whitegrid')

# Ensure LaTeX is disabled if not available
matplotlib.rcParams['text.usetex'] = USE_LATEX

matplotlib.rc('xtick', labelsize=17)
matplotlib.rc('ytick', labelsize=17)
colors = sns.color_palette(palette='tab20')
speedup_color = colors[9]


def normalize(time_dict):
    """Normalize times relative to Unfused (so Unfused = 1.0)

    Missing data (None or 0) is treated as 0 in the output.
    """
    normalized = {}
    for key in time_dict:
        normalized[key] = []

    for i, unfused_time in enumerate(time_dict["Unfused"]):
        # Treat None as 0
        unfused_val = unfused_time if unfused_time is not None else 0
        for key in time_dict:
            val = time_dict[key][i]
            # Treat None as 0
            if val is None:
                val = 0
            if val > 0 and unfused_val > 0:
                normalized[key].append(unfused_val / val)
            else:
                normalized[key].append(0)  # Handle missing data
    return normalized


def load_results_from_json(json_path):
    """Load benchmark results from JSON file and extract latencies.

    Missing benchmark points are treated as 0 (will show as 0 in the plot).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', data)  # Handle both formats

    extracted = {
        'sae': {'dims': [], 'unfused': [], 'partial': [], 'fused': []},
        'gcn': {'dims': [], 'unfused': [], 'partial': [], 'fused': []},
        'graphsage': {'dims': [], 'unfused': [], 'partial': [], 'fused': []},
        'gpt3': {'dims': [], 'unfused': [], 'partial': [], 'fused': []},
    }

    def safe_get_cycles(result_dict, key):
        """Safely get cycles value, returning 0 if missing or failed.

        If any MLIR file in the configuration failed (success=False),
        or if cycles is None/0, returns 0.
        """
        if result_dict is None:
            return 0
        sub_dict = result_dict.get(key, {})
        if sub_dict is None:
            return 0
        # Check if the configuration succeeded
        success = sub_dict.get('success', True)  # Default to True for backwards compat
        if not success:
            return 0
        cycles = sub_dict.get('cycles')
        return cycles if cycles is not None else 0

    # Dataset display name mappings
    sae_names = {'imagenet': 'ImageNet', 'nih': 'NIH-CXR', 'luna16': 'LUNA16'}
    gcn_names = {'cora': 'cora', 'cora_ml': 'cora_ml', 'dblp': 'dblp', 'collab': 'collab', 'mag': 'mag'}
    graphsage_names = gcn_names.copy()
    gpt3_names = {'16': '16', '32': '32', '64': '64'}

    # Extract SAE results
    if 'sae' in results:
        for dataset in ['imagenet', 'nih', 'luna16']:
            if dataset in results['sae']:
                r = results['sae'][dataset]
                extracted['sae']['dims'].append(sae_names.get(dataset, dataset))
                extracted['sae']['unfused'].append(safe_get_cycles(r, 'unfused'))
                extracted['sae']['partial'].append(safe_get_cycles(r, 'partially_fused'))
                extracted['sae']['fused'].append(safe_get_cycles(r, 'fully_fused'))

    # Extract GCN results
    if 'gcn' in results:
        for dataset in ['cora', 'cora_ml', 'dblp', 'collab', 'mag']:
            if dataset in results['gcn']:
                r = results['gcn'][dataset]
                extracted['gcn']['dims'].append(gcn_names.get(dataset, dataset))
                extracted['gcn']['unfused'].append(safe_get_cycles(r, 'unfused'))
                extracted['gcn']['partial'].append(safe_get_cycles(r, 'partially_fused'))
                extracted['gcn']['fused'].append(safe_get_cycles(r, 'fully_fused'))

    # Extract GraphSAGE results
    if 'graphsage' in results:
        for dataset in ['cora', 'cora_ml', 'dblp', 'collab', 'mag']:
            if dataset in results['graphsage']:
                r = results['graphsage'][dataset]
                extracted['graphsage']['dims'].append(graphsage_names.get(dataset, dataset))
                extracted['graphsage']['unfused'].append(safe_get_cycles(r, 'unfused'))
                extracted['graphsage']['partial'].append(safe_get_cycles(r, 'partially_fused'))
                extracted['graphsage']['fused'].append(safe_get_cycles(r, 'fully_fused'))

    # Extract GPT-3 results (use full 12-decoder model latencies)
    if 'gpt3' in results:
        for block_size in ['16', '32', '64']:
            if block_size in results['gpt3']:
                r = results['gpt3'][block_size]
                extracted['gpt3']['dims'].append(gpt3_names.get(block_size, block_size))

                # Use full_model_12_decoders if available, otherwise use single-decoder * 12
                full_model = r.get('full_model_12_decoders', {})
                if full_model:
                    # Get values with 0 fallback for missing data
                    unfused = full_model.get('unfused')
                    partial = full_model.get('partially_fused')
                    fused = full_model.get('fully_fused')
                    extracted['gpt3']['unfused'].append(unfused if unfused is not None else 0)
                    extracted['gpt3']['partial'].append(partial if partial is not None else 0)
                    extracted['gpt3']['fused'].append(fused if fused is not None else 0)
                else:
                    # Fallback: multiply single decoder by 12
                    unfused = safe_get_cycles(r, 'unfused')
                    partial = safe_get_cycles(r, 'partially_fused')
                    fused = safe_get_cycles(r, 'fully_fused')
                    extracted['gpt3']['unfused'].append(unfused * 12 if unfused else 0)
                    extracted['gpt3']['partial'].append(partial * 12 if partial else 0)
                    extracted['gpt3']['fused'].append(fused * 12 if fused else 0)

    return extracted


def get_default_data():
    """Return default/hardcoded data for plotting."""
    return {
        'sae': {
            'dims': ["ImageNet", "NIH-CXR", "LUNA16"],
            'unfused': [81273+264+26785+791, 106161+264+26785+1031, 106161+264+26785+1031],
            'partial': [81278+27317, 106166+27557, 106166+27557],
            'fused': [162480, 212147, 212147],
        },
        'gcn': {
            'dims': ["cora", "cora_ml", "dblp", "collab", "mag"],
            'unfused': [48998261, 115435265, 934042655, 1056927396, 16012893504],
            'partial': [29997611, 61259360, 627775406, 409951944, 7505550758],
            'fused': [166361105, 842383059, 3589346391, 55634377783, 185792502899],
        },
        'graphsage': {
            'dims': ["cora", "cora_ml", "dblp", "collab", "mag"],
            'unfused': [79355629, 183474564, 1269087780, 1569940814, 29409284934],
            'partial': [37798698, 99243656, 627775394, 505455837, 7505550758],
            'fused': [166361105, 842388679, 3589346391, 55634376841, 849409284934],
        },
        'gpt3': {
            'dims': ["16", "32", "64"],
            'unfused': [98511714108, 99652522812, 102275175360],
            'partial': [56340821076, 60497286996, 60497286996],
            'fused': [36439754748, 40596220668, 40596220668],
        },
    }


def convert_to_plot_format(data):
    """Convert extracted data to plotting format."""
    return {
        'Unfused': data['unfused'],
        'Partially Fused': data['partial'],
        'Fully Fused': data['fused'],
    }


def plot_figure12(data, output_path='fusion_ablation.pdf'):
    """Generate the Figure 12 plot."""

    legend = ["Unfused", "Partially Fused", "Fully Fused"]

    # Prepare data for each model
    sae_dims = tuple(data['sae']['dims'])
    sae_times = normalize(convert_to_plot_format(data['sae']))
    x_sae = np.arange(len(sae_dims))

    gcn_dims = tuple(data['gcn']['dims'])
    gcn_times = normalize(convert_to_plot_format(data['gcn']))
    x_gcn = np.arange(len(gcn_dims))

    graphsage_dims = tuple(data['graphsage']['dims'])
    graphsage_times = normalize(convert_to_plot_format(data['graphsage']))
    x_graphsage = np.arange(len(graphsage_dims))

    gpt3_dims = tuple(data['gpt3']['dims'])
    gpt3_times = normalize(convert_to_plot_format(data['gpt3']))
    x_gpt3 = np.arange(len(gpt3_dims))

    width = 0.2
    multiplier = 0

    fig = plt.figure(figsize=(19, 2.2), dpi=300)
    gs = fig.add_gridspec(1, 4, hspace=0, wspace=0)
    (ax1, ax2, ax3, ax4) = gs.subplots(sharex=False, sharey=True)

    ax1.set_ylabel('Speedup', fontsize=20)
    ax1.set_xlabel("Dataset", fontsize=20)
    ax2.set_xlabel("Dataset", fontsize=20)
    ax3.set_xlabel("Dataset", fontsize=20)
    ax4.set_xlabel("Block Size", fontsize=20)

    # Add model name labels on top
    sec_ax1 = ax1.secondary_xaxis('top')
    sec_ax1.set_xlabel('SAE', fontsize=18)
    sec_ax1.set_xticklabels([])

    sec_ax2 = ax2.secondary_xaxis('top')
    sec_ax2.set_xlabel('GCN', fontsize=18)
    sec_ax2.set_xticklabels([])

    sec_ax3 = ax3.secondary_xaxis('top')
    sec_ax3.set_xlabel('GraphSAGE', fontsize=18)
    sec_ax3.set_xticklabels([])

    sec_ax4 = ax4.secondary_xaxis('top')
    sec_ax4.set_xlabel('GPT-3 w/ BigBird', fontsize=18)
    sec_ax4.set_xticklabels([])

    # Plot SAE
    for i, (label, ablation_data) in enumerate(sae_times.items()):
        offset = width * multiplier
        rects = ax1.bar(x_sae + offset, ablation_data,
                        width, label=label, color=colors[i])
        multiplier += 1
    ax1.set_xticks(x_sae + 0.2, sae_dims)

    # Plot GCN
    for i, (label, ablation_data) in enumerate(gcn_times.items()):
        offset = width * multiplier
        rects = ax2.bar(x_gcn + offset, ablation_data,
                        width, label=label, color=colors[i])
        multiplier += 1
    ax2.set_xticks(x_gcn + 2 * (0.223) + .3, gcn_dims)

    # Plot GraphSAGE
    for i, (label, ablation_data) in enumerate(graphsage_times.items()):
        offset = width * multiplier
        rects = ax3.bar(x_graphsage + offset, ablation_data,
                        width, label=label, color=colors[i])
        multiplier += 1
    ax3.set_xticks(x_graphsage + 3 * width + 0.8, graphsage_dims)

    # Plot GPT-3
    for i, (label, ablation_data) in enumerate(gpt3_times.items()):
        offset = width * multiplier
        rects = ax4.bar(x_gpt3 + offset, ablation_data,
                        width, label=label, color=colors[i])
        multiplier += 1
    ax4.set_xticks(x_gpt3 + 5 * width + 1.0, gpt3_dims)

    ax1.legend(legend, ncol=1, loc='upper left', fontsize='14', bbox_to_anchor=(0, 1.05))

    plt.show()
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    return fig


def print_data_summary(data):
    """Print a summary of the data being plotted."""
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)

    for model in ['sae', 'gcn', 'graphsage', 'gpt3']:
        print(f"\n{model.upper()}:")
        print(f"  Datasets: {data[model]['dims']}")
        print(f"  Unfused:  {data[model]['unfused']}")
        print(f"  Partial:  {data[model]['partial']}")
        print(f"  Fused:    {data[model]['fused']}")

        # Calculate speedups
        print("  Speedups (Unfused/Fused):")
        for i, dim in enumerate(data[model]['dims']):
            unfused = data[model]['unfused'][i]
            fused = data[model]['fused'][i]
            if unfused and fused and fused > 0:
                speedup = unfused / fused
                print(f"    {dim}: {speedup:.2f}x")
            else:
                print(f"    {dim}: N/A")


def main():
    parser = argparse.ArgumentParser(description='Plot Figure 12 from benchmark results')
    parser.add_argument('--json', '-j', type=str, default=None,
                        help='Path to JSON results file from run_figure12_benchmarks.py')
    parser.add_argument('--output', '-o', type=str, default='results/figure12.pdf',
                        help='Output PDF filename (default: results/figure12.pdf)')
    parser.add_argument('--use-defaults', action='store_true',
                        help='Use hardcoded default data instead of JSON file')
    parser.add_argument('--print-summary', action='store_true',
                        help='Print data summary before plotting')
    args = parser.parse_args()

    # Load data
    if args.use_defaults or args.json is None:
        if args.json is None and not args.use_defaults:
            print("No JSON file specified, using default hardcoded data.")
            print("Use --json <file> to load from benchmark results.")
        data = get_default_data()
    else:
        json_path = Path(args.json)
        if not json_path.exists():
            print(f"ERROR: JSON file not found: {json_path}")
            print("Using default hardcoded data instead.")
            data = get_default_data()
        else:
            print(f"Loading results from: {json_path}")
            data = load_results_from_json(json_path)

            # Fill in missing data with defaults if needed
            default_data = get_default_data()
            for model in ['sae', 'gcn', 'graphsage', 'gpt3']:
                if not data[model]['dims']:
                    print(f"  WARNING: No {model.upper()} data found, using defaults")
                    data[model] = default_data[model]

    if args.print_summary:
        print_data_summary(data)

    # Generate plot
    plot_figure12(data, args.output)


if __name__ == '__main__':
    main()
