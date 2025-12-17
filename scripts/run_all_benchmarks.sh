#!/bin/bash
# FuseFlow Artifact - Run All Benchmarks and Generate All Plots
# This script runs all experiments and generates all figures from the paper

set -e  # Exit on error

echo "========================================"
echo "FuseFlow Artifact - Running All Benchmarks"
echo "========================================"
echo ""
echo "This will run all experiments and generate all figures."
echo "Estimated total time: ~96 compute-hours"
echo ""
echo "Progress will be saved as each figure completes."
echo "You can safely interrupt and resume later."
echo ""

# Create results directory
mkdir -p results

# Figure 12 - Main performance comparison (SAE, GCN, GraphSAGE, GPT-3)
echo "========================================"
echo "Running Figure 12: Performance Comparison"
echo "Estimated time: up to ~96 compute-hours (medium mode, 4 workers)"
echo "========================================"
python3 scripts/run_figure12_benchmarks.py --mode medium --workers 4 --no-hbm
echo "Generating Figure 12 plot..."
python3 scripts/plot_figure12.py --input figure12_results.json --output results/figure12.pdf
echo "✓ Figure 12 complete: results/figure12.pdf"
echo ""

# Figure 14 - GCN FLOPs and memory metrics
echo "========================================"
echo "Running Figure 14: GCN FLOPs and Memory Analysis"
echo "Estimated time: ~5 compute-minutes"
echo "========================================"
python3 scripts/process_figure14_metrics.py
echo "Generating Figure 14 plot..."
python3 scripts/plot_figure14.py
echo "✓ Figure 14 complete: results/figure14.pdf"
echo ""

# Figure 15a - Parallelism Factor Sweep
echo "========================================"
echo "Running Figure 15a: Parallelism Factor Sweep"
echo "Estimated time: ~1 compute-hour"
echo "========================================"
python3 scripts/run_figure15a_sweep.py
echo "Generating Figure 15a plot..."
python3 scripts/plot_figure15a.py
echo "✓ Figure 15a complete: results/figure15a.pdf"
echo ""

# Figure 15b - Parallelism Location Sweep
echo "========================================"
echo "Running Figure 15b: Parallelism Location Sweep"
echo "Estimated time: ~10 compute-minutes"
echo "========================================"
python3 scripts/run_figure15b_sweep.py
echo "Generating Figure 15b plot..."
python3 scripts/plot_figure15b.py
echo "✓ Figure 15b complete: results/figure15b.pdf"
echo ""

# Figure 16 - Block size comparison
echo "========================================"
echo "Running Figure 16: Block Size Comparison"
echo "Estimated time: ~30 compute-minutes"
echo "========================================"
python3 scripts/run_figure16.py
echo "Generating Figure 16 plot..."
python3 scripts/plot_figure16.py
echo "✓ Figure 16 complete: results/figure16.pdf"
echo ""

# Figure 17 - Dataflow order sweep
echo "========================================"
echo "Running Figure 17: Dataflow Order Sweep"
echo "Estimated time: ~5 compute-minutes"
echo "========================================"
python3 scripts/run_figure17_sweep.py
echo "Generating Figure 17 plot..."
python3 scripts/plot_figure17.py
echo "✓ Figure 17 complete: results/figure17.pdf"
echo ""

echo "========================================"
echo "All Benchmarks Complete!"
echo "========================================"
echo ""
echo "Generated figures:"
echo "  - results/figure12.pdf"
echo "  - results/figure14.pdf"
echo "  - results/figure15a.pdf"
echo "  - results/figure15b.pdf"
echo "  - results/figure16.pdf"
echo "  - results/figure17.pdf"
echo ""
echo "Result data files:"
echo "  - figure12_results.json"
echo "  - figure15a_results.json"
echo "  - figure15b_results.json"
echo "  - figure16_results.json"
echo "  - figure17_results.json"
echo ""
echo "To copy results to your host machine, exit the container (CTRL-p, CTRL-q)"
echo "and run the following from your host:"
echo ""
echo "  ./scripts/extract_results.sh"
echo ""
