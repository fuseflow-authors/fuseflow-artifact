#!/usr/bin/env python3
"""
Autoencoder Fusion Benchmark Script

Runs autoencoder benchmarks with two fusion strategies:
1. Fully Fused: All operations in one MLIR file
2. Unfused: 5 separate operations (enc_spmm, enc_bias, enc_relu, dec_spmm, dec_bias)

Supports multiple datasets with their native resolutions:
- ImageNet: 224x224 (input_dim=50176, hidden_dim=256)
- NIH: 1024x1024 (input_dim=1048576, hidden_dim=512)
- LUNA16: 512x512 (input_dim=262144, hidden_dim=512)

Sums up cycles for comparison.
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path

# Add the scripts directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
ARTIFACT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import PROJECT_ROOT, ARTIFACT_ROOT

# Dataset configurations
DATASET_CONFIG = {
    'imagenet': {
        'name': 'ImageNet',
        'resolution': '224x224',
        'input_dim': 50176,
        'hidden_dim': 256,
        'fused_mlir': 'autoencoder_imagenet_batched.mlir',
        'encoder_fused_mlir': 'autoencoder_batched_encoder_fused.mlir',
        'decoder_fused_mlir': 'autoencoder_batched_decoder_fused.mlir',
        'unfused_prefix': 'autoencoder_batched_unfused',
    },
    'nih': {
        'name': 'NIH Chest X-ray',
        'resolution': '1024x1024',
        'input_dim': 1048576,
        'hidden_dim': 512,
        'fused_mlir': 'autoencoder_nih_batched.mlir',
        'encoder_fused_mlir': 'autoencoder_nih_encoder_fused.mlir',
        'decoder_fused_mlir': 'autoencoder_nih_decoder_fused.mlir',
        'unfused_prefix': 'autoencoder_nih_unfused',
    },
    'luna16': {
        'name': 'LUNA16 CT',
        'resolution': '512x512',
        'input_dim': 262144,
        'hidden_dim': 512,
        'fused_mlir': 'autoencoder_luna16_batched.mlir',
        'encoder_fused_mlir': 'autoencoder_luna16_encoder_fused.mlir',
        'decoder_fused_mlir': 'autoencoder_luna16_decoder_fused.mlir',
        'unfused_prefix': 'autoencoder_luna16_unfused',
    },
}


def run_single_mlir(mlir_file, sparsity, build_dir, parfactor=1):
    """Run a single MLIR file and return the cycle count."""
    cmd = [
        'python3', str(SCRIPT_DIR / 'run_end_to_end.py'),
        '--infile', str(mlir_file),
        '--build', str(build_dir),
        '--useGen',
        '-sp', str(sparsity),
        '-par', str(parfactor),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {mlir_file.name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            cwd=ARTIFACT_ROOT,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout for large NIH matrices
        )

        # Parse cycles from output
        output = result.stdout + result.stderr
        cycles = None
        success = False

        for line in output.split('\n'):
            if 'Simulation completed' in line:
                # Extract (True/False, cycles) tuple
                if 'True' in line:
                    success = True
                    # Extract number from tuple
                    import re
                    match = re.search(r'\(True,\s*(\d+)\)', line)
                    if match:
                        cycles = int(match.group(1))
                elif 'False' in line:
                    success = False

        if cycles:
            print(f"Result: Success={success}, Cycles={cycles:,}")
        else:
            print(f"Result: Could not parse cycles")
            print(f"STDOUT: {result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout}")
            print(f"STDERR: {result.stderr[-500:] if len(result.stderr) > 500 else result.stderr}")

        return success, cycles

    except subprocess.TimeoutExpired:
        print(f"Timeout running {mlir_file.name}")
        return False, None
    except Exception as e:
        print(f"Error running {mlir_file.name}: {e}")
        return False, None


def run_benchmark(dataset, sparsity, build_dir, parfactor, only=None):
    """Run fusion benchmark for a specific dataset."""
    config = DATASET_CONFIG[dataset]
    models_dir = ARTIFACT_ROOT / 'fuseflow-compiler' / 'tests' / 'models'

    print("\n" + "="*80)
    print(f"DATASET: {config['name']} ({config['resolution']})")
    print(f"Input Dim: {config['input_dim']:,}, Hidden Dim: {config['hidden_dim']}")
    print("="*80)

    results = {}

    # ===== FULLY FUSED =====
    if not only or only == 'fused':
        print("\n" + "="*80)
        print("BENCHMARK 1: FULLY FUSED (all operations in one file)")
        print("="*80)

        fused_mlir = models_dir / config['fused_mlir']
        if fused_mlir.exists():
            success, cycles = run_single_mlir(fused_mlir, sparsity, build_dir, parfactor)
            results['fully_fused'] = {
                'success': success,
                'cycles': cycles,
                'files': [fused_mlir.name]
            }
        else:
            print(f"ERROR: {fused_mlir} not found")

    # ===== PARTIALLY FUSED =====
    if not only or only == 'partially_fused':
        print("\n" + "="*80)
        print("BENCHMARK 2: PARTIALLY FUSED (encoder + decoder separate)")
        print("="*80)

        partial_files = [
            models_dir / config['encoder_fused_mlir'],
            models_dir / config['decoder_fused_mlir'],
        ]

        partial_cycles = 0
        partial_success = True
        partial_results = []

        for mlir_file in partial_files:
            if mlir_file.exists():
                success, cycles = run_single_mlir(mlir_file, sparsity, build_dir, parfactor)
                partial_results.append({'file': mlir_file.name, 'success': success, 'cycles': cycles})
                if success and cycles:
                    partial_cycles += cycles
                else:
                    partial_success = False
            else:
                print(f"ERROR: {mlir_file} not found")
                partial_success = False

        results['partially_fused'] = {
            'success': partial_success,
            'cycles': partial_cycles if partial_success else None,
            'files': [f.name for f in partial_files],
            'breakdown': partial_results
        }

    # ===== UNFUSED =====
    if not only or only == 'unfused':
        print("\n" + "="*80)
        print("BENCHMARK 3: UNFUSED (5 separate operations)")
        print("="*80)

        unfused_prefix = config['unfused_prefix']
        unfused_files = [
            models_dir / f'{unfused_prefix}_1_enc_spmm.mlir',
            models_dir / f'{unfused_prefix}_2_enc_bias.mlir',
            models_dir / f'{unfused_prefix}_3_enc_relu.mlir',
            models_dir / f'{unfused_prefix}_4_dec_spmm.mlir',
            models_dir / f'{unfused_prefix}_5_dec_bias.mlir',
        ]

        unfused_cycles = 0
        unfused_success = True
        unfused_results = []

        for mlir_file in unfused_files:
            if mlir_file.exists():
                success, cycles = run_single_mlir(mlir_file, sparsity, build_dir, parfactor)
                unfused_results.append({'file': mlir_file.name, 'success': success, 'cycles': cycles})
                if success and cycles:
                    unfused_cycles += cycles
                else:
                    unfused_success = False
            else:
                print(f"ERROR: {mlir_file} not found")
                unfused_success = False

        results['unfused'] = {
            'success': unfused_success,
            'cycles': unfused_cycles if unfused_success else None,
            'files': [f.name for f in unfused_files],
            'breakdown': unfused_results
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Autoencoder Fusion Benchmark')
    parser.add_argument('-sp', '--sparsity', type=float, default=0.9,
                        help='Weight sparsity (default: 0.9)')
    parser.add_argument('-b', '--build', type=str, default=None,
                        help='Path to build directory')
    parser.add_argument('-par', '--parfactor', type=int, default=1,
                        help='Parallel factor')
    parser.add_argument('--only', type=str, choices=['fused', 'partially_fused', 'unfused'],
                        help='Only run specific benchmark type')
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'nih', 'luna16', 'all'],
                        default='imagenet',
                        help='Dataset to benchmark (default: imagenet)')
    args = parser.parse_args()

    build_dir = Path(args.build) if args.build else PROJECT_ROOT / 'build'

    # Determine which datasets to run
    if args.dataset == 'all':
        datasets = ['imagenet', 'nih', 'luna16']
    else:
        datasets = [args.dataset]

    all_results = {}

    for dataset in datasets:
        results = run_benchmark(dataset, args.sparsity, build_dir, args.parfactor, args.only)
        all_results[dataset] = results

        # Print summary for this dataset
        config = DATASET_CONFIG[dataset]
        print("\n" + "="*80)
        print(f"BENCHMARK SUMMARY: {config['name']} ({config['resolution']})")
        print("="*80)
        print(f"Weight Sparsity: {args.sparsity * 100:.0f}%")
        print(f"Parallel Factor: {args.parfactor}")
        print()

        for name, data in results.items():
            if data['success'] and data['cycles']:
                print(f"{name.upper():20} : {data['cycles']:>15,} cycles")
                if 'breakdown' in data:
                    for item in data['breakdown']:
                        if item['cycles']:
                            print(f"  - {item['file']:45} : {item['cycles']:>12,} cycles")
            else:
                print(f"{name.upper():20} : FAILED")

        # Calculate speedups (unfused as baseline)
        if 'unfused' in results and results['unfused']['cycles']:
            unfused_cycles = results['unfused']['cycles']
            print("\n" + "-"*60)
            print("SPEEDUPS (vs Unfused baseline)")
            print("-"*60)

            for name, data in results.items():
                if data['success'] and data['cycles']:
                    speedup = unfused_cycles / data['cycles']
                    print(f"{name.upper():20} : {speedup:.2f}x faster")

    # Print combined summary if multiple datasets
    if len(datasets) > 1:
        print("\n" + "="*100)
        print("COMBINED SUMMARY (All Datasets)")
        print("="*100)
        print(f"{'Dataset':<20} {'Resolution':<12} {'Unfused':>15} {'Partial':>15} {'Fused':>15} {'Speedup':>10}")
        print("-"*100)

        for dataset in datasets:
            config = DATASET_CONFIG[dataset]
            results = all_results[dataset]
            unfused = results.get('unfused', {}).get('cycles', 'N/A')
            partial = results.get('partially_fused', {}).get('cycles', 'N/A')
            fused = results.get('fully_fused', {}).get('cycles', 'N/A')

            if isinstance(unfused, int) and isinstance(fused, int):
                speedup = f"{unfused/fused:.2f}x"
                unfused_str = f"{unfused:,}"
                fused_str = f"{fused:,}"
            else:
                speedup = "N/A"
                unfused_str = str(unfused)
                fused_str = str(fused)

            partial_str = f"{partial:,}" if isinstance(partial, int) else str(partial)

            print(f"{config['name']:<20} {config['resolution']:<12} {unfused_str:>15} {partial_str:>15} {fused_str:>15} {speedup:>10}")

    return all_results


if __name__ == '__main__':
    main()
