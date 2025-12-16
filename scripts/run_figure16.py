#!/usr/bin/env python3
"""
Block vs Non-Block Comparison Benchmark Script

Runs MHA (Multihead Attention) benchmarks comparing:
- Non-blocked (scalar mode): blockmode=1, no trueblock
- True block sparse: blockmode=16/32/64 with --trueblock

For block sizes: 16, 32, 64

Results show speedup from true block sparse vs scalar mode.
"""

import subprocess
import os
import sys
import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime

# Add the scripts directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
ARTIFACT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import PROJECT_ROOT, ARTIFACT_ROOT

# Default timeout (1 hour per run - scalar mode can be very slow)
DEFAULT_TIMEOUT = 3600

# MHA MLIR file
MHA_MLIR = 'gpt-3/multihead_attention.mlir'


def run_mha_benchmark(block_size, use_trueblock, sparsity, build_dir, parfactor=1,
                      timeout=DEFAULT_TIMEOUT, enable_hbm=True):
    """Run a single MHA benchmark.

    Args:
        block_size: Block size (16, 32, 64)
        use_trueblock: If True, use true block sparse mode; if False, use scalar mode
        sparsity: Weight sparsity for mask (0.0 to 1.0)
        build_dir: Path to build directory
        parfactor: Parallel factor
        timeout: Timeout in seconds
        enable_hbm: Whether to enable HBM memory simulation

    Returns:
        dict: {'success': bool, 'cycles': int or None, 'output': str}
    """
    models_dir = ARTIFACT_ROOT / 'samml' / 'tests' / 'models'
    mlir_file = models_dir / MHA_MLIR

    if not mlir_file.exists():
        return {'success': False, 'cycles': None, 'error': f'File not found: {mlir_file}'}

    # Set DATA_PATH environment variable
    env = os.environ.copy()
    if 'DATA_PATH' not in env:
        env['DATA_PATH'] = '/tmp/data'

    # Set COMAL_ENABLE_HBM environment variable
    env['COMAL_ENABLE_HBM'] = '1' if enable_hbm else '0'

    # Build command
    cmd = [
        sys.executable, str(SCRIPT_DIR / 'run_end_to_end.py'),
        '--infile', str(mlir_file),
        '--build', str(build_dir),
        '-sp', str(sparsity),
        '-par', str(parfactor),
        '--block', str(block_size),
        '--useGen',
    ]

    if use_trueblock:
        # True block sparse mode
        cmd.extend(['--blockmode', str(block_size), '--trueblock'])
        mode_name = f"True Block {block_size}"
    else:
        # Scalar mode (non-blocked)
        cmd.extend(['--blockmode', '1'])
        mode_name = f"Scalar (block={block_size})"

    print(f"\n{'='*60}")
    print(f"Running: {mode_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            cwd=ARTIFACT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )

        # Parse cycles from output
        output = result.stdout + result.stderr
        cycles = None
        success = False

        for line in output.split('\n'):
            if 'Simulation completed' in line:
                if 'True' in line:
                    success = True
                    match = re.search(r'\(True,\s*(\d+)\)', line)
                    if match:
                        cycles = int(match.group(1))
                elif 'False' in line:
                    success = False

        if cycles:
            print(f"Result: Success={success}, Cycles={cycles:,}")
        else:
            print(f"Result: Could not parse cycles")
            # Print last part of output for debugging
            if len(output) > 2000:
                print(f"Output (last 2000 chars): ...{output[-2000:]}")
            else:
                print(f"Output: {output}")

        return {'success': success, 'cycles': cycles, 'output': output}

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return {'success': False, 'cycles': None, 'error': f'TIMEOUT after {timeout}s'}
    except Exception as e:
        print(f"ERROR: {e}")
        return {'success': False, 'cycles': None, 'error': str(e)}


def run_comparison(sparsity, build_dir, parfactor, timeout, block_sizes=None, enable_hbm=True):
    """Run the full block vs non-block comparison.

    Args:
        sparsity: Weight sparsity
        build_dir: Path to build directory
        parfactor: Parallel factor
        timeout: Timeout per benchmark
        block_sizes: List of block sizes to test (default: [16, 32, 64])
        enable_hbm: Whether to enable HBM simulation

    Returns:
        dict: Results for all block sizes
    """
    if block_sizes is None:
        block_sizes = [16, 32, 64]

    results = {}

    for bs in block_sizes:
        print(f"\n{'#'*80}")
        print(f"# BLOCK SIZE: {bs}")
        print(f"{'#'*80}")

        bs_results = {}

        # Run non-blocked (scalar) mode
        print(f"\n--- Non-Blocked (Scalar) Mode ---")
        scalar_result = run_mha_benchmark(
            block_size=bs,
            use_trueblock=False,
            sparsity=sparsity,
            build_dir=build_dir,
            parfactor=parfactor,
            timeout=timeout,
            enable_hbm=enable_hbm
        )
        bs_results['scalar'] = {
            'success': scalar_result['success'],
            'cycles': scalar_result['cycles'],
            'mode': 'scalar',
            'blockmode': 1,
        }

        # Run true block sparse mode
        print(f"\n--- True Block Sparse Mode (block={bs}) ---")
        block_result = run_mha_benchmark(
            block_size=bs,
            use_trueblock=True,
            sparsity=sparsity,
            build_dir=build_dir,
            parfactor=parfactor,
            timeout=timeout,
            enable_hbm=enable_hbm
        )
        bs_results['trueblock'] = {
            'success': block_result['success'],
            'cycles': block_result['cycles'],
            'mode': 'trueblock',
            'blockmode': bs,
        }

        # Calculate speedup
        scalar_cycles = scalar_result['cycles']
        block_cycles = block_result['cycles']

        if scalar_cycles and block_cycles:
            speedup = scalar_cycles / block_cycles
            bs_results['speedup'] = speedup
            print(f"\n>>> Speedup (Scalar -> True Block {bs}): {speedup:.2f}x")
        else:
            bs_results['speedup'] = None
            print(f"\n>>> Could not calculate speedup (missing data)")

        results[str(bs)] = bs_results

    return results


def print_summary(results):
    """Print a summary table of all results."""
    print("\n" + "="*100)
    print("BLOCK VS NON-BLOCK COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Block Size':<15} {'Scalar (cycles)':<20} {'True Block (cycles)':<20} {'Speedup':<15}")
    print("-"*70)

    for bs, res in sorted(results.items(), key=lambda x: int(x[0])):
        scalar_cycles = res.get('scalar', {}).get('cycles')
        block_cycles = res.get('trueblock', {}).get('cycles')
        speedup = res.get('speedup')

        scalar_str = f"{scalar_cycles:,}" if scalar_cycles else "FAIL"
        block_str = f"{block_cycles:,}" if block_cycles else "FAIL"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"

        print(f"{bs:<15} {scalar_str:<20} {block_str:<20} {speedup_str:<15}")

    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Block vs Non-Block Comparison Benchmark for MHA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all block sizes with default settings
  python run_block_comparison.py

  # Run only block size 16 and 64
  python run_block_comparison.py --block-sizes 16 64

  # Run with custom sparsity and parfactor
  python run_block_comparison.py -sp 0.8 -par 2

  # Run without HBM simulation (faster but less accurate)
  python run_block_comparison.py --no-hbm
"""
    )
    parser.add_argument('-sp', '--sparsity', type=float, default=0.9,
                        help='Weight sparsity for BigBird mask (default: 0.9)')
    parser.add_argument('-b', '--build', type=str, default=None,
                        help='Path to build directory')
    parser.add_argument('-par', '--parfactor', type=int, default=1,
                        help='Parallel factor (default: 1)')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help=f'Timeout per benchmark in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--block-sizes', type=int, nargs='+', default=[16, 32, 64],
                        choices=[16, 32, 64],
                        help='Block sizes to test (default: 16 32 64)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')

    # HBM simulation toggle
    parser.add_argument('--hbm', action='store_true', default=True,
                        help='Enable HBM memory simulation (default: enabled)')
    parser.add_argument('--no-hbm', action='store_false', dest='hbm',
                        help='Disable HBM memory simulation (faster but less accurate)')

    args = parser.parse_args()

    build_dir = Path(args.build) if args.build else PROJECT_ROOT / 'build'

    print("="*100)
    print("BLOCK VS NON-BLOCK COMPARISON - MHA Benchmark")
    print("="*100)
    print(f"Block sizes: {args.block_sizes}")
    print(f"Sparsity: {args.sparsity * 100:.0f}%")
    print(f"Parallel factor: {args.parfactor}")
    print(f"Timeout: {args.timeout}s")
    print(f"Build dir: {build_dir}")
    print(f"HBM simulation: {'ENABLED' if args.hbm else 'DISABLED'}")
    print("="*100)

    start_time = time.time()

    # Run comparison
    results = run_comparison(
        sparsity=args.sparsity,
        build_dir=build_dir,
        parfactor=args.parfactor,
        timeout=args.timeout,
        block_sizes=args.block_sizes,
        enable_hbm=args.hbm
    )

    elapsed_time = time.time() - start_time

    # Print summary
    print_summary(results)

    print(f"\nTotal elapsed time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")

    # Save results to JSON
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = ARTIFACT_ROOT / f'block_comparison_results_{timestamp}.json'

    results_with_metadata = {
        'metadata': {
            'sparsity': args.sparsity,
            'parfactor': args.parfactor,
            'timeout': args.timeout,
            'block_sizes': args.block_sizes,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'hbm_enabled': args.hbm,
        },
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
