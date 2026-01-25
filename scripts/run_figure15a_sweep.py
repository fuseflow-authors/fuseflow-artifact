#!/usr/bin/env python3
"""
Figure 15a - MHA Parallelization Factor Sweep

Sweeps parfactor from 1 to 64 (powers of 2) on stream level 2
to measure the impact of parallelization on MHA performance.
"""

import subprocess
import os
import sys
import argparse
import json
import re
from pathlib import Path
from datetime import datetime

# Add the scripts directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
ARTIFACT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import PROJECT_ROOT, ARTIFACT_ROOT

DEFAULT_TIMEOUT = 3600  # 1 hour per benchmark


def run_mha_benchmark(mlir_file, build_dir, sparsity, parfactor, stream_level,
                      block_size=64, timeout=DEFAULT_TIMEOUT, parfactor2=None, stream_level2=None):
    """Run a single MHA benchmark with specified parallelization parameters.

    Args:
        mlir_file: Path to the MLIR file
        build_dir: Path to build directory
        sparsity: Weight sparsity (0.0 to 1.0)
        parfactor: Parallel factor for primary stream level
        stream_level: Primary stream level (1 or 2)
        block_size: Block size for BigBird attention
        timeout: Timeout in seconds
        parfactor2: Optional parallel factor for secondary stream level
        stream_level2: Optional secondary stream level

    Returns:
        tuple: (success, cycles, full_output)
    """
    env = os.environ.copy()
    if 'DATA_PATH' not in env:
        env['DATA_PATH'] = '/tmp/data'

    cmd = [
        sys.executable, str(SCRIPT_DIR / 'run_end_to_end.py'),
        '--infile', str(mlir_file),
        '--build', str(build_dir),
        '-sp', str(sparsity),
        '-par', str(parfactor),
        '-sl', str(stream_level),
        '--block', str(block_size),
        '--useGen',
    ]

    # Add secondary parallelization if specified
    if parfactor2 is not None and stream_level2 is not None:
        cmd.extend(['-par2', str(parfactor2), '-sl2', str(stream_level2)])

    print(f"\n{'='*60}")
    print(f"Running MHA: stream_level={stream_level}, parfactor={parfactor}", end='')
    if parfactor2 is not None:
        print(f", stream_level2={stream_level2}, parfactor2={parfactor2}", end='')
    print()
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

        output = result.stdout + result.stderr
        cycles = None
        success = False

        # Try to parse cycles from output
        for line in output.split('\n'):
            if 'Simulation completed' in line:
                if 'True' in line:
                    success = True
                    match = re.search(r'\(True,\s*(\d+)\)', line)
                    if match:
                        cycles = int(match.group(1))
            # Also check for "Elapsed Cycles" format
            if 'Elapsed Cycles' in line:
                match = re.search(r'Elapsed Cycles[:\s]+(\d+)', line)
                if match:
                    cycles = int(match.group(1))
                    success = True

        if cycles:
            print(f"Result: Success={success}, Cycles={cycles:,}")
        else:
            print(f"Result: Could not parse cycles")
            if len(output) > 1000:
                print(f"Output (last 1000 chars): ...{output[-1000:]}")
            else:
                print(f"Output: {output}")

        return success, cycles, output

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return False, None, f"TIMEOUT after {timeout}s"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, None, f"ERROR: {e}"


def run_figure15a_sweep(build_dir, block_size=64, sparsity=0.0, timeout=DEFAULT_TIMEOUT):
    """Run Figure 15a: Parallelization sweep on stream level 2.

    Sweeps parfactor from 1 to 64 (powers of 2) on stream level 2.

    Returns:
        dict: Results with parfactor as keys and cycles as values
    """
    models_dir = ARTIFACT_ROOT / 'fuseflow-compiler' / 'tests' / 'models'
    mlir_file = models_dir / 'gpt-3' / 'multihead_attention.mlir'

    if not mlir_file.exists():
        print(f"ERROR: MLIR file not found: {mlir_file}")
        return {}

    print("\n" + "="*80)
    print("FIGURE 15a: MHA Parallelization Factor Sweep (Stream Level 2)")
    print("="*80)
    print(f"MLIR file: {mlir_file}")
    print(f"Block size: {block_size}")
    print(f"Sparsity: {sparsity}")
    print(f"Parfactors: 1, 2, 4, 8, 16, 32, 64")
    print("="*80)

    results = {}
    parfactors = [1, 2, 4, 8, 16, 32, 64]

    for par in parfactors:
        success, cycles, _ = run_mha_benchmark(
            mlir_file, build_dir, sparsity, par, stream_level=2,
            block_size=block_size, timeout=timeout
        )
        results[str(par)] = {
            'success': success,
            'cycles': cycles,
            'parfactor': par,
            'stream_level': 2
        }

    return results


def print_summary(results):
    """Print a summary of the sweep results."""
    print("\n" + "="*80)
    print("FIGURE 15a SUMMARY - Stream Level 2 Parallelization Sweep")
    print("="*80)
    print(f"{'Parfactor':<15} {'Cycles':>20} {'Speedup vs Par=1':>20}")
    print("-"*60)

    baseline_cycles = None
    if '1' in results and results['1'].get('cycles'):
        baseline_cycles = results['1']['cycles']

    for par in ['1', '2', '4', '8', '16', '32', '64']:
        if par in results:
            r = results[par]
            cycles = r.get('cycles')
            if cycles:
                cycles_str = f"{cycles:,}"
                if baseline_cycles:
                    speedup = baseline_cycles / cycles
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"
            else:
                cycles_str = "FAILED"
                speedup_str = "N/A"
            print(f"{par:<15} {cycles_str:>20} {speedup_str:>20}")


def main():
    parser = argparse.ArgumentParser(description='Figure 15a: MHA Parallelization Factor Sweep')
    parser.add_argument('-b', '--build', type=str, default=None,
                        help='Path to build directory')
    parser.add_argument('--block', type=int, default=64,
                        help='Block size for BigBird attention (default: 64)')
    parser.add_argument('-sp', '--sparsity', type=float, default=0.0,
                        help='Weight sparsity (default: 0.0 for dense)')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help=f'Timeout per benchmark in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')

    args = parser.parse_args()

    build_dir = Path(args.build) if args.build else PROJECT_ROOT / 'build'

    print("="*80)
    print("FIGURE 15a - MHA PARALLELIZATION FACTOR SWEEP")
    print("="*80)
    print(f"Build Dir: {build_dir}")
    print(f"Block Size: {args.block}")
    print(f"Sparsity: {args.sparsity}")
    print(f"Timeout: {args.timeout}s")
    print("="*80)

    # Run the sweep
    results = run_figure15a_sweep(
        build_dir,
        block_size=args.block,
        sparsity=args.sparsity,
        timeout=args.timeout
    )

    # Print summary
    print_summary(results)

    # Save results to JSON
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = ARTIFACT_ROOT / f'figure15a_results_{timestamp}.json'

    results_with_metadata = {
        'metadata': {
            'figure': '15a',
            'description': 'MHA Parallelization Factor Sweep (Stream Level 2)',
            'block_size': args.block,
            'sparsity': args.sparsity,
            'timeout': args.timeout,
            'timestamp': datetime.now().isoformat(),
        },
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
