#!/usr/bin/env python3
"""
Run Figure 15b Sweep - Parallelization Level Comparison

Sweeps parallelization factors (1, 2, 4) across different stream levels:
- Stream level 1 with par factors 1, 2, 4
- Stream level 2 with par factors 1, 2, 4
- Both stream levels parallelized (par=4 for both)

Uses MHA with block size 16 and useGen for correct BigBird mask generation.
"""

import subprocess
import os
import sys
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

# Configuration
MHA_MLIR = 'gpt-3/multihead_attention.mlir'
BLOCK_SIZE = 64
SPARSITY = 0.0
DEFAULT_TIMEOUT = 3600  # 1 hour per run


def run_benchmark(parfactor, streamlevel, build_dir, timeout=DEFAULT_TIMEOUT,
                  parfactor2=None, streamlevel2=None):
    """Run a single MHA benchmark with specified parallelization settings.

    Args:
        parfactor: Parallel factor for first level
        streamlevel: Stream level for first parallelization
        build_dir: Path to build directory
        timeout: Timeout in seconds
        parfactor2: Optional second parallel factor (for multi-level)
        streamlevel2: Optional second stream level (for multi-level)

    Returns:
        dict: {'success': bool, 'cycles': int or None, 'output': str}
    """
    models_dir = ARTIFACT_ROOT / 'fuseflow-compiler' / 'tests' / 'models'
    mlir_file = models_dir / MHA_MLIR

    if not mlir_file.exists():
        return {'success': False, 'cycles': None, 'error': f'File not found: {mlir_file}'}

    # Set DATA_PATH environment variable
    env = os.environ.copy()
    if 'DATA_PATH' not in env:
        env['DATA_PATH'] = '/tmp/data'

    # Disable HBM for faster runs
    env['COMAL_ENABLE_HBM'] = '0'

    # Build command
    cmd = [
        sys.executable, str(SCRIPT_DIR / 'run_end_to_end.py'),
        '--infile', str(mlir_file),
        '--build', str(build_dir),
        '-sp', str(SPARSITY),
        '-par', str(parfactor),
        '-sl', str(streamlevel),
        '--block', str(BLOCK_SIZE),
        '--useGen',
        '--blockmode', '1',  # Scalar mode
    ]

    # Add second level parallelization if specified
    if parfactor2 is not None and streamlevel2 is not None:
        cmd.extend(['-par2', str(parfactor2), '-sl2', str(streamlevel2)])

    # Create description for logging
    if parfactor2 is not None:
        mode_name = f"par={parfactor}/sl={streamlevel} + par2={parfactor2}/sl2={streamlevel2}"
    else:
        mode_name = f"par={parfactor}, streamlevel={streamlevel}"

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
            if len(output) > 1000:
                print(f"Output (last 1000 chars): ...{output[-1000:]}")
            else:
                print(f"Output: {output}")

        return {'success': success, 'cycles': cycles, 'output': output}

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return {'success': False, 'cycles': None, 'error': f'TIMEOUT after {timeout}s'}
    except Exception as e:
        print(f"ERROR: {e}")
        return {'success': False, 'cycles': None, 'error': str(e)}


def run_sweep(build_dir, timeout):
    """Run the full figure 15b sweep."""
    results = {
        'stream_level_1': {},  # par factors 1, 2, 4 for stream level 1
        'stream_level_2': {},  # par factors 1, 2, 4 for stream level 2
        'combined': None,      # both levels with par=4
    }

    par_factors = [1, 2, 4]

    # Sweep stream level 1
    print("\n" + "#"*80)
    print("# STREAM LEVEL 1 SWEEP")
    print("#"*80)

    for par in par_factors:
        result = run_benchmark(
            parfactor=par,
            streamlevel=1,
            build_dir=build_dir,
            timeout=timeout
        )
        results['stream_level_1'][str(par)] = {
            'success': result['success'],
            'cycles': result['cycles'],
            'parfactor': par,
            'streamlevel': 1,
        }

    # Sweep stream level 2
    print("\n" + "#"*80)
    print("# STREAM LEVEL 2 SWEEP")
    print("#"*80)

    for par in par_factors:
        result = run_benchmark(
            parfactor=par,
            streamlevel=2,
            build_dir=build_dir,
            timeout=timeout
        )
        results['stream_level_2'][str(par)] = {
            'success': result['success'],
            'cycles': result['cycles'],
            'parfactor': par,
            'streamlevel': 2,
        }

    # Combined: both stream levels with par=4
    print("\n" + "#"*80)
    print("# COMBINED (par=4 for both stream levels)")
    print("#"*80)

    result = run_benchmark(
        parfactor=4,
        streamlevel=1,
        build_dir=build_dir,
        timeout=timeout,
        parfactor2=4,
        streamlevel2=2
    )
    results['combined'] = {
        'success': result['success'],
        'cycles': result['cycles'],
        'parfactor': 4,
        'streamlevel': 1,
        'parfactor2': 4,
        'streamlevel2': 2,
    }

    return results


def print_summary(results):
    """Print a summary table of all results."""
    print("\n" + "="*100)
    print("FIGURE 15b SWEEP SUMMARY - Parallelization Level Comparison")
    print("="*100)

    # Get baseline (par=1, stream level 1)
    baseline = results['stream_level_1'].get('1', {}).get('cycles')

    print(f"\n{'Stream Level 1 Sweep:'}")
    print(f"{'Par Factor':<15} {'Cycles':<20} {'Speedup':<15}")
    print("-"*50)
    for par in ['1', '2', '4']:
        data = results['stream_level_1'].get(par, {})
        cycles = data.get('cycles')
        if cycles:
            speedup = baseline / cycles if baseline else 0
            print(f"{par:<15} {cycles:,<20} {speedup:.2f}x")
        else:
            print(f"{par:<15} {'FAIL':<20} {'N/A':<15}")

    print(f"\n{'Stream Level 2 Sweep:'}")
    print(f"{'Par Factor':<15} {'Cycles':<20} {'Speedup':<15}")
    print("-"*50)
    for par in ['1', '2', '4']:
        data = results['stream_level_2'].get(par, {})
        cycles = data.get('cycles')
        if cycles:
            speedup = baseline / cycles if baseline else 0
            print(f"{par:<15} {cycles:,<20} {speedup:.2f}x")
        else:
            print(f"{par:<15} {'FAIL':<20} {'N/A':<15}")

    print(f"\n{'Combined (par=4 both levels):'}")
    combined = results.get('combined', {})
    cycles = combined.get('cycles')
    if cycles:
        speedup = baseline / cycles if baseline else 0
        print(f"Cycles: {cycles:,}, Speedup: {speedup:.2f}x")
    else:
        print("FAIL")

    print("="*100)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Figure 15b Sweep - Parallelization Level Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full sweep
  python run_figure15b_sweep.py

  # Run with custom timeout
  python run_figure15b_sweep.py --timeout 7200
"""
    )
    parser.add_argument('-b', '--build', type=str, default=None,
                        help='Path to build directory')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help=f'Timeout per benchmark in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')

    args = parser.parse_args()

    build_dir = Path(args.build) if args.build else PROJECT_ROOT / 'build'

    print("="*100)
    print("FIGURE 15b SWEEP - Parallelization Level Comparison")
    print("="*100)
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Sparsity: {SPARSITY * 100:.0f}%")
    print(f"Timeout: {args.timeout}s")
    print(f"Build dir: {build_dir}")
    print("="*100)

    start_time = time.time()

    # Run sweep
    results = run_sweep(build_dir, args.timeout)

    elapsed_time = time.time() - start_time

    # Print summary
    print_summary(results)

    print(f"\nTotal elapsed time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")

    # Save results to JSON
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = ARTIFACT_ROOT / f'figure15b_results_{timestamp}.json'

    results_with_metadata = {
        'metadata': {
            'block_size': BLOCK_SIZE,
            'sparsity': SPARSITY,
            'timeout': args.timeout,
            'elapsed_time': elapsed_time,
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
