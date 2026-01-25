#!/usr/bin/env python3
"""
Figure 17: Dataflow Order Sweep

This script runs the end-to-end pipeline for multiple dataflow orders
and collects cycle counts to analyze the impact of different loop orderings.
Uses nested_matmuls.mlir with KarateClub dataset.
"""

import subprocess
import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Dataflow orders to sweep (indices into the topological sort orderings)
DATAFLOW_ORDERS = [0, 1, 2, 3, 4, 5, 12, 13, 14, 16, 17, 18, 19, 22, 23]

# Labels for each dataflow order (from the original plotting script)
ORDER_LABELS = [
    'ikjl', 'iklj', 'ijkl', 'ijlk', 'ilkj', 'iljk',
    'jikl', 'jilk', 'jkil', 'jlik', 'jlki',
    'likj', 'lijk', 'ljik', 'ljki'
]

# Default test case for Figure 17: nested matmuls with dynamic dimensions
DEFAULT_MLIR = "fuseflow-compiler/tests/kernels/nested_matmuls.mlir"
DEFAULT_DATASET = "KarateClub"
DEFAULT_INDATA = "User"  # Dummy value, always passed


def run_dataflow_order(mlir_file: str, order_index: int, build_dir: str,
                       sparsity: float, par_factor: int,
                       dataset: str, indata: str, outformat: str,
                       use_gen: bool = True, skip_iterate_locate: bool = False) -> dict:
    """
    Run end-to-end for a specific dataflow order.

    Returns:
        dict with keys: order_index, cycles, success, error (if any)
    """
    result = {
        'order_index': order_index,
        'cycles': None,
        'success': False,
        'error': None
    }

    cmd = [
        sys.executable, 'scripts/run_end_to_end.py',
        '--infile', mlir_file,
        '--build', build_dir,
        '--loop-order-index', str(order_index),
        '-sp', str(sparsity),
        '-par', str(par_factor),
        '-inDataset', dataset,
        '-inData', indata,
        '--outformat', outformat,
    ]

    if use_gen:
        cmd.append('--useGen')

    if skip_iterate_locate:
        cmd.append('--skip-iterate-locate')

    print(f"\n{'='*60}")
    print(f"Running dataflow order {order_index}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per order
            cwd=str(Path(__file__).parent.parent)
        )

        output = proc.stdout + proc.stderr
        print(output)

        # Parse cycle count from output
        # Looking for "Simulation completed (True, <cycles>)" pattern
        for line in output.split('\n'):
            if 'Simulation completed' in line:
                # Parse: Simulation completed (True, 12345)
                try:
                    if '(True,' in line:
                        cycles_str = line.split('(True,')[1].split(')')[0].strip()
                        result['cycles'] = int(cycles_str)
                        result['success'] = True
                    elif '(False,' in line:
                        result['error'] = 'Simulation returned False'
                except (IndexError, ValueError) as e:
                    result['error'] = f'Failed to parse cycles: {e}'
                break

        if result['cycles'] is None and result['error'] is None:
            result['error'] = 'No simulation output found'

    except subprocess.TimeoutExpired:
        result['error'] = 'Timeout (600s)'
    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description='Figure 17: Dataflow Order Sweep')
    parser.add_argument('--infile', type=str, default=DEFAULT_MLIR,
                        help='MLIR file to sweep dataflow orders on')
    parser.add_argument('-b', '--build', type=str, default='fuseflow-compiler/build',
                        help='Path to build directory')
    parser.add_argument('-sp', '--sparsity', type=float, default=0.9,
                        help='Sparsity level')
    parser.add_argument('-par', '--parfactor', type=int, default=1,
                        help='Parallel factor')
    parser.add_argument('-inDataset', type=str, default=DEFAULT_DATASET,
                        help='Dataset name (default: KarateClub)')
    parser.add_argument('-inData', type=str, default=DEFAULT_INDATA,
                        help='Data name (default: User)')
    parser.add_argument('--outformat', '-oformat', type=str, default='UNC',
                        help='Output format (default: UNC)')
    parser.add_argument('--output', type=str, default='figure17_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--orders', type=str, default=None,
                        help='Comma-separated list of order indices to run (default: all)')
    parser.add_argument('--no-gen', action='store_true',
                        help='Do not use synthetic data generator')
    parser.add_argument('--skip-iterate-locate', action='store_true', default=False,
                        help='Skip the InsertIterateLocate pass (default: False, pass is applied)')
    args = parser.parse_args()

    # Determine which orders to run
    if args.orders:
        orders = [int(x.strip()) for x in args.orders.split(',')]
    else:
        orders = DATAFLOW_ORDERS

    print(f"Figure 17: Dataflow Order Sweep")
    print(f"MLIR file: {args.infile}")
    print(f"Build dir: {args.build}")
    print(f"Sparsity: {args.sparsity}")
    print(f"Par factor: {args.parfactor}")
    print(f"Dataset: {args.inDataset}")
    print(f"InData: {args.inData}")
    print(f"Output format: {args.outformat}")
    print(f"Orders to sweep: {orders}")
    print(f"Output file: {args.output}")

    results = {
        'metadata': {
            'mlir_file': args.infile,
            'sparsity': args.sparsity,
            'par_factor': args.parfactor,
            'dataset': args.inDataset,
            'indata': args.inData,
            'outformat': args.outformat,
            'timestamp': datetime.now().isoformat(),
            'orders_requested': orders,
        },
        'results': []
    }

    for i, order_idx in enumerate(orders):
        print(f"\n[{i+1}/{len(orders)}] Running order {order_idx}...")

        result = run_dataflow_order(
            mlir_file=args.infile,
            order_index=order_idx,
            build_dir=args.build,
            sparsity=args.sparsity,
            par_factor=args.parfactor,
            dataset=args.inDataset,
            indata=args.inData,
            outformat=args.outformat,
            use_gen=not args.no_gen,
            skip_iterate_locate=args.skip_iterate_locate
        )

        results['results'].append(result)

        # Save intermediate results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        if result['success']:
            print(f"  -> Order {order_idx}: {result['cycles']} cycles")
        else:
            print(f"  -> Order {order_idx}: FAILED - {result['error']}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    successful = [r for r in results['results'] if r['success']]
    failed = [r for r in results['results'] if not r['success']]

    print(f"Successful: {len(successful)}/{len(orders)}")
    print(f"Failed: {len(failed)}/{len(orders)}")

    if successful:
        cycles = [r['cycles'] for r in successful]
        max_cycles = max(cycles)
        min_cycles = min(cycles)
        print(f"\nCycle range: {min_cycles} - {max_cycles}")
        print(f"Speedup range: {max_cycles/max_cycles:.2f}x - {max_cycles/min_cycles:.2f}x (over worst)")

        print("\nResults by order:")
        for r in sorted(successful, key=lambda x: x['cycles']):
            speedup = max_cycles / r['cycles']
            print(f"  Order {r['order_index']:2d}: {r['cycles']:>10d} cycles ({speedup:.2f}x speedup)")

    if failed:
        print("\nFailed orders:")
        for r in failed:
            print(f"  Order {r['order_index']}: {r['error']}")

    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
