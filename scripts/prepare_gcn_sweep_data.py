#!/usr/bin/env python3
"""
Prepare data directories for GCN sparsity sweep.
This script pre-generates all tensor data needed for the sweep,
with separate directories for each sparsity level for adjacency matrices.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add parent directory to path to import from sam
ARTIFACT_ROOT = Path(__file__).parent.parent
SAM_ROOT = ARTIFACT_ROOT / "sam"
sys.path.insert(0, str(SAM_ROOT))

from sam.onyx.synthetic.generate_random_mats import generate_matrix

# Cora dataset dimensions
CORA_NODES = 1767
CORA_FEATURES = 50
HIDDEN_DIM = 16
OUTPUT_DIM = 121

# Sparsity levels
SPARSITY_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]

SEED = 25
OUTPUT_FORMAT = "CSF"


def generate_tensor(output_dir, name, shape, mode_ordering, sparsity, seed, output_format):
    """Generate a single tensor using the sam generator."""
    cmd = [
        'python',
        str(SAM_ROOT / 'sam' / 'onyx' / 'synthetic' / 'generate_random_mats.py'),
        '--output_dir', str(output_dir),
        '--name', name,
        '--shape', *[str(s) for s in shape],
        '--mode_ordering', *[str(m) for m in mode_ordering],
        '--sparsity', str(sparsity),
        '--seed', str(seed),
        '--output_format', output_format
    ]

    print(f"  Generating {name} with shape {shape}, sparsity {sparsity}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr}")
        return False

    return True


def prepare_gcn_adj_data(base_dir, component_name, sparsity_levels):
    """
    Prepare data for GCN adjacency operations (gcn_adj_x1, gcn_adj_x2).
    These need separate directories for each sparsity level.
    """
    print(f"\nPreparing data for {component_name}")

    for sparsity in sparsity_levels:
        percentage = sparsity * 100
        output_dir = base_dir / f"sparsity_{percentage:.1f}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Sparsity: {percentage:.1f}%")

        # Adjacency matrix (sparse)
        generate_tensor(
            output_dir,
            "t0-0,1",  # Adjacency matrix name
            [CORA_NODES, CORA_NODES],
            [0, 1],
            sparsity,
            SEED,
            OUTPUT_FORMAT
        )

        # Feature matrix (dense/semi-sparse - use default sparsity)
        generate_tensor(
            output_dir,
            "t1-0,1",  # Feature matrix name
            [CORA_NODES, CORA_FEATURES],
            [0, 1],
            sparsity,  # Can use fixed sparsity or same as adj
            SEED,
            OUTPUT_FORMAT
        )

        # Output tensor
        generate_tensor(
            output_dir,
            "t2-0,1",  # Output tensor name
            [CORA_NODES, CORA_FEATURES],
            [0, 1],
            0.0,  # Dense output
            SEED,
            OUTPUT_FORMAT
        )


def prepare_gcn_dense_data(base_dir, component_name, tensor_configs):
    """
    Prepare data for dense GCN operations (linear, relu, softmax, etc.).
    These only need one data directory since they're sparsity-independent.
    """
    print(f"\nPreparing data for {component_name}")

    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for tensor_name, shape, mode_order, sparsity in tensor_configs:
        generate_tensor(
            output_dir,
            tensor_name,
            shape,
            mode_order,
            sparsity,
            SEED,
            OUTPUT_FORMAT
        )


def prepare_gcn_fused_data(base_dir, variant_name, sparsity_levels):
    """
    Prepare data for fused GCN variants (partial and fully fused).
    These need separate directories for each sparsity level.
    """
    print(f"\nPreparing data for {variant_name}")

    for sparsity in sparsity_levels:
        percentage = sparsity * 100
        output_dir = base_dir / f"sparsity_{percentage:.1f}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Sparsity: {percentage:.1f}%")

        # Feature matrix
        generate_tensor(
            output_dir,
            "t0-0,1",
            [CORA_NODES, CORA_FEATURES],
            [0, 1],
            0.0,  # Dense features
            SEED,
            OUTPUT_FORMAT
        )

        # Adjacency matrix (sparse - varies with sparsity)
        generate_tensor(
            output_dir,
            "t1-0,1",
            [CORA_NODES, CORA_NODES],
            [0, 1],
            sparsity,
            SEED,
            OUTPUT_FORMAT
        )


def main():
    parser = argparse.ArgumentParser(description='Prepare data for GCN sparsity sweep')
    parser.add_argument('--output-base', type=str,
                        default=str(ARTIFACT_ROOT / "data_sweep"),
                        help='Base directory for generated data')
    parser.add_argument('--sparsity-levels', type=float, nargs='+',
                        default=SPARSITY_LEVELS,
                        help='Sparsity levels to generate')
    args = parser.parse_args()

    base_dir = Path(args.output_base)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("GCN Sparsity Sweep - Data Preparation")
    print("="*60)
    print(f"Output base directory: {base_dir}")
    print(f"Sparsity levels: {args.sparsity_levels}")
    print("="*60)

    # Sparse components (vary with sparsity)
    sparse_components = ["gcn_adj_x1", "gcn_adj_x2"]

    for component in sparse_components:
        component_dir = base_dir / "unfused" / component
        prepare_gcn_adj_data(component_dir, component, args.sparsity_levels)

    # Dense components (sparsity-independent)
    # Note: These are simplified - you may need to adjust based on actual tensor requirements
    dense_components = {
        "gcn_linear1_mul": [],  # Add tensor configs as needed
        "gcn_linear1_bias": [],
        "gcn_relu": [],
        "gcn_linear2_mul": [],
        "gcn_linear2_bias": [],
        "gcn_softmax": [],
    }

    # Fused variants
    fused_variants = ["partial_fused_layer1", "partial_fused_layer2", "fully_fused"]

    for variant in fused_variants:
        variant_dir = base_dir / variant
        prepare_gcn_fused_data(variant_dir, variant, args.sparsity_levels)

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
