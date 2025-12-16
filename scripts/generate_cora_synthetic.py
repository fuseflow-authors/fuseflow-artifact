#!/usr/bin/env python3
"""
Generate synthetic Cora-sized datasets for GCN sparsity sweep.
Creates data with correct tensor shapes for each sparsity level.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add paths
ARTIFACT_ROOT = Path(__file__).parent.parent
SAM_ROOT = ARTIFACT_ROOT / "sam"
sys.path.insert(0, str(SAM_ROOT))

from sam.onyx.generate_matrices import MatrixGenerator

# Cora dataset dimensions
CORA_NODES = 1767
CORA_FEATURES = 50
HIDDEN_DIM = 16
OUTPUT_DIM = 121

SEED = 25
np.random.seed(SEED)


def generate_sparse_adj_matrix(num_nodes, sparsity, output_dir, tensor_name="t0-0,1"):
    """
    Generate a sparse adjacency matrix with specified sparsity.

    Args:
        num_nodes: Number of nodes in the graph
        sparsity: Sparsity level (0.0 to 1.0)
        output_dir: Output directory path
        tensor_name: Name for the tensor files
    """
    print(f"  Generating adjacency matrix: {num_nodes}x{num_nodes}, sparsity={sparsity:.3f}")

    # Create adjacency matrix generator
    adj_gen = MatrixGenerator(
        tensor_name,
        shape=(num_nodes, num_nodes),
        sparsity=sparsity,
        dump_dir=str(output_dir),
        format="CSF",
        seed=SEED
    )

    adj_gen.dump_outputs("CSF")
    print(f"    Saved to {output_dir}/{tensor_name}*")


def generate_dense_feature_matrix(num_nodes, num_features, output_dir, tensor_name="t1-0,1"):
    """
    Generate a dense feature matrix.
    """
    print(f"  Generating feature matrix: {num_nodes}x{num_features}")

    # Create feature matrix (can be sparse or dense)
    feat_gen = MatrixGenerator(
        tensor_name,
        shape=(num_nodes, num_features),
        sparsity=0.0,  # Dense features
        dump_dir=str(output_dir),
        format="CSF",
        seed=SEED
    )

    feat_gen.dump_outputs("CSF")
    print(f"    Saved to {output_dir}/{tensor_name}*")


def generate_output_matrix(num_nodes, num_features, output_dir, tensor_name="t2-0,1"):
    """
    Generate output matrix placeholder.
    """
    print(f"  Generating output matrix: {num_nodes}x{num_features}")

    out_gen = MatrixGenerator(
        tensor_name,
        shape=(num_nodes, num_features),
        sparsity=0.0,  # Dense output
        dump_dir=str(output_dir),
        format="CSF",
        seed=SEED
    )

    out_gen.dump_outputs("CSF")
    print(f"    Saved to {output_dir}/{tensor_name}*")


def main():
    print("="*70)
    print("Generating Cora-sized Synthetic Data for GCN Sparsity Sweep")
    print("="*70)
    print(f"Dimensions: {CORA_NODES} nodes, {CORA_FEATURES} features")
    print(f"Hidden: {HIDDEN_DIM}, Output: {OUTPUT_DIM}")
    print("="*70)

    # Sparsity levels to generate
    sparsity_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]

    # Base data directory
    base_dir = ARTIFACT_ROOT / "data_synthetic_cora"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate for each sparsity level
    for sparsity in sparsity_levels:
        percentage = sparsity * 100
        print(f"\n{'='*70}")
        print(f"Sparsity: {percentage:.1f}%")
        print(f"{'='*70}")

        # Create directory for this sparsity level
        sparsity_dir = base_dir / f"sparsity_{percentage:.1f}"
        sparsity_dir.mkdir(parents=True, exist_ok=True)

        # Generate adjacency matrix (sparse, varies with sparsity)
        generate_sparse_adj_matrix(CORA_NODES, sparsity, sparsity_dir, "t0-0,1")

        # Generate feature matrix (same for all sparsity levels)
        generate_dense_feature_matrix(CORA_NODES, CORA_FEATURES, sparsity_dir, "t1-0,1")

        # Generate output matrix placeholder
        generate_output_matrix(CORA_NODES, CORA_FEATURES, sparsity_dir, "t2-0,1")

        print(f"  ✓ Complete: {sparsity_dir}")

    print("\n" + "="*70)
    print("Data generation complete!")
    print(f"Data saved to: {base_dir}")
    print("="*70)

    # Print usage instructions
    print("\nUsage:")
    print("  To use with run_end_to_end.py, specify:")
    print(f"    --matdir {base_dir}/sparsity_50.0")
    print("  or")
    print(f"    --matdir {base_dir}/sparsity_99.0")


if __name__ == "__main__":
    main()
