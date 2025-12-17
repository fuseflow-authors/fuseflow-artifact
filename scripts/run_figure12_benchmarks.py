#!/usr/bin/env python3
"""
Figure 12 Comprehensive Benchmark Script for FuseFlow Paper

Runs all benchmarks required for Figure 12 artifact evaluation:
1. SAE (Sparse Autoencoder) - 3 datasets: ImageNet, NIH-CXR, LUNA16
2. GCN - 5 datasets: cora, cora_ml, dblp, collab, mag
3. GraphSAGE - 5 datasets: cora, cora_ml, dblp, collab, mag
4. GPT-3 w/ BigBird - 3 block sizes: 16, 32, 64

Each model runs 3 fusion configurations:
- Fully Fused
- Partially Fused
- Unfused

Results are saved to JSON for plotting.
"""

import subprocess
import os
import sys
import argparse
import json
import re
import time
import threading
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# Add the scripts directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
ARTIFACT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import PROJECT_ROOT, ARTIFACT_ROOT

# 1-day timeout in seconds (86400)
DEFAULT_TIMEOUT = 86400

# Default number of parallel workers
DEFAULT_WORKERS = 2

# Thread-safe print lock
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()

# Default sparsity values per model type
DEFAULT_SPARSITY = {
    'sae': 0.9,        # 90% for autoencoders
    'gcn': 0.5,        # 50% for GCN
    'graphsage': 0.5,  # 50% for GraphSAGE
    'gpt3': 0.9,       # 90% for GPT-3/BigBird
}

# ============================================================================
# SAE (Sparse Autoencoder) Configuration
# ============================================================================
SAE_CONFIG = {
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
        'name': 'NIH-CXR',
        'resolution': '1024x1024',
        'input_dim': 1048576,
        'hidden_dim': 512,
        'fused_mlir': 'autoencoder_nih_batched.mlir',
        'encoder_fused_mlir': 'autoencoder_nih_encoder_fused.mlir',
        'decoder_fused_mlir': 'autoencoder_nih_decoder_fused.mlir',
        'unfused_prefix': 'autoencoder_nih_unfused',
    },
    'luna16': {
        'name': 'LUNA16',
        'resolution': '512x512',
        'input_dim': 262144,
        'hidden_dim': 512,
        'fused_mlir': 'autoencoder_luna16_batched.mlir',
        'encoder_fused_mlir': 'autoencoder_luna16_encoder_fused.mlir',
        'decoder_fused_mlir': 'autoencoder_luna16_decoder_fused.mlir',
        'unfused_prefix': 'autoencoder_luna16_unfused',
    },
}

# ============================================================================
# GCN Configuration
# ============================================================================
GCN_CONFIG = {
    'cora': {
        'name': 'Cora',
        'dataset_type': 'Planetoid',
        'dataset_name': 'Cora',
        'fused_mlir': 'gcn/gcn_sparse.mlir',
        'unfused_dir': 'gcn_unfused',
        'unfused_ops': [
            'gcn_adj_x1.mlir',
            'gcn_linear1_mul.mlir',
            'gcn_linear1_bias.mlir',
            'gcn_relu.mlir',
            'gcn_adj_x2.mlir',
            'gcn_linear2_mul.mlir',
            'gcn_linear2_bias.mlir',
            'gcn_softmax.mlir',
        ],
        # Partially fused: Layer 1 fused, Layer 2 fused
        'partial_ops': [
            'one_layer_gcn.mlir',  # Layer 1 fused
            'one_layer_gcn2.mlir',  # Layer 2 fused
        ],
    },
    'cora_ml': {
        'name': 'Cora-ML',
        'dataset_type': 'CitationFull',
        'dataset_name': 'cora_ml',
        'fused_mlir': 'gcn/gcn_sparse.mlir',
        'unfused_dir': 'gcn_unfused',
        'unfused_ops': [
            'gcn_adj_x1.mlir',
            'gcn_linear1_mul.mlir',
            'gcn_linear1_bias.mlir',
            'gcn_relu.mlir',
            'gcn_adj_x2.mlir',
            'gcn_linear2_mul.mlir',
            'gcn_linear2_bias.mlir',
            'gcn_softmax.mlir',
        ],
        'partial_ops': [
            'one_layer_gcn.mlir',
            'one_layer_gcn2.mlir',
        ],
    },
    'dblp': {
        'name': 'DBLP',
        'dataset_type': 'CitationFull',
        'dataset_name': 'dblp',
        'fused_mlir': 'gcn/gcn_sparse.mlir',
        'unfused_dir': 'gcn_unfused',
        'unfused_ops': [
            'gcn_adj_x1.mlir',
            'gcn_linear1_mul.mlir',
            'gcn_linear1_bias.mlir',
            'gcn_relu.mlir',
            'gcn_adj_x2.mlir',
            'gcn_linear2_mul.mlir',
            'gcn_linear2_bias.mlir',
            'gcn_softmax.mlir',
        ],
        'partial_ops': [
            'one_layer_gcn.mlir',
            'one_layer_gcn2.mlir',
        ],
    },
    'collab': {
        'name': 'OGB-Collab',
        'dataset_type': 'ogbl',
        'dataset_name': 'ogbl-collab',
        'fused_mlir': 'gcn/gcn_sparse.mlir',
        'unfused_dir': 'gcn_unfused',
        'unfused_ops': [
            'gcn_adj_x1.mlir',
            'gcn_linear1_mul.mlir',
            'gcn_linear1_bias.mlir',
            'gcn_relu.mlir',
            'gcn_adj_x2.mlir',
            'gcn_linear2_mul.mlir',
            'gcn_linear2_bias.mlir',
            'gcn_softmax.mlir',
        ],
        'partial_ops': [
            'one_layer_gcn.mlir',
            'one_layer_gcn2.mlir',
        ],
    },
    'mag': {
        'name': 'OGB-MAG',
        'dataset_type': 'ogbn',
        'dataset_name': 'ogbn-mag',
        'fused_mlir': 'gcn/gcn_sparse.mlir',
        'unfused_dir': 'gcn_unfused',
        'unfused_ops': [
            'gcn_adj_x1.mlir',
            'gcn_linear1_mul.mlir',
            'gcn_linear1_bias.mlir',
            'gcn_relu.mlir',
            'gcn_adj_x2.mlir',
            'gcn_linear2_mul.mlir',
            'gcn_linear2_bias.mlir',
            'gcn_softmax.mlir',
        ],
        'partial_ops': [
            'one_layer_gcn.mlir',
            'one_layer_gcn2.mlir',
        ],
    },
}

# ============================================================================
# GraphSAGE Configuration
# ============================================================================
GRAPHSAGE_CONFIG = {
    'cora': {
        'name': 'Cora',
        'dataset_type': 'Planetoid',
        'dataset_name': 'Cora',
        'fused_mlir': 'graphsage/graphsage_sparse.mlir',
        'unfused_dir': 'graphsage_unfused',
        'unfused_ops': [
            'gcn_adj_x1.mlir',
            'gcn_linear1_mul.mlir',
            'gcn_linear1_bias.mlir',
            'graphsage_linear_adds.mlir',
            'gcn_relu.mlir',
            'gcn_adj_x2.mlir',
            'gcn_linear2_mul.mlir',
            'gcn_linear2_bias.mlir',
        ],
        'partial_ops': [
            'one_layer_graphsage.mlir',
            'one_layer_graphsage2.mlir',
        ],
    },
    'cora_ml': {
        'name': 'Cora-ML',
        'dataset_type': 'CitationFull',
        'dataset_name': 'cora_ml',
        'fused_mlir': 'graphsage/graphsage_sparse.mlir',
        'unfused_dir': 'graphsage_unfused',
        'unfused_ops': [
            'gcn_adj_x1.mlir',
            'gcn_linear1_mul.mlir',
            'gcn_linear1_bias.mlir',
            'graphsage_linear_adds.mlir',
            'gcn_relu.mlir',
            'gcn_adj_x2.mlir',
            'gcn_linear2_mul.mlir',
            'gcn_linear2_bias.mlir',
        ],
        'partial_ops': [
            'one_layer_graphsage.mlir',
            'one_layer_graphsage2.mlir',
        ],
    },
    'dblp': {
        'name': 'DBLP',
        'dataset_type': 'CitationFull',
        'dataset_name': 'dblp',
        'fused_mlir': 'graphsage/graphsage_sparse.mlir',
        'unfused_dir': 'graphsage_unfused',
        'unfused_ops': [
            'gcn_adj_x1.mlir',
            'gcn_linear1_mul.mlir',
            'gcn_linear1_bias.mlir',
            'graphsage_linear_adds.mlir',
            'gcn_relu.mlir',
            'gcn_adj_x2.mlir',
            'gcn_linear2_mul.mlir',
            'gcn_linear2_bias.mlir',
        ],
        'partial_ops': [
            'one_layer_graphsage.mlir',
            'one_layer_graphsage2.mlir',
        ],
    },
    'collab': {
        'name': 'OGB-Collab',
        'dataset_type': 'ogbl',
        'dataset_name': 'ogbl-collab',
        'fused_mlir': 'graphsage/graphsage_sparse.mlir',
        'unfused_dir': 'graphsage_unfused',
        'unfused_ops': [
            'gcn_adj_x1.mlir',
            'gcn_linear1_mul.mlir',
            'gcn_linear1_bias.mlir',
            'graphsage_linear_adds.mlir',
            'gcn_relu.mlir',
            'gcn_adj_x2.mlir',
            'gcn_linear2_mul.mlir',
            'gcn_linear2_bias.mlir',
        ],
        'partial_ops': [
            'one_layer_graphsage.mlir',
            'one_layer_graphsage2.mlir',
        ],
    },
    'mag': {
        'name': 'OGB-MAG',
        'dataset_type': 'ogbn',
        'dataset_name': 'ogbn-mag',
        'fused_mlir': 'graphsage/graphsage_sparse.mlir',
        'unfused_dir': 'graphsage_unfused',
        'unfused_ops': [
            'gcn_adj_x1.mlir',
            'gcn_linear1_mul.mlir',
            'gcn_linear1_bias.mlir',
            'graphsage_linear_adds.mlir',
            'gcn_relu.mlir',
            'gcn_adj_x2.mlir',
            'gcn_linear2_mul.mlir',
            'gcn_linear2_bias.mlir',
        ],
        'partial_ops': [
            'one_layer_graphsage.mlir',
            'one_layer_graphsage2.mlir',
        ],
    },
}

# ============================================================================
# GPT-3 / BigBird Configuration
# ============================================================================
# Fully Fused = 3 components (fuses start of next decoder into current):
#   1. Layernorm + QKV projections
#   2. Fused Attention (multihead_attention with --useGen and block size)
#   3. Out linear + residual + layernorm + FFN + residual + next decoder's layernorm + QKV projections
#
# Partially Fused = 3 components (fuses remaining parts of each decoder separately):
#   1. Layernorm + QKV projections
#   2. Fused Attention (multihead_attention with --useGen and block size)
#   3. Out linear + residual + layernorm + FFN + residual (WITHOUT next decoder's LN and QKV)
#
# Unfused = 18 individual operations
GPT3_CONFIG = {
    '16': {
        'name': 'BigBird-16',
        'block_size': 16,
        # Fully fused: 3 components (with next decoder's LN+QKV fused in)
        'fused_ops': [
            {'file': 'gpt-3/layernorm_linear.mlir', 'use_gen': False, 'desc': 'Layernorm + QKV projections'},
            {'file': 'gpt-3/multihead_attention.mlir', 'use_gen': True, 'desc': 'Fused Attention'},
            {'file': 'gpt-3/outLinear_layernorm_FFN_layernorm_QKVprojection.mlir', 'use_gen': False, 'desc': 'OutLinear + Residual + LN + FFN + Residual + LN + QKV'},
        ],
        # Partially fused: 3 components (without next decoder's LN+QKV)
        'partial_fused_ops': [
            {'file': 'gpt-3/layernorm_linear.mlir', 'use_gen': False, 'desc': 'Layernorm + QKV projections'},
            {'file': 'gpt-3/multihead_attention.mlir', 'use_gen': True, 'desc': 'Fused Attention'},
            {'file': 'gpt-3/outLinear_residual_layernorm_FFN_residual.mlir', 'use_gen': False, 'desc': 'OutLinear + Residual + LN + FFN + Residual'},
        ],
        # Unfused: 18 individual operations
        'unfused_ops': [
            {'file': 'gpt-3_unfused/layernorm1.mlir', 'use_gen': False, 'desc': 'Layernorm1'},
            {'file': 'gpt-3_unfused/projection_mul.mlir', 'use_gen': False, 'desc': 'QKV projection matmul'},
            {'file': 'gpt-3_unfused/projection_bias.mlir', 'use_gen': False, 'desc': 'QKV projection bias'},
            {'file': 'gpt-3_unfused/mhaQK_mul.mlir', 'use_gen': False, 'desc': 'Q*Kt'},
            {'file': 'gpt-3_unfused/mhaQK_mask.mlir', 'use_gen': True, 'desc': 'Attention Mask'},
            {'file': 'gpt-3_unfused/mhaQK_scale.mlir', 'use_gen': True, 'desc': 'Scaling'},
            {'file': 'gpt-3_unfused/mhaQK_softmax.mlir', 'use_gen': True, 'desc': 'Softmax'},
            {'file': 'gpt-3_unfused/mhaQK_mul2.mlir', 'use_gen': True, 'desc': 'QKt*V'},
            {'file': 'gpt-3_unfused/outLinear_mul.mlir', 'use_gen': False, 'desc': 'Out linear matmul'},
            {'file': 'gpt-3_unfused/outLinear_bias.mlir', 'use_gen': False, 'desc': 'Out linear bias'},
            {'file': 'gpt-3_unfused/residual1.mlir', 'use_gen': False, 'desc': 'Residual 1'},
            {'file': 'gpt-3_unfused/layernorm2.mlir', 'use_gen': False, 'desc': 'Layernorm2'},
            {'file': 'gpt-3_unfused/ffnLinear1_mul.mlir', 'use_gen': False, 'desc': 'FFN linear1 matmul'},
            {'file': 'gpt-3_unfused/ffnLinear1_bias.mlir', 'use_gen': False, 'desc': 'FFN linear1 bias'},
            {'file': 'gpt-3_unfused/ffn_relu.mlir', 'use_gen': False, 'desc': 'ReLU/GeLU'},
            {'file': 'gpt-3_unfused/ffnLinear2_mul.mlir', 'use_gen': False, 'desc': 'FFN linear2 matmul'},
            {'file': 'gpt-3_unfused/ffnLinear2_bias.mlir', 'use_gen': False, 'desc': 'FFN linear2 bias'},
            {'file': 'gpt-3_unfused/residual2.mlir', 'use_gen': False, 'desc': 'Residual 2'},
        ],
    },
    '32': {
        'name': 'BigBird-32',
        'block_size': 32,
        'fused_ops': [
            {'file': 'gpt-3/layernorm_linear.mlir', 'use_gen': False, 'desc': 'Layernorm + QKV projections'},
            {'file': 'gpt-3/multihead_attention.mlir', 'use_gen': True, 'desc': 'Fused Attention'},
            {'file': 'gpt-3/outLinear_layernorm_FFN_layernorm_QKVprojection.mlir', 'use_gen': False, 'desc': 'OutLinear + Residual + LN + FFN + Residual + LN + QKV'},
        ],
        'partial_fused_ops': [
            {'file': 'gpt-3/layernorm_linear.mlir', 'use_gen': False, 'desc': 'Layernorm + QKV projections'},
            {'file': 'gpt-3/multihead_attention.mlir', 'use_gen': True, 'desc': 'Fused Attention'},
            {'file': 'gpt-3/outLinear_residual_layernorm_FFN_residual.mlir', 'use_gen': False, 'desc': 'OutLinear + Residual + LN + FFN + Residual'},
        ],
        'unfused_ops': [
            {'file': 'gpt-3_unfused/layernorm1.mlir', 'use_gen': False, 'desc': 'Layernorm1'},
            {'file': 'gpt-3_unfused/projection_mul.mlir', 'use_gen': False, 'desc': 'QKV projection matmul'},
            {'file': 'gpt-3_unfused/projection_bias.mlir', 'use_gen': False, 'desc': 'QKV projection bias'},
            {'file': 'gpt-3_unfused/mhaQK_mul.mlir', 'use_gen': False, 'desc': 'Q*Kt'},
            {'file': 'gpt-3_unfused/mhaQK_mask.mlir', 'use_gen': True, 'desc': 'Attention Mask'},
            {'file': 'gpt-3_unfused/mhaQK_scale.mlir', 'use_gen': True, 'desc': 'Scaling'},
            {'file': 'gpt-3_unfused/mhaQK_softmax.mlir', 'use_gen': True, 'desc': 'Softmax'},
            {'file': 'gpt-3_unfused/mhaQK_mul2.mlir', 'use_gen': True, 'desc': 'QKt*V'},
            {'file': 'gpt-3_unfused/outLinear_mul.mlir', 'use_gen': False, 'desc': 'Out linear matmul'},
            {'file': 'gpt-3_unfused/outLinear_bias.mlir', 'use_gen': False, 'desc': 'Out linear bias'},
            {'file': 'gpt-3_unfused/residual1.mlir', 'use_gen': False, 'desc': 'Residual 1'},
            {'file': 'gpt-3_unfused/layernorm2.mlir', 'use_gen': False, 'desc': 'Layernorm2'},
            {'file': 'gpt-3_unfused/ffnLinear1_mul.mlir', 'use_gen': False, 'desc': 'FFN linear1 matmul'},
            {'file': 'gpt-3_unfused/ffnLinear1_bias.mlir', 'use_gen': False, 'desc': 'FFN linear1 bias'},
            {'file': 'gpt-3_unfused/ffn_relu.mlir', 'use_gen': False, 'desc': 'ReLU/GeLU'},
            {'file': 'gpt-3_unfused/ffnLinear2_mul.mlir', 'use_gen': False, 'desc': 'FFN linear2 matmul'},
            {'file': 'gpt-3_unfused/ffnLinear2_bias.mlir', 'use_gen': False, 'desc': 'FFN linear2 bias'},
            {'file': 'gpt-3_unfused/residual2.mlir', 'use_gen': False, 'desc': 'Residual 2'},
        ],
    },
    '64': {
        'name': 'BigBird-64',
        'block_size': 64,
        'fused_ops': [
            {'file': 'gpt-3/layernorm_linear.mlir', 'use_gen': False, 'desc': 'Layernorm + QKV projections'},
            {'file': 'gpt-3/multihead_attention.mlir', 'use_gen': True, 'desc': 'Fused Attention'},
            {'file': 'gpt-3/outLinear_layernorm_FFN_layernorm_QKVprojection.mlir', 'use_gen': False, 'desc': 'OutLinear + Residual + LN + FFN + Residual + LN + QKV'},
        ],
        'partial_fused_ops': [
            {'file': 'gpt-3/layernorm_linear.mlir', 'use_gen': False, 'desc': 'Layernorm + QKV projections'},
            {'file': 'gpt-3/multihead_attention.mlir', 'use_gen': True, 'desc': 'Fused Attention'},
            {'file': 'gpt-3/outLinear_residual_layernorm_FFN_residual.mlir', 'use_gen': False, 'desc': 'OutLinear + Residual + LN + FFN + Residual'},
        ],
        'unfused_ops': [
            {'file': 'gpt-3_unfused/layernorm1.mlir', 'use_gen': False, 'desc': 'Layernorm1'},
            {'file': 'gpt-3_unfused/projection_mul.mlir', 'use_gen': False, 'desc': 'QKV projection matmul'},
            {'file': 'gpt-3_unfused/projection_bias.mlir', 'use_gen': False, 'desc': 'QKV projection bias'},
            {'file': 'gpt-3_unfused/mhaQK_mul.mlir', 'use_gen': False, 'desc': 'Q*Kt'},
            {'file': 'gpt-3_unfused/mhaQK_mask.mlir', 'use_gen': True, 'desc': 'Attention Mask'},
            {'file': 'gpt-3_unfused/mhaQK_scale.mlir', 'use_gen': True, 'desc': 'Scaling'},
            {'file': 'gpt-3_unfused/mhaQK_softmax.mlir', 'use_gen': True, 'desc': 'Softmax'},
            {'file': 'gpt-3_unfused/mhaQK_mul2.mlir', 'use_gen': True, 'desc': 'QKt*V'},
            {'file': 'gpt-3_unfused/outLinear_mul.mlir', 'use_gen': False, 'desc': 'Out linear matmul'},
            {'file': 'gpt-3_unfused/outLinear_bias.mlir', 'use_gen': False, 'desc': 'Out linear bias'},
            {'file': 'gpt-3_unfused/residual1.mlir', 'use_gen': False, 'desc': 'Residual 1'},
            {'file': 'gpt-3_unfused/layernorm2.mlir', 'use_gen': False, 'desc': 'Layernorm2'},
            {'file': 'gpt-3_unfused/ffnLinear1_mul.mlir', 'use_gen': False, 'desc': 'FFN linear1 matmul'},
            {'file': 'gpt-3_unfused/ffnLinear1_bias.mlir', 'use_gen': False, 'desc': 'FFN linear1 bias'},
            {'file': 'gpt-3_unfused/ffn_relu.mlir', 'use_gen': False, 'desc': 'ReLU/GeLU'},
            {'file': 'gpt-3_unfused/ffnLinear2_mul.mlir', 'use_gen': False, 'desc': 'FFN linear2 matmul'},
            {'file': 'gpt-3_unfused/ffnLinear2_bias.mlir', 'use_gen': False, 'desc': 'FFN linear2 bias'},
            {'file': 'gpt-3_unfused/residual2.mlir', 'use_gen': False, 'desc': 'Residual 2'},
        ],
    },
}


def run_single_mlir(mlir_file, sparsity, build_dir, parfactor=1,
                    dataset_type=None, dataset_name=None, block_size=None,
                    timeout=DEFAULT_TIMEOUT, use_gen=True, enable_hbm=True):
    """Run a single MLIR file and return the cycle count and full output.

    Args:
        mlir_file: Path to the MLIR file
        sparsity: Weight sparsity (0.0 to 1.0)
        build_dir: Path to build directory
        parfactor: Parallel factor
        dataset_type: Dataset type for GCN/GraphSAGE (e.g., 'Planetoid')
        dataset_name: Dataset name (e.g., 'Cora')
        block_size: Block size for BigBird attention
        timeout: Timeout in seconds
        use_gen: Whether to use --useGen flag (False for GPT-3 which uses dense dims)
        enable_hbm: Whether to enable HBM memory simulation (default: True)

    Returns:
        tuple: (success, cycles, full_output) where full_output is the combined stdout+stderr
    """
    # Set DATA_PATH environment variable if not set
    env = os.environ.copy()
    if 'DATA_PATH' not in env:
        env['DATA_PATH'] = '/tmp/data'

    # Set COMAL_ENABLE_HBM environment variable
    env['COMAL_ENABLE_HBM'] = '1' if enable_hbm else '0'

    cmd = [
        sys.executable, str(SCRIPT_DIR / 'run_end_to_end.py'),
        '--infile', str(mlir_file),
        '--build', str(build_dir),
        '-sp', str(sparsity),
        '-par', str(parfactor),
    ]

    # Add --useGen flag if requested (not for GPT-3)
    if use_gen:
        cmd.append('--useGen')
    else:
        # For ops without data generator, use uncompressed format
        cmd.extend(['--outformat', 'UNC'])

    # Add dataset info if provided
    if dataset_type and dataset_name:
        cmd.extend(['-inDataset', dataset_type, '-inData', dataset_name])

    # Add block size for BigBird
    if block_size:
        cmd.extend(['--block', str(block_size)])

    safe_print(f"\n{'='*60}")
    safe_print(f"Running: {mlir_file.name if hasattr(mlir_file, 'name') else mlir_file}")
    safe_print(f"Command: {' '.join(cmd)}")
    safe_print(f"{'='*60}")

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
            safe_print(f"Result: Success={success}, Cycles={cycles:,}")
        else:
            safe_print(f"Result: Could not parse cycles")
            # Print last part of output for debugging
            if len(output) > 2000:
                safe_print(f"Output (last 2000 chars): ...{output[-2000:]}")
            else:
                safe_print(f"Output: {output}")

        return success, cycles, output

    except subprocess.TimeoutExpired:
        safe_print(f"TIMEOUT after {timeout}s running {mlir_file}")
        return False, None, f"TIMEOUT after {timeout}s"
    except Exception as e:
        safe_print(f"ERROR running {mlir_file}: {e}")
        return False, None, f"ERROR: {e}"


class BenchmarkJob:
    """Represents a single benchmark job to be executed."""
    def __init__(self, job_id, mlir_file, sparsity, build_dir, parfactor=1,
                 dataset_type=None, dataset_name=None, block_size=None,
                 timeout=DEFAULT_TIMEOUT, use_gen=True, enable_hbm=True,
                 output_dir=None, model_name=None, dataset_key=None, fusion_type=None, op_name=None,
                 cleanup_data=True):
        self.job_id = job_id
        self.mlir_file = mlir_file
        self.sparsity = sparsity
        self.build_dir = build_dir
        self.parfactor = parfactor
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.block_size = block_size
        self.timeout = timeout
        self.use_gen = use_gen
        self.enable_hbm = enable_hbm
        self.output_dir = output_dir
        self.model_name = model_name
        self.dataset_key = dataset_key
        self.fusion_type = fusion_type
        self.op_name = op_name
        self.cleanup_data = cleanup_data


def get_data_dir_for_job(job):
    """Get the data directory path for a job based on its MLIR file.

    The data directory follows the pattern: ARTIFACT_ROOT/data/{test_type}/{test_name}_b{block_size}
    where test_type is the parent directory name and test_name is the MLIR file stem.
    """
    mlir_path = Path(job.mlir_file)
    test_type = mlir_path.parent.name
    test_name = mlir_path.stem

    # Always add block size suffix
    data_test_name = f"{test_name}_b{job.block_size}"

    return ARTIFACT_ROOT / "data" / test_type / data_test_name


def cleanup_job_data(job):
    """Clean up the data directory for a completed job to save disk space."""
    data_dir = get_data_dir_for_job(job)
    if data_dir.exists():
        try:
            shutil.rmtree(data_dir)
            safe_print(f"  [Cleanup] Removed data directory: {data_dir}")
        except Exception as e:
            safe_print(f"  [Cleanup] Warning: Failed to remove {data_dir}: {e}")


def run_job(job):
    """Execute a single benchmark job. Used by the thread pool."""
    success, cycles, output = run_single_mlir(
        job.mlir_file, job.sparsity, job.build_dir, job.parfactor,
        dataset_type=job.dataset_type, dataset_name=job.dataset_name,
        block_size=job.block_size, timeout=job.timeout,
        use_gen=job.use_gen, enable_hbm=job.enable_hbm
    )

    # Save output if output_dir is specified
    if job.output_dir and job.model_name and job.fusion_type and job.op_name:
        save_simulator_output(job.output_dir, job.model_name, job.dataset_key or 'default',
                              job.fusion_type, job.op_name, output)

    # Clean up data directory after successful completion to save disk space
    if job.cleanup_data and success:
        cleanup_job_data(job)

    return {
        'job_id': job.job_id,
        'success': success,
        'cycles': cycles,
        'output': output,
        'model_name': job.model_name,
        'dataset_key': job.dataset_key,
        'fusion_type': job.fusion_type,
        'op_name': job.op_name,
    }


def run_jobs_parallel(jobs, max_workers=DEFAULT_WORKERS):
    """Run multiple benchmark jobs in parallel with a limited number of workers.

    Args:
        jobs: List of BenchmarkJob objects
        max_workers: Maximum number of simultaneous jobs

    Returns:
        List of result dictionaries
    """
    results = []
    total_jobs = len(jobs)

    safe_print(f"\n{'#'*80}")
    safe_print(f"# Running {total_jobs} jobs with {max_workers} parallel workers")
    safe_print(f"{'#'*80}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(run_job, job): job for job in jobs}

        completed = 0
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                safe_print(f"\n[Progress: {completed}/{total_jobs}] Completed: {job.op_name}")
            except Exception as e:
                safe_print(f"\n[Progress: {completed}/{total_jobs}] FAILED: {job.op_name} - {e}")
                results.append({
                    'job_id': job.job_id,
                    'success': False,
                    'cycles': None,
                    'output': str(e),
                    'model_name': job.model_name,
                    'dataset_key': job.dataset_key,
                    'fusion_type': job.fusion_type,
                    'op_name': job.op_name,
                })

    return results


def save_simulator_output(output_dir, model_name, dataset_name, fusion_type, op_name, output_text):
    """Save simulator stdout output to organized directory structure.

    Args:
        output_dir: Base output directory
        model_name: Model name (sae, gcn, graphsage, gpt3)
        dataset_name: Dataset/config name (cora, imagenet, 16, etc.)
        fusion_type: Fusion configuration (fully_fused, partially_fused, unfused)
        op_name: Operation/file name
        output_text: Full simulator output text
    """
    # Create directory structure: output_dir/model/dataset/fusion_type/
    dir_path = Path(output_dir) / model_name / dataset_name / fusion_type
    dir_path.mkdir(parents=True, exist_ok=True)

    # Sanitize op_name for filename
    safe_op_name = op_name.replace('/', '_').replace('.mlir', '')
    file_path = dir_path / f"{safe_op_name}.txt"

    with open(file_path, 'w') as f:
        f.write(output_text)

    return file_path


def run_sae_benchmarks(sparsity, build_dir, parfactor, timeout, datasets=None, output_dir=None, enable_hbm=True, workers=DEFAULT_WORKERS, cleanup_data=True):
    """Run Sparse Autoencoder benchmarks for all 3 datasets with parallel execution."""
    models_dir = ARTIFACT_ROOT / 'samml' / 'tests' / 'models'

    if datasets is None:
        datasets = list(SAE_CONFIG.keys())

    # Build all jobs
    jobs = []
    job_id = 0

    for dataset in datasets:
        if dataset not in SAE_CONFIG:
            print(f"WARNING: Unknown SAE dataset {dataset}, skipping")
            continue

        config = SAE_CONFIG[dataset]

        # 1. Fully Fused
        fused_mlir = models_dir / config['fused_mlir']
        if fused_mlir.exists():
            jobs.append(BenchmarkJob(
                job_id=job_id, mlir_file=fused_mlir, sparsity=sparsity,
                build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                model_name='sae', dataset_key=dataset, fusion_type='fully_fused',
                op_name=config['fused_mlir'], cleanup_data=cleanup_data
            ))
            job_id += 1

        # 2. Partially Fused (encoder + decoder)
        for mlir_name in [config['encoder_fused_mlir'], config['decoder_fused_mlir']]:
            mlir_file = models_dir / mlir_name
            if mlir_file.exists():
                jobs.append(BenchmarkJob(
                    job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
                    build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                    use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                    model_name='sae', dataset_key=dataset, fusion_type='partially_fused',
                    op_name=mlir_name, cleanup_data=cleanup_data
                ))
                job_id += 1

        # 3. Unfused (5 separate operations)
        unfused_prefix = config['unfused_prefix']
        unfused_files = [
            f'{unfused_prefix}_1_enc_spmm.mlir',
            f'{unfused_prefix}_2_enc_bias.mlir',
            f'{unfused_prefix}_3_enc_relu.mlir',
            f'{unfused_prefix}_4_dec_spmm.mlir',
            f'{unfused_prefix}_5_dec_bias.mlir',
        ]
        for mlir_name in unfused_files:
            mlir_file = models_dir / mlir_name
            if mlir_file.exists():
                jobs.append(BenchmarkJob(
                    job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
                    build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                    use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                    model_name='sae', dataset_key=dataset, fusion_type='unfused',
                    op_name=mlir_name, cleanup_data=cleanup_data
                ))
                job_id += 1

    # Run all jobs in parallel
    print(f"\n{'='*80}")
    print(f"SAE BENCHMARKS: Running {len(jobs)} jobs with {workers} parallel workers")
    print(f"{'='*80}")

    job_results = run_jobs_parallel(jobs, max_workers=workers)

    # Organize results by dataset
    results = {}
    for dataset in datasets:
        if dataset not in SAE_CONFIG:
            continue
        config = SAE_CONFIG[dataset]
        dataset_results = {}

        # Filter results for this dataset
        dataset_jobs = [r for r in job_results if r.get('dataset_key') == dataset]

        # Fully fused
        fused_jobs = [r for r in dataset_jobs if r.get('fusion_type') == 'fully_fused']
        if fused_jobs:
            r = fused_jobs[0]
            dataset_results['fully_fused'] = {'success': r['success'], 'cycles': r['cycles']}
        else:
            dataset_results['fully_fused'] = {'success': False, 'cycles': None, 'error': 'file_not_found'}

        # Partially fused
        partial_jobs = [r for r in dataset_jobs if r.get('fusion_type') == 'partially_fused']
        partial_breakdown = [{'file': r['op_name'], 'success': r['success'], 'cycles': r['cycles']} for r in partial_jobs]
        partial_success = all(r['success'] for r in partial_jobs) if partial_jobs else False
        partial_cycles = sum(r['cycles'] or 0 for r in partial_jobs) if partial_success else None
        dataset_results['partially_fused'] = {
            'success': partial_success,
            'cycles': partial_cycles,
            'breakdown': partial_breakdown
        }

        # Unfused
        unfused_jobs = [r for r in dataset_jobs if r.get('fusion_type') == 'unfused']
        unfused_breakdown = [{'file': r['op_name'], 'success': r['success'], 'cycles': r['cycles']} for r in unfused_jobs]
        unfused_success = all(r['success'] for r in unfused_jobs) if unfused_jobs else False
        unfused_cycles = sum(r['cycles'] or 0 for r in unfused_jobs) if unfused_success else None
        dataset_results['unfused'] = {
            'success': unfused_success,
            'cycles': unfused_cycles,
            'breakdown': unfused_breakdown
        }

        results[dataset] = dataset_results

    return results


def run_gcn_benchmarks(sparsity, build_dir, parfactor, timeout, datasets=None, output_dir=None, enable_hbm=True, workers=DEFAULT_WORKERS, cleanup_data=True):
    """Run GCN benchmarks for all 5 datasets with parallel execution."""
    models_dir = ARTIFACT_ROOT / 'samml' / 'tests' / 'models'

    if datasets is None:
        datasets = list(GCN_CONFIG.keys())

    # Build all jobs
    jobs = []
    job_id = 0

    for dataset in datasets:
        if dataset not in GCN_CONFIG:
            print(f"WARNING: Unknown GCN dataset {dataset}, skipping")
            continue

        config = GCN_CONFIG[dataset]
        ds_type = config['dataset_type']
        ds_name = config['dataset_name']

        # 1. Fully Fused
        # Skip fully_fused for ogbn-mag (too large for medium mode)
        skip_fully_fused = dataset == 'mag'

        fused_mlir = models_dir / config['fused_mlir']
        if fused_mlir.exists() and not skip_fully_fused:
            jobs.append(BenchmarkJob(
                job_id=job_id, mlir_file=fused_mlir, sparsity=sparsity,
                build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                dataset_type=ds_type, dataset_name=ds_name,
                use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                model_name='gcn', dataset_key=dataset, fusion_type='fully_fused',
                op_name=config['fused_mlir'], cleanup_data=cleanup_data
            ))
            job_id += 1

        # 2. Partially Fused (layer-wise)
        for op in config.get('partial_ops', []):
            mlir_file = models_dir / config['unfused_dir'] / op
            if mlir_file.exists():
                jobs.append(BenchmarkJob(
                    job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
                    build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                    dataset_type=ds_type, dataset_name=ds_name,
                    use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                    model_name='gcn', dataset_key=dataset, fusion_type='partially_fused',
                    op_name=op, cleanup_data=cleanup_data
                ))
                job_id += 1

        # 3. Unfused (op-by-op)
        for op in config['unfused_ops']:
            mlir_file = models_dir / config['unfused_dir'] / op
            if mlir_file.exists():
                jobs.append(BenchmarkJob(
                    job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
                    build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                    dataset_type=ds_type, dataset_name=ds_name,
                    use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                    model_name='gcn', dataset_key=dataset, fusion_type='unfused',
                    op_name=op, cleanup_data=cleanup_data
                ))
                job_id += 1

    # Run all jobs in parallel
    print(f"\n{'='*80}")
    print(f"GCN BENCHMARKS: Running {len(jobs)} jobs with {workers} parallel workers")
    print(f"{'='*80}")

    job_results = run_jobs_parallel(jobs, max_workers=workers)

    # Organize results by dataset
    results = {}
    for dataset in datasets:
        if dataset not in GCN_CONFIG:
            continue
        config = GCN_CONFIG[dataset]
        dataset_results = {}

        # Filter results for this dataset
        dataset_jobs = [r for r in job_results if r.get('dataset_key') == dataset]

        # Fully fused
        fused_jobs = [r for r in dataset_jobs if r.get('fusion_type') == 'fully_fused']
        if fused_jobs:
            r = fused_jobs[0]
            dataset_results['fully_fused'] = {'success': r['success'], 'cycles': r['cycles']}
        else:
            dataset_results['fully_fused'] = {'success': False, 'cycles': None, 'error': 'file_not_found'}

        # Partially fused
        partial_jobs = [r for r in dataset_jobs if r.get('fusion_type') == 'partially_fused']
        partial_breakdown = [{'file': r['op_name'], 'success': r['success'], 'cycles': r['cycles']} for r in partial_jobs]
        partial_success = all(r['success'] for r in partial_jobs) if partial_jobs else False
        partial_cycles = sum(r['cycles'] or 0 for r in partial_jobs) if partial_success else None
        dataset_results['partially_fused'] = {
            'success': partial_success,
            'cycles': partial_cycles,
            'breakdown': partial_breakdown
        }

        # Unfused
        unfused_jobs = [r for r in dataset_jobs if r.get('fusion_type') == 'unfused']
        unfused_breakdown = [{'file': r['op_name'], 'success': r['success'], 'cycles': r['cycles']} for r in unfused_jobs]
        unfused_success = all(r['success'] for r in unfused_jobs) if unfused_jobs else False
        unfused_cycles = sum(r['cycles'] or 0 for r in unfused_jobs) if unfused_success else None
        dataset_results['unfused'] = {
            'success': unfused_success,
            'cycles': unfused_cycles,
            'breakdown': unfused_breakdown
        }

        results[dataset] = dataset_results

    return results


def run_graphsage_benchmarks(sparsity, build_dir, parfactor, timeout, datasets=None, output_dir=None, enable_hbm=True, workers=DEFAULT_WORKERS, cleanup_data=True):
    """Run GraphSAGE benchmarks for all 5 datasets with parallel execution."""
    models_dir = ARTIFACT_ROOT / 'samml' / 'tests' / 'models'

    if datasets is None:
        datasets = list(GRAPHSAGE_CONFIG.keys())

    # Build all jobs
    jobs = []
    job_id = 0

    for dataset in datasets:
        if dataset not in GRAPHSAGE_CONFIG:
            print(f"WARNING: Unknown GraphSAGE dataset {dataset}, skipping")
            continue

        config = GRAPHSAGE_CONFIG[dataset]
        ds_type = config['dataset_type']
        ds_name = config['dataset_name']

        # 1. Fully Fused
        # Skip fully_fused for ogbl-collab and ogbn-mag (too large for medium mode)
        skip_fully_fused = dataset in ['collab', 'mag']

        fused_mlir = models_dir / config['fused_mlir']
        if fused_mlir.exists() and not skip_fully_fused:
            jobs.append(BenchmarkJob(
                job_id=job_id, mlir_file=fused_mlir, sparsity=sparsity,
                build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                dataset_type=ds_type, dataset_name=ds_name,
                use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                model_name='graphsage', dataset_key=dataset, fusion_type='fully_fused',
                op_name=config['fused_mlir'], cleanup_data=cleanup_data
            ))
            job_id += 1

        # 2. Partially Fused
        for op in config.get('partial_ops', []):
            mlir_file = models_dir / config['unfused_dir'] / op
            if mlir_file.exists():
                jobs.append(BenchmarkJob(
                    job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
                    build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                    dataset_type=ds_type, dataset_name=ds_name,
                    use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                    model_name='graphsage', dataset_key=dataset, fusion_type='partially_fused',
                    op_name=op, cleanup_data=cleanup_data
                ))
                job_id += 1

        # 3. Unfused
        for op in config['unfused_ops']:
            mlir_file = models_dir / config['unfused_dir'] / op
            if mlir_file.exists():
                jobs.append(BenchmarkJob(
                    job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
                    build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                    dataset_type=ds_type, dataset_name=ds_name,
                    use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                    model_name='graphsage', dataset_key=dataset, fusion_type='unfused',
                    op_name=op, cleanup_data=cleanup_data
                ))
                job_id += 1

    # Run all jobs in parallel
    print(f"\n{'='*80}")
    print(f"GRAPHSAGE BENCHMARKS: Running {len(jobs)} jobs with {workers} parallel workers")
    print(f"{'='*80}")

    job_results = run_jobs_parallel(jobs, max_workers=workers)

    # Organize results by dataset
    results = {}
    for dataset in datasets:
        if dataset not in GRAPHSAGE_CONFIG:
            continue
        config = GRAPHSAGE_CONFIG[dataset]
        dataset_results = {}

        # Filter results for this dataset
        dataset_jobs = [r for r in job_results if r.get('dataset_key') == dataset]

        # Fully fused
        fused_jobs = [r for r in dataset_jobs if r.get('fusion_type') == 'fully_fused']
        if fused_jobs:
            r = fused_jobs[0]
            dataset_results['fully_fused'] = {'success': r['success'], 'cycles': r['cycles']}
        else:
            dataset_results['fully_fused'] = {'success': False, 'cycles': None, 'error': 'file_not_found'}

        # Partially fused
        partial_jobs = [r for r in dataset_jobs if r.get('fusion_type') == 'partially_fused']
        partial_breakdown = [{'file': r['op_name'], 'success': r['success'], 'cycles': r['cycles']} for r in partial_jobs]
        partial_success = all(r['success'] for r in partial_jobs) if partial_jobs else False
        partial_cycles = sum(r['cycles'] or 0 for r in partial_jobs) if partial_success else None
        dataset_results['partially_fused'] = {
            'success': partial_success,
            'cycles': partial_cycles,
            'breakdown': partial_breakdown
        }

        # Unfused
        unfused_jobs = [r for r in dataset_jobs if r.get('fusion_type') == 'unfused']
        unfused_breakdown = [{'file': r['op_name'], 'success': r['success'], 'cycles': r['cycles']} for r in unfused_jobs]
        unfused_success = all(r['success'] for r in unfused_jobs) if unfused_jobs else False
        unfused_cycles = sum(r['cycles'] or 0 for r in unfused_jobs) if unfused_success else None
        dataset_results['unfused'] = {
            'success': unfused_success,
            'cycles': unfused_cycles,
            'breakdown': unfused_breakdown
        }

        results[dataset] = dataset_results

    return results


def run_gpt3_benchmarks(sparsity, build_dir, parfactor, timeout, block_sizes=None, output_dir=None, enable_hbm=True, workers=DEFAULT_WORKERS, cleanup_data=True):
    """Run GPT-3/BigBird benchmarks for all block sizes with parallel execution.

    OPTIMIZATION: Block-size-independent components are run only ONCE:
      - Layernorm + QKV projections (fused component 1)
      - OutLinear + Residual + LN + FFN + Residual + LN + QKV (fused component 3)
      - OutLinear + Residual + LN + FFN + Residual (partial component 3)
      - All non-MHA unfused ops (13 ops)

    Block-size-dependent components are run for EACH block size:
      - Fused Attention (multihead_attention.mlir with --useGen)
      - Unfused MHA ops: mhaQK_mul, mhaQK_mask, mhaQK_scale, mhaQK_softmax, mhaQK_mul2

    Fully Fused = 3 components:
      1. Layernorm + QKV projections (no --useGen) - run once
      2. Fused Attention (with --useGen and block size) - run per block size
      3. OutLinear + Residual + LN + FFN + Residual + LN + QKV (no --useGen) - run once

    Unfused = 18 individual operations (5 MHA ops with --useGen, 13 non-MHA ops)
    """
    models_dir = ARTIFACT_ROOT / 'samml' / 'tests' / 'models'

    if block_sizes is None:
        block_sizes = list(GPT3_CONFIG.keys())

    # Use first config as reference (all have same non-MHA ops)
    ref_config = GPT3_CONFIG[block_sizes[0]]

    # =========================================================================
    # STEP 1: Build and run all shared (block-size-independent) jobs in parallel
    # =========================================================================
    print(f"\n{'#'*80}")
    print(f"# GPT-3: Running block-size-independent components (run once)")
    print(f"{'#'*80}")

    shared_jobs = []
    job_id = 0

    # Fused component 1: Layernorm + QKV projections
    comp1_op = ref_config['fused_ops'][0]
    mlir_file = models_dir / comp1_op['file']
    if mlir_file.exists():
        shared_jobs.append(BenchmarkJob(
            job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
            build_dir=build_dir, parfactor=parfactor, timeout=timeout,
            use_gen=comp1_op['use_gen'], enable_hbm=enable_hbm, output_dir=output_dir,
            model_name='gpt3', dataset_key='shared', fusion_type='fused_comp1',
            op_name=comp1_op['file'], cleanup_data=cleanup_data
        ))
        job_id += 1

    # Fused component 3 full: OutLinear + ... + LN + QKV
    comp3_full_op = ref_config['fused_ops'][2]
    mlir_file = models_dir / comp3_full_op['file']
    if mlir_file.exists():
        shared_jobs.append(BenchmarkJob(
            job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
            build_dir=build_dir, parfactor=parfactor, timeout=timeout,
            use_gen=comp3_full_op['use_gen'], enable_hbm=enable_hbm, output_dir=output_dir,
            model_name='gpt3', dataset_key='shared', fusion_type='fused_comp3_full',
            op_name=comp3_full_op['file'], cleanup_data=cleanup_data
        ))
        job_id += 1

    # Partial component 3: OutLinear + ... + Residual
    comp3_partial_op = ref_config['partial_fused_ops'][2]
    mlir_file = models_dir / comp3_partial_op['file']
    if mlir_file.exists():
        shared_jobs.append(BenchmarkJob(
            job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
            build_dir=build_dir, parfactor=parfactor, timeout=timeout,
            use_gen=comp3_partial_op['use_gen'], enable_hbm=enable_hbm, output_dir=output_dir,
            model_name='gpt3', dataset_key='shared', fusion_type='partial_comp3',
            op_name=comp3_partial_op['file'], cleanup_data=cleanup_data
        ))
        job_id += 1

    # Non-MHA unfused ops (13 ops)
    non_mha_unfused_ops = [op for op in ref_config['unfused_ops'] if not op['use_gen']]
    for op in non_mha_unfused_ops:
        mlir_file = models_dir / op['file']
        if mlir_file.exists():
            shared_jobs.append(BenchmarkJob(
                job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
                build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                use_gen=False, enable_hbm=enable_hbm, output_dir=output_dir,
                model_name='gpt3', dataset_key='shared', fusion_type='unfused_non_mha',
                op_name=op['file'], cleanup_data=cleanup_data
            ))
            job_id += 1

    # Run shared jobs in parallel
    print(f"\nRunning {len(shared_jobs)} shared jobs with {workers} parallel workers")
    shared_job_results = run_jobs_parallel(shared_jobs, max_workers=workers)

    # Extract shared results
    comp1_result = None
    comp3_full_result = None
    comp3_partial_result = None
    non_mha_results = []

    for r in shared_job_results:
        fusion_type = r.get('fusion_type')
        if fusion_type == 'fused_comp1':
            comp1_result = {
                'file': r['op_name'], 'desc': comp1_op['desc'],
                'success': r['success'], 'cycles': r['cycles'], 'use_gen': comp1_op['use_gen']
            }
        elif fusion_type == 'fused_comp3_full':
            comp3_full_result = {
                'file': r['op_name'], 'desc': comp3_full_op['desc'],
                'success': r['success'], 'cycles': r['cycles'], 'use_gen': comp3_full_op['use_gen']
            }
        elif fusion_type == 'partial_comp3':
            comp3_partial_result = {
                'file': r['op_name'], 'desc': comp3_partial_op['desc'],
                'success': r['success'], 'cycles': r['cycles'], 'use_gen': comp3_partial_op['use_gen']
            }
        elif fusion_type == 'unfused_non_mha':
            # Find matching op for description
            op_desc = next((op['desc'] for op in non_mha_unfused_ops if op['file'] == r['op_name']), r['op_name'])
            non_mha_results.append({
                'file': r['op_name'], 'desc': op_desc,
                'success': r['success'], 'cycles': r['cycles'], 'use_gen': False
            })

    non_mha_success = all(r['success'] for r in non_mha_results) if non_mha_results else False
    non_mha_cycles = sum(r['cycles'] or 0 for r in non_mha_results) if non_mha_success else 0

    print(f"\n  Non-MHA unfused total: {non_mha_cycles:,} cycles (success={non_mha_success})")

    # Store shared results
    shared_results = {
        'comp1_ln_qkv': comp1_result,
        'comp3_full': comp3_full_result,
        'comp3_partial': comp3_partial_result,
        'non_mha_unfused': {
            'success': non_mha_success,
            'cycles': non_mha_cycles,
            'breakdown': non_mha_results
        }
    }

    # =========================================================================
    # STEP 2: Build and run all block-size-dependent jobs in parallel
    # =========================================================================
    mha_jobs = []
    for block_size in block_sizes:
        if block_size not in GPT3_CONFIG:
            print(f"WARNING: Unknown GPT-3 block size {block_size}, skipping")
            continue

        config = GPT3_CONFIG[block_size]
        bs = config['block_size']

        # Fused Attention (component 2) - block-size dependent
        comp2_op = config['fused_ops'][1]
        mlir_file = models_dir / comp2_op['file']
        if mlir_file.exists():
            mha_jobs.append(BenchmarkJob(
                job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
                build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                block_size=bs, use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                model_name='gpt3', dataset_key=f'block_{block_size}', fusion_type='fused_attention',
                op_name=comp2_op['file'], cleanup_data=cleanup_data
            ))
            job_id += 1

        # MHA unfused ops (5 ops that depend on block size)
        mha_unfused_ops = [op for op in config['unfused_ops'] if op['use_gen']]
        for op in mha_unfused_ops:
            mlir_file = models_dir / op['file']
            if mlir_file.exists():
                mha_jobs.append(BenchmarkJob(
                    job_id=job_id, mlir_file=mlir_file, sparsity=sparsity,
                    build_dir=build_dir, parfactor=parfactor, timeout=timeout,
                    block_size=bs, use_gen=True, enable_hbm=enable_hbm, output_dir=output_dir,
                    model_name='gpt3', dataset_key=f'block_{block_size}', fusion_type='unfused_mha',
                    op_name=op['file'], cleanup_data=cleanup_data
                ))
                job_id += 1

    # Run MHA jobs in parallel
    print(f"\n{'='*80}")
    print(f"GPT-3: Running {len(mha_jobs)} MHA jobs with {workers} parallel workers")
    print(f"{'='*80}")

    mha_job_results = run_jobs_parallel(mha_jobs, max_workers=workers)

    # =========================================================================
    # STEP 3: Organize results by block size
    # =========================================================================
    results = {}
    for block_size in block_sizes:
        if block_size not in GPT3_CONFIG:
            continue

        config = GPT3_CONFIG[block_size]
        bs = config['block_size']
        dataset_key = f'block_{block_size}'

        # Filter MHA results for this block size
        block_mha_results = [r for r in mha_job_results if r.get('dataset_key') == dataset_key]

        # Get fused attention result
        comp2_result = None
        for r in block_mha_results:
            if r.get('fusion_type') == 'fused_attention':
                comp2_result = {
                    'file': r['op_name'], 'desc': config['fused_ops'][1]['desc'],
                    'success': r['success'], 'cycles': r['cycles'], 'use_gen': True
                }
                break

        # Get MHA unfused results
        mha_results = []
        mha_unfused_ops = [op for op in config['unfused_ops'] if op['use_gen']]
        for r in block_mha_results:
            if r.get('fusion_type') == 'unfused_mha':
                op_desc = next((op['desc'] for op in mha_unfused_ops if op['file'] == r['op_name']), r['op_name'])
                mha_results.append({
                    'file': r['op_name'], 'desc': op_desc,
                    'success': r['success'], 'cycles': r['cycles'], 'use_gen': True
                })

        mha_success = all(r['success'] for r in mha_results) if mha_results else False
        mha_cycles = sum(r['cycles'] or 0 for r in mha_results) if mha_success else 0

        dataset_results = {}

        # Fully Fused breakdown
        fused_breakdown = [comp1_result, comp2_result, comp3_full_result]
        fused_success = all(r and r.get('success') for r in fused_breakdown)
        fused_cycles = sum(r.get('cycles', 0) or 0 for r in fused_breakdown) if fused_success else None

        dataset_results['fully_fused'] = {
            'success': fused_success,
            'cycles': fused_cycles,
            'breakdown': fused_breakdown
        }

        # Partially Fused breakdown
        partial_breakdown = [comp1_result, comp2_result, comp3_partial_result]
        partial_success = all(r and r.get('success') for r in partial_breakdown)
        partial_cycles = sum(r.get('cycles', 0) or 0 for r in partial_breakdown) if partial_success else None

        dataset_results['partially_fused'] = {
            'success': partial_success,
            'cycles': partial_cycles,
            'breakdown': partial_breakdown
        }

        # Unfused breakdown (combine non-MHA shared + MHA block-specific)
        unfused_breakdown = non_mha_results + mha_results
        unfused_success = non_mha_success and mha_success
        unfused_cycles = (non_mha_cycles + mha_cycles) if unfused_success else None

        dataset_results['unfused'] = {
            'success': unfused_success,
            'cycles': unfused_cycles,
            'breakdown': unfused_breakdown
        }

        # Calculate full 12-decoder model latency
        num_decoders = 12
        full_model_cycles = {}

        # Unfused: 18 ops × 12 decoders
        if unfused_success and unfused_cycles:
            full_model_cycles['unfused'] = unfused_cycles * num_decoders

        # Partially fused: 3 components × 12 decoders
        if partial_success and partial_cycles:
            full_model_cycles['partially_fused'] = partial_cycles * num_decoders

        # Fully fused: More complex calculation
        comp1_cycles = comp1_result.get('cycles') if comp1_result else None
        comp2_cycles = comp2_result.get('cycles') if comp2_result else None
        comp3_full_cycles = comp3_full_result.get('cycles') if comp3_full_result else None
        comp3_partial_cycles = comp3_partial_result.get('cycles') if comp3_partial_result else None

        if all([comp1_cycles, comp2_cycles, comp3_full_cycles, comp3_partial_cycles]):
            full_model_cycles['fully_fused'] = (
                1 * comp1_cycles +
                num_decoders * comp2_cycles +
                (num_decoders - 1) * comp3_full_cycles +
                1 * comp3_partial_cycles
            )
            full_model_cycles['fully_fused_breakdown'] = {
                'comp1_ln_qkv': comp1_cycles,
                'comp2_attention': comp2_cycles,
                'comp3_full': comp3_full_cycles,
                'comp3_partial': comp3_partial_cycles,
            }

        dataset_results['full_model_12_decoders'] = full_model_cycles
        dataset_results['shared_components'] = shared_results

        # Print full model summary
        print(f"\n{'='*60}")
        print(f"FULL MODEL LATENCY (12 decoders) - Block Size {bs}")
        print(f"{'='*60}")
        for config_name, cycles in full_model_cycles.items():
            if config_name != 'fully_fused_breakdown' and cycles:
                print(f"  {config_name:<20}: {cycles:>15,} cycles")

        if 'unfused' in full_model_cycles and 'fully_fused' in full_model_cycles:
            speedup = full_model_cycles['unfused'] / full_model_cycles['fully_fused']
            print(f"\n  Speedup (Unfused -> Fully Fused): {speedup:.2f}x")
        if 'partially_fused' in full_model_cycles and 'fully_fused' in full_model_cycles:
            speedup = full_model_cycles['partially_fused'] / full_model_cycles['fully_fused']
            print(f"  Speedup (Partial -> Fully Fused): {speedup:.2f}x")

        results[block_size] = dataset_results

    return results


def print_summary(all_results):
    """Print a summary of all benchmark results."""
    print("\n" + "="*100)
    print("FIGURE 12 BENCHMARK SUMMARY")
    print("="*100)

    for model_name, model_results in all_results.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*80}")

        # Check if this is GPT-3 (has full_model_12_decoders)
        is_gpt3 = model_name == 'gpt3'

        if is_gpt3:
            # Show single-decoder results
            print("\n--- Single Decoder Results ---")
            print(f"{'Block Size':<20} {'Unfused':>15} {'Partial':>15} {'Fused':>15} {'Speedup':>10}")
            print("-"*80)

        else:
            print(f"{'Dataset/Config':<20} {'Unfused':>15} {'Partial':>15} {'Fused':>15} {'Speedup':>10}")
            print("-"*80)

        for dataset, results in model_results.items():
            unfused = results.get('unfused', {}).get('cycles', 'N/A')
            partial = results.get('partially_fused', {}).get('cycles', 'N/A')
            fused = results.get('fully_fused', {}).get('cycles', 'N/A')

            if isinstance(unfused, int) and isinstance(fused, int) and fused > 0:
                speedup = f"{unfused/fused:.2f}x"
                unfused_str = f"{unfused:,}"
                fused_str = f"{fused:,}"
            else:
                speedup = "N/A"
                unfused_str = str(unfused) if unfused else "FAIL"
                fused_str = str(fused) if fused else "FAIL"

            partial_str = f"{partial:,}" if isinstance(partial, int) else str(partial) if partial else "FAIL"

            print(f"{dataset:<20} {unfused_str:>15} {partial_str:>15} {fused_str:>15} {speedup:>10}")

        # For GPT-3, also show full 12-decoder model results
        if is_gpt3:
            print("\n--- Full Model (12 Decoders) Results ---")
            print(f"{'Block Size':<20} {'Unfused':>18} {'Partial':>18} {'Fused':>18} {'Speedup':>10}")
            print("-"*90)

            for dataset, results in model_results.items():
                full_model = results.get('full_model_12_decoders', {})
                unfused = full_model.get('unfused', 'N/A')
                partial = full_model.get('partially_fused', 'N/A')
                fused = full_model.get('fully_fused', 'N/A')

                if isinstance(unfused, int) and isinstance(fused, int) and fused > 0:
                    speedup = f"{unfused/fused:.2f}x"
                    unfused_str = f"{unfused:,}"
                    fused_str = f"{fused:,}"
                else:
                    speedup = "N/A"
                    unfused_str = str(unfused) if unfused else "FAIL"
                    fused_str = str(fused) if fused else "FAIL"

                partial_str = f"{partial:,}" if isinstance(partial, int) else str(partial) if partial else "FAIL"

                print(f"Block {dataset:<15} {unfused_str:>18} {partial_str:>18} {fused_str:>18} {speedup:>10}")


def main():
    parser = argparse.ArgumentParser(description='Figure 12 Comprehensive Benchmark Runner')
    parser.add_argument('-sp', '--sparsity', type=float, default=0.9,
                        help='Weight sparsity (default: 0.9)')
    parser.add_argument('-b', '--build', type=str, default=None,
                        help='Path to build directory')
    parser.add_argument('-par', '--parfactor', type=int, default=1,
                        help='Parallel factor')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help=f'Timeout per benchmark in seconds (default: {DEFAULT_TIMEOUT} = 1 day)')
    parser.add_argument('--model', type=str, choices=['sae', 'gcn', 'graphsage', 'gpt3', 'all'],
                        default='all', help='Which model(s) to benchmark')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save simulator stdout outputs (for Figure 14)')

    # Model-specific filters
    parser.add_argument('--sae-datasets', type=str, nargs='+',
                        choices=['imagenet', 'nih', 'luna16'],
                        help='SAE datasets to run')
    parser.add_argument('--gcn-datasets', type=str, nargs='+',
                        choices=['cora', 'cora_ml', 'dblp', 'collab', 'mag'],
                        help='GCN datasets to run')
    parser.add_argument('--graphsage-datasets', type=str, nargs='+',
                        choices=['cora', 'cora_ml', 'dblp', 'collab', 'mag'],
                        help='GraphSAGE datasets to run')
    parser.add_argument('--gpt3-blocks', type=str, nargs='+',
                        choices=['16', '32', '64'],
                        help='GPT-3/BigBird block sizes to run')

    # Preset modes for convenience
    parser.add_argument('--mode', type=str, choices=['fast', 'medium', 'complete'],
                        help='Preset benchmark mode:\n'
                             '  fast: All SAE, GCN/GraphSAGE with cora/cora_ml/dblp, all GPT-3 blocks\n'
                             '  medium: All datasets, skips fully_fused for large graphs (GCN mag, GraphSAGE collab/mag)\n'
                             '  complete: All datasets with all fusion types')

    # HBM simulation toggle
    parser.add_argument('--hbm', action='store_true', default=True,
                        help='Enable HBM memory simulation (default: enabled)')
    parser.add_argument('--no-hbm', action='store_false', dest='hbm',
                        help='Disable HBM memory simulation (faster but less accurate)')

    # Parallel execution
    parser.add_argument('-w', '--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Number of parallel benchmark workers (default: {DEFAULT_WORKERS})')

    # Data cleanup
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Disable automatic cleanup of data directories after benchmark completion (saves disk space by default)')

    args = parser.parse_args()

    # Apply preset mode configurations
    if args.mode:
        # Define preset configurations
        PRESET_CONFIGS = {
            'fast': {
                'sae_datasets': ['imagenet', 'nih', 'luna16'],
                'gcn_datasets': ['cora', 'cora_ml', 'dblp'],
                'graphsage_datasets': ['cora', 'cora_ml', 'dblp'],
                'gpt3_blocks': ['16', '32', '64'],
            },
            'medium': {
                'sae_datasets': ['imagenet', 'nih', 'luna16'],
                'gcn_datasets': ['cora', 'cora_ml', 'dblp', 'collab', 'mag'],
                'graphsage_datasets': ['cora', 'cora_ml', 'dblp', 'collab', 'mag'],
                'gpt3_blocks': ['16', '32', '64'],
            },
            'complete': {
                'sae_datasets': ['imagenet', 'nih', 'luna16'],
                'gcn_datasets': ['cora', 'cora_ml', 'dblp', 'collab', 'mag'],
                'graphsage_datasets': ['cora', 'cora_ml', 'dblp', 'collab', 'mag'],
                'gpt3_blocks': ['16', '32', '64'],
            },
        }
        preset = PRESET_CONFIGS[args.mode]
        # Only apply preset values if user didn't explicitly specify them
        if args.sae_datasets is None:
            args.sae_datasets = preset['sae_datasets']
        if args.gcn_datasets is None:
            args.gcn_datasets = preset['gcn_datasets']
        if args.graphsage_datasets is None:
            args.graphsage_datasets = preset['graphsage_datasets']
        if args.gpt3_blocks is None:
            args.gpt3_blocks = preset['gpt3_blocks']
        print(f"Using preset mode: {args.mode}")

    build_dir = Path(args.build) if args.build else PROJECT_ROOT / 'build'

    # Set up output directory for simulator outputs
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Simulator outputs will be saved to: {output_dir}")

    # Determine sparsity per model (use command-line override if provided, else defaults)
    # If user provides --sparsity, use it for all models; otherwise use per-model defaults
    user_sparsity = args.sparsity
    sparsity_override = user_sparsity != 0.9  # Check if user changed from default

    print("="*100)
    print("FIGURE 12 ARTIFACT EVALUATION - FuseFlow Paper")
    print("="*100)
    if args.mode:
        print(f"Mode: {args.mode}")
    if sparsity_override:
        print(f"Sparsity (override): {user_sparsity * 100:.0f}% for all models")
    else:
        print(f"Sparsity: SAE={DEFAULT_SPARSITY['sae']*100:.0f}%, GCN={DEFAULT_SPARSITY['gcn']*100:.0f}%, GraphSAGE={DEFAULT_SPARSITY['graphsage']*100:.0f}%, GPT-3={DEFAULT_SPARSITY['gpt3']*100:.0f}%")
    print(f"Parallel Factor: {args.parfactor}")
    print(f"Timeout: {args.timeout} seconds ({args.timeout/3600:.1f} hours)")
    print(f"Build Dir: {build_dir}")
    print(f"Model(s): {args.model}")
    if args.sae_datasets:
        print(f"SAE datasets: {', '.join(args.sae_datasets)}")
    if args.gcn_datasets:
        print(f"GCN datasets: {', '.join(args.gcn_datasets)}")
    if args.graphsage_datasets:
        print(f"GraphSAGE datasets: {', '.join(args.graphsage_datasets)}")
    if args.gpt3_blocks:
        print(f"GPT-3 blocks: {', '.join(args.gpt3_blocks)}")
    print(f"Parallel workers: {args.workers}")
    print("="*100)

    all_results = {}
    start_time = time.time()

    # Run benchmarks with appropriate sparsity per model
    if args.model in ['sae', 'all']:
        sae_sparsity = user_sparsity if sparsity_override else DEFAULT_SPARSITY['sae']
        print("\n" + "#"*100)
        print(f"# SPARSE AUTOENCODER (SAE) BENCHMARKS - {sae_sparsity*100:.0f}% sparsity")
        print("#"*100)
        all_results['sae'] = run_sae_benchmarks(
            sae_sparsity, build_dir, args.parfactor, args.timeout,
            datasets=args.sae_datasets, output_dir=output_dir, enable_hbm=args.hbm,
            workers=args.workers, cleanup_data=not args.no_cleanup
        )

    if args.model in ['gcn', 'all']:
        gcn_sparsity = user_sparsity if sparsity_override else DEFAULT_SPARSITY['gcn']
        print("\n" + "#"*100)
        print(f"# GCN BENCHMARKS - {gcn_sparsity*100:.0f}% sparsity")
        print("#"*100)
        all_results['gcn'] = run_gcn_benchmarks(
            gcn_sparsity, build_dir, args.parfactor, args.timeout,
            datasets=args.gcn_datasets, output_dir=output_dir, enable_hbm=args.hbm,
            workers=args.workers, cleanup_data=not args.no_cleanup
        )

    if args.model in ['graphsage', 'all']:
        graphsage_sparsity = user_sparsity if sparsity_override else DEFAULT_SPARSITY['graphsage']
        print("\n" + "#"*100)
        print(f"# GRAPHSAGE BENCHMARKS - {graphsage_sparsity*100:.0f}% sparsity")
        print("#"*100)
        all_results['graphsage'] = run_graphsage_benchmarks(
            graphsage_sparsity, build_dir, args.parfactor, args.timeout,
            datasets=args.graphsage_datasets, output_dir=output_dir, enable_hbm=args.hbm,
            workers=args.workers, cleanup_data=not args.no_cleanup
        )

    if args.model in ['gpt3', 'all']:
        gpt3_sparsity = user_sparsity if sparsity_override else DEFAULT_SPARSITY['gpt3']
        print("\n" + "#"*100)
        print(f"# GPT-3/BIGBIRD BENCHMARKS - {gpt3_sparsity*100:.0f}% sparsity")
        print("#"*100)
        all_results['gpt3'] = run_gpt3_benchmarks(
            gpt3_sparsity, build_dir, args.parfactor, args.timeout,
            block_sizes=args.gpt3_blocks, output_dir=output_dir, enable_hbm=args.hbm,
            workers=args.workers, cleanup_data=not args.no_cleanup
        )

    elapsed_time = time.time() - start_time

    # Print summary
    print_summary(all_results)

    print(f"\n{'='*100}")
    print(f"Total elapsed time: {elapsed_time/3600:.2f} hours")
    print(f"{'='*100}")

    # Save results to JSON
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = ARTIFACT_ROOT / f'figure12_results_{timestamp}.json'

    results_with_metadata = {
        'metadata': {
            'sparsity': args.sparsity,
            'parfactor': args.parfactor,
            'timeout': args.timeout,
            'workers': args.workers,
            'hbm_enabled': args.hbm,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'output_dir': str(output_dir) if output_dir else None,
        },
        'results': all_results
    }

    with open(output_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    if output_dir:
        print(f"Simulator outputs saved to: {output_dir}")
        print(f"  Directory structure: {output_dir}/<model>/<dataset>/<fusion_type>/<op_name>.txt")

    return all_results


if __name__ == '__main__':
    main()
