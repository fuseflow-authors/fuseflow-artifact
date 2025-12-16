#!/bin/bash
# FuseFlow Artifact Setup Script
#
# This script sets up the complete FuseFlow artifact including:
# - SAMML compiler (MLIR-based)
# - Comal cycle-accurate simulator (Rust)
# - SAM library
# - Tortilla visualizer
#
# Prerequisites:
# - CMake 3.20+
# - Ninja build system
# - Clang/Clang++ (for LLVM build)
# - Rust (install via rustup)
# - Python 3.8+
# - protoc (Protocol Buffers compiler) version 24.0
# - graphviz (for visualization)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== FuseFlow Artifact Setup ==="
echo "Working directory: $SCRIPT_DIR"

# ============================================================================
# Step 1: Initialize Git Submodules
# ============================================================================
echo ""
echo "=== Step 1: Initializing git submodules ==="
git submodule update --init --recursive

# ============================================================================
# Step 2: Create and Setup Python Virtual Environment
# ============================================================================
echo ""
echo "=== Step 2: Setting up Python virtual environment ==="
python3 -m venv fuseflow-venv
source fuseflow-venv/bin/activate

pip install --upgrade pip

# Install SAMML requirements (lit for LLVM testing)
pip install -r samml/requirements.txt

# Install SAM requirements
pip install -r sam/requirements.txt

# Install tortilla-visualizer requirements
pip install -r tortilla-visualizer/requirement.txt

# Install additional dependencies for benchmarks
pip install torch torch_geometric maturin networkx tqdm pandas

# Install SAM package in editable mode
pip install -e sam/

# ============================================================================
# Step 3: Build LLVM/MLIR (for SAMML compiler)
# ============================================================================
echo ""
echo "=== Step 3: Building LLVM/MLIR (this will take a while ~30-60 min) ==="
echo "Note: Requires 32GB+ RAM. If build fails, try reducing parallelism."

cd samml/external/llvm-project
mkdir -p build && cd build

# Build MLIR with optimizations for memory usage
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON

# Build with parallelism (reduce -j value if running out of memory)
cmake --build . --target check-mlir -j$(nproc)

cd "$SCRIPT_DIR"

# ============================================================================
# Step 4: Build OR-Tools (for SAMML compiler optimization passes)
# ============================================================================
echo ""
echo "=== Step 4: Building OR-Tools ==="

cd samml/external

# Clone or-tools if not present (it's a submodule)
if [ ! -d "or-tools" ]; then
    git clone https://github.com/google/or-tools.git
fi

cd or-tools
mkdir -p build && cd build

cmake -S .. -B . \
    -DBUILD_DEPS=ON \
    -DUSE_SCIP=OFF \
    -DUSE_HIGHS=OFF \
    -DUSE_COINOR=OFF

cmake --build . --config Release --target all -j$(nproc)

cd "$SCRIPT_DIR"

# ============================================================================
# Step 5: Build SAMML Compiler
# ============================================================================
echo ""
echo "=== Step 5: Building SAMML compiler ==="

cd samml
mkdir -p build && cd build

cmake -G Ninja .. \
    -DLLVM_EXTERNAL_LIT=$(which lit)

ninja

cd "$SCRIPT_DIR"

# ============================================================================
# Step 6: Build Comal (Rust Simulator with Python Bindings)
# ============================================================================
echo ""
echo "=== Step 6: Building Comal simulator ==="

# Ensure Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

cd comal

# Build with maturin for Python bindings
source "$SCRIPT_DIR/fuseflow-venv/bin/activate"
maturin develop --release

cd "$SCRIPT_DIR"

# ============================================================================
# Step 7: Build Tortilla Visualizer (Optional)
# ============================================================================
echo ""
echo "=== Step 7: Setting up Tortilla Visualizer ==="

cd tortilla-visualizer

# Generate protobuf files if needed
if command -v protoc &> /dev/null; then
    echo "Protobuf compiler found, regenerating proto files..."
    # Proto files are already generated, but can be regenerated if needed
fi

cd "$SCRIPT_DIR"

# ============================================================================
# Setup Complete
# ============================================================================
echo ""
echo "=============================================="
echo "=== FuseFlow Artifact Setup Complete! ==="
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source fuseflow-venv/bin/activate"
echo ""
echo "To run a quick test (GCN on Cora):"
echo "  python3 scripts/run_figure12_benchmarks.py --model gcn --gcn-datasets cora --mode fast"
echo ""
echo "To run all Figure 12 benchmarks:"
echo "  python3 scripts/run_figure12_benchmarks.py --mode full"
echo ""
echo "For more information, see README.md"
