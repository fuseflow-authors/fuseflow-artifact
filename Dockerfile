# FuseFlow Artifact Docker Image
#
# Build: docker build -t fuseflow-artifact .
# Run:   docker run -it --rm fuseflow-artifact
#
# For artifact evaluation with result persistence:
#   docker run -it --rm -v $(pwd)/results:/fuseflow-artifact/results fuseflow-artifact

FROM ubuntu:22.04

LABEL maintainer="anonymous"
LABEL description="FuseFlow: A Fusion-centric Compilation Framework for Sparse Deep Learning"

# Set timezone
ENV TZ=UTC

# ============================================================================
# Stage 1: Install system dependencies
# ============================================================================
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    clang \
    lld \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    graphviz \
    git \
    curl \
    wget \
    unzip \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CMake 3.28 (required by SAMML)
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.sh -O /tmp/cmake-install.sh && \
    chmod +x /tmp/cmake-install.sh && \
    /tmp/cmake-install.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake-install.sh

# ============================================================================
# Stage 2: Install Rust
# ============================================================================
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# ============================================================================
# Stage 3: Install Protocol Buffers compiler v24.0
# ============================================================================
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v24.0/protoc-24.0-linux-x86_64.zip \
    && unzip protoc-24.0-linux-x86_64.zip -d /usr/local \
    && rm protoc-24.0-linux-x86_64.zip

# ============================================================================
# Stage 4: Copy LLVM source ONLY (for maximum cache efficiency)
# ============================================================================
# NOTE: LLVM build is the slowest step (~30+ min). By copying only LLVM source
# first, this layer is cached unless LLVM source itself changes.
WORKDIR /fuseflow-artifact
COPY samml/external/llvm-project samml/external/llvm-project

# ============================================================================
# Stage 5: Build LLVM/MLIR (this takes a long time - cache this!)
# ============================================================================
RUN cd samml/external/llvm-project && \
    mkdir -p build && cd build && \
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD="Native" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_USE_SPLIT_DWARF=ON \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_ENABLE_LLD=ON && \
    ninja MLIRMlirOptMain && \
    ninja check-mlir

# ============================================================================
# Stage 6: Copy OR-Tools source and build
# ============================================================================
COPY samml/external/or-tools samml/external/or-tools
RUN cd samml/external/or-tools && \
    mkdir -p build && cd build && \
    cmake -S .. -B . \
        -DBUILD_DEPS=ON \
        -DUSE_SCIP=OFF \
        -DUSE_HIGHS=OFF \
        -DUSE_COINOR=OFF && \
    cmake --build . --config Release --target all -j$(nproc) && \
    cmake --build . --config Release --target install -v

# ============================================================================
# Stage 7: Copy dependency files and install Python dependencies
# ============================================================================
COPY samml/requirements.txt samml/requirements.txt
COPY sam/requirements.txt sam/requirements.txt
COPY tortilla-visualizer/requirement.txt tortilla-visualizer/requirement.txt

RUN python3 -m venv /opt/fuseflow-venv
ENV PATH="/opt/fuseflow-venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/fuseflow-venv"

RUN pip install --upgrade pip && \
    pip install -r samml/requirements.txt && \
    pip install -r sam/requirements.txt && \
    pip install -r tortilla-visualizer/requirement.txt && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install torch_geometric networkx tqdm pandas scipy matplotlib && \
    pip install -U "maturin[patchelf]" && \
    pip install ogb

# ============================================================================
# Stage 7.5: Patch PyTorch libraries for PyTorch 2.6+ compatibility
# ============================================================================
# Fix PyTorch 2.6+ weights_only default breaking change for OGB datasets
COPY scripts/patch_pytorch_libs.sh scripts/patch_pytorch_libs.sh
RUN chmod +x scripts/patch_pytorch_libs.sh && \
    bash scripts/patch_pytorch_libs.sh

# ============================================================================
# Stage 8: Copy remaining source code
# ============================================================================
COPY . .

# Install SAM in editable mode (needs full source)
RUN pip install -e sam/

# ============================================================================
# Stage 9: Build SAMML compiler
# ============================================================================
RUN cd samml && \
    mkdir -p build && cd build && \
    cmake -G Ninja .. -DLLVM_EXTERNAL_LIT=$(which lit) && \
    ninja

# ============================================================================
# Stage 10: Build Comal simulator
# ============================================================================
RUN cd comal && \
    maturin develop --release

# ============================================================================
# Stage 11: Build Tortilla visualizer
# ============================================================================
RUN cd tortilla-visualizer && \
    make

# ============================================================================
# Stage 12: Create results directory and set permissions
# ============================================================================
RUN mkdir -p /fuseflow-artifact/results && \
    chmod -R 755 /fuseflow-artifact

# Set environment variables for runtime
ENV DATA_PATH=/tmp/data
ENV PYTHONPATH="/fuseflow-artifact:${PYTHONPATH}"

# ============================================================================
# Default command: interactive shell
# ============================================================================
WORKDIR /fuseflow-artifact

# Verification script
RUN echo '#!/bin/bash\n\
echo "=== FuseFlow Artifact Environment ==="\n\
echo ""\n\
echo "Verifying installation..."\n\
python3 -c "import comal; print(\"✓ Comal OK\")"\n\
python3 -c "import sam; print(\"✓ SAM OK\")"\n\
./samml/build/tools/sam-opt --version > /dev/null 2>&1 && echo "✓ SAMML compiler OK"\n\
echo ""\n\
echo "Quick start commands:"\n\
echo "  # Run GCN benchmark on Cora (fast mode)"\n\
echo "  python3 scripts/run_figure12_benchmarks.py --model gcn --gcn-datasets cora --mode fast"\n\
echo ""\n\
echo "  # Run all Figure 12 benchmarks"\n\
echo "  python3 scripts/run_figure12_benchmarks.py --mode full"\n\
echo ""\n\
echo "  # Generate plots"\n\
echo "  python3 scripts/plot_figure12.py --input figure12_results.json --output results/figure12.pdf"\n\
echo ""' > /usr/local/bin/fuseflow-info && chmod +x /usr/local/bin/fuseflow-info

CMD ["fuseflow-info && /bin/bash"]
ENTRYPOINT ["/bin/bash", "-c"]
