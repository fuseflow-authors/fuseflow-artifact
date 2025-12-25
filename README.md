# fuseflow-artifact
Repo for FuseFlow artifact generation

## Overview
- Getting Started (5 human-minutes + 30 compute-minutes)
- **Data Persistence Setup** - **CRITICAL: Read this to avoid data loss**
- **Quick Start: Run All Benchmarks** (5 human-minutes + 96 compute-hours) - **Recommended for artifact reviewers**
- Run Experiments:
    - Run Top-Level Script (5 human-minutes + 96 compute-hours)
    - Run Figure 12: Performance Comparison (5 human-minutes + up to 96 compute-hours)
    - Run Figure 14: GCN FLOPs and Memory Analysis (5 human-minutes + 5 compute-minutes)
    - Run Figure 15a: Parallelism Factor Sweep (5 human-minutes + 1 compute-hours)
    - Run Figure 15b: Parallelism Location Sweep (5 human-minutes + 10 compute-minutes)
    - Run Figure 16: Block Size Comparison (5 human-minutes + 30 compute-minutes)
    - Run Figure 17: Dataflow Order Sweep (5 human-minutes + 5 compute-minutes)
- Validate All Results
----
- [Optional] How to Reuse Artifact Beyond the Paper (10+ human-minutes)
- [Optional] Detailed Explanation of What the Top-Level Script Does

## Getting Started
This guide assumes the user has a working installation of Docker and some version of Python 3 installed.

- Run the following commands to build the docker image named `fuseflow-artifact` locally from the files in this GitHub repo.
  ```
  git submodule update --init --recursive
  docker build -t fuseflow-artifact .
  ```
  *NOTE:* Building the Docker image requires ~16GB RAM during LLVM compilation, ~50GB disk space, and takes 10-20 minutes depending on CPU.

## Data Persistence Setup

**CRITICAL:** The default Docker workflow can cause data loss on system crashes or container restarts. Follow these steps to ensure your results are preserved:

### Step 1: Create persistent storage directories on your host machine
```bash
mkdir -p ~/fuseflow_data/results
mkdir -p ~/fuseflow_data/logs
mkdir -p ~/fuseflow_data/checkpoints
```

### Step 2: Run container with volume mounts (NO --rm flag!)
```bash
docker run -d -it \
  -v ~/fuseflow_data/results:/fuseflow-artifact/results \
  -v ~/fuseflow_data/logs:/fuseflow-artifact/logs \
  -v ~/fuseflow_data/checkpoints:/fuseflow-artifact/checkpoints \
  --name fuseflow-container \
  fuseflow-artifact bash
```

**Key changes from old approach:**
- ✅ **Removed `--rm` flag** - Container persists after stopping
- ✅ **Added volume mounts** - Data stored on host filesystem
- ✅ **Named container** - Easy to restart with `docker start fuseflow-container`

### Step 3: Attach to the container
```bash
docker attach fuseflow-container
```

### Step 4: Exit safely without killing container
- **IMPORTANT:** Do not type `exit` - this will kill the container
- **Use `CTRL-p, CTRL-q`** to detach safely
- Container keeps running in background

### Step 5: Reattach after system crash or logout
```bash
docker start fuseflow-container    # Start if stopped
docker attach fuseflow-container   # Reattach to running container
```

### What happens now when your system crashes?
- ✅ All results, logs, and checkpoints are safe in `~/fuseflow_data/`
- ✅ Container can be restarted with `docker start fuseflow-container`
- ✅ No data loss from ephemeral container storage

## Quick Start: Run All Benchmarks (5 human-minutes + 96 compute-hours)
For artifact reviewers who want to run all experiments with a single command:

- Within the Docker container, run:
  ```
  ./scripts/run_all_benchmarks.sh 2>&1 | tee logs/run_all_benchmarks.log
  ```
  This script will:
  - Run all benchmark experiments (Figures 12, 14, 15a, 15b, 16, 17)
  - Generate all PDF plots automatically
  - Save results to the `results/` directory (persisted to host via volume mount)
  - Log all output to `logs/run_all_benchmarks.log`

- **Monitor progress** (from host machine):
  ```bash
  tail -f ~/fuseflow_data/logs/run_all_benchmarks.log
  ```

- **Check results while running** (from host machine):
  ```bash
  ls -lh ~/fuseflow_data/results/
  ```

- Once complete, exit the Docker container (`CTRL-p, CTRL-q`) and view results:
  ```bash
  ls ~/fuseflow_data/results/
  ```
  All PDFs and JSON files are now on your host filesystem.

## Run Top-Level Script (5 human-minutes + 96 compute-hours)
We provide scripts to generate all of the results within the container.

**IMPORTANT:** Always redirect output to log files to preserve results:

- Within the Docker container, run the following commands to generate all results:
  ```
  # Figure 12 - Main performance comparison (SAE, GCN, GraphSAGE, GPT-3)
  python3 scripts/run_figure12_benchmarks.py --mode complete 2>&1 | tee logs/figure12.log

  # Figure 14 - GCN FLOPs and memory metrics
  python3 scripts/process_figure14_metrics.py 2>&1 | tee logs/figure14.log

  # Figure 15a - Sparsity sweep
  python3 scripts/run_figure15a_sweep.py 2>&1 | tee logs/figure15a.log

  # Figure 15b - Parallelism sweep
  python3 scripts/run_figure15b_sweep.py 2>&1 | tee logs/figure15b.log

  # Figure 16 - Block size comparison
  python3 scripts/run_figure16.py 2>&1 | tee logs/figure16.log

  # Figure 17 - Dataflow order sweep
  python3 scripts/run_figure17_sweep.py 2>&1 | tee logs/figure17.log
  ```

- **Monitor from host machine:**
  ```bash
  tail -f ~/fuseflow_data/logs/figure12.log
  ```

- Once this completes, results are automatically saved to `~/fuseflow_data/results/` on your host machine.

## Run Figure 12: Performance Comparison (5 human-minutes + 96 compute-hours)
Figure 12 compares performance across four model architectures (SAE, GCN, GraphSAGE, GPT-3).

### Artifact Configuration for Figure 12

**Default Configuration (Medium Mode, No HBM):**
By default, the artifact runs Figure 12 in `medium` mode with HBM simulation disabled (`--no-hbm`) to validate the fusion performance trends shown in the paper while keeping evaluation time practical (~96 compute-hours). This configuration:

- Runs all datasets for all models
- Skips the `fully_fused` configuration only for the largest datasets (GCN MAG, GraphSAGE Collab/MAG), since `fully_fused` is shown to be inefficient overall for GCN and GraphSAGE across all datasets
- Disables HBM memory simulation to reduce total runtime from weeks to days

**Parallel Execution:**
The Figure 12 benchmark script supports parallel execution of simulation jobs using the `--workers` flag. By default, the artifact runs with 4 parallel workers (requires peak ~140GB memory). Users can adjust the number of workers to balance speed vs. memory usage:

To modify the worker count, edit the `--workers`/`-w` parameter in [scripts/run_all_benchmarks.sh](scripts/run_all_benchmarks.sh#L26) or specify it when running manually:
```bash
python3 scripts/run_figure12_benchmarks.py --mode medium --workers 4 --no-hbm 2>&1 | tee logs/figure12.log
```

**HBM Simulation:**
HBM simulation is disabled by default to keep artifact evaluation time reasonable. Enabling HBM increases total runtime to over a week due to detailed memory simulation overhead. Disabling HBM affects **absolute latency values** but preserves **qualitative fusion trends** (relative ordering of unfused, partially fused, and fully fused configurations), since these trends are primarily driven by fusion-induced changes in intermediate materialization, recomputation, and coordinate processing rather than peak off-chip bandwidth.

Reviewers who wish to reproduce the paper's HBM-backed results may enable it by removing the `--no-hbm` flag in [scripts/run_all_benchmarks.sh](scripts/run_all_benchmarks.sh#L26).

**Benchmark Modes:**
The Figure 12 sweep script supports three modes with different dataset coverage:

| Mode | SAE Datasets | GCN Datasets | GraphSAGE Datasets | GPT-3 Block Sizes | Notes |
|------|--------------|--------------|--------------------|--------------------|-------|
| `fast` | All 3 (ImageNet, NIH-CXR, LUNA16) | 3 smaller (Cora, Cora-ML, DBLP) | 3 smaller (Cora, Cora-ML, DBLP) | All 3 (16, 32, 64) | Quick testing (~1 compute-hour) |
| `medium` | All 3 (ImageNet, NIH-CXR, LUNA16) | All 5 (Cora, Cora-ML, DBLP, Collab, MAG) | All 5 (Cora, Cora-ML, DBLP, Collab, MAG) | All 3 (16, 32, 64) | **Default for artifact** (~96 compute-hours). Skips `fully_fused` for MAG (GCN) and Collab/MAG (GraphSAGE) |
| `complete` | All 3 (ImageNet, NIH-CXR, LUNA16) | All 5 with all fusion types | All 5 with all fusion types | All 3 (16, 32, 64) | Full paper results (>1 week). Runs `fully_fused` on large datasets |

The `medium` mode skips the `fully_fused` configuration for the largest GCN and GraphSAGE datasets (which are shown to be inefficient overall) to avoid long-running experiments that would add over a week of simulation time. We only run the experiments that complete in a reasonable time while still validating the key fusion performance trends.

**Models evaluated:**
| Model | Datasets/Configs |
|-------|------------------|
| SAE (Sparse Autoencoder) | ImageNet, NIH-CXR, LUNA16 |
| GCN | Cora, Cora-ML, DBLP, Collab, MAG |
| GraphSAGE | Cora, Cora-ML, DBLP, Collab, MAG |
| GPT-3 w/ BigBird | Block sizes: 16, 32, 64 |

Choose one of the following options to run:

1. Run `--mode fast` to run a restricted set of experiments for quick testing (about 1 compute-hour):
   ```
   python3 scripts/run_figure12_benchmarks.py --model gcn --gcn-datasets cora --mode fast 2>&1 | tee logs/figure12_fast.log
   ```

2. Run `--mode complete` to run the full set of benchmarks that will take over a week:
   ```
   python3 scripts/run_figure12_benchmarks.py --mode complete 2>&1 | tee logs/figure12_complete.log
   ```

3. Run specific models or datasets:
   ```
   # Run only GCN on specific datasets
   python3 scripts/run_figure12_benchmarks.py --model gcn --gcn-datasets cora cora_ml dblp 2>&1 | tee logs/figure12_gcn.log

   # Run only SAE benchmarks
   python3 scripts/run_figure12_benchmarks.py --model sae 2>&1 | tee logs/figure12_sae.log

   # Run only GPT-3 BigBird benchmarks
   python3 scripts/run_figure12_benchmarks.py --model gpt3 2>&1 | tee logs/figure12_gpt3.log
   ```

- The script generates a `figure12_results.json` file with cycle counts for each configuration (automatically saved to `results/`).

- Once all desired benchmarks are run, generate Figure 12 as a PDF:
  ```
  python3 scripts/plot_figure12.py --input figure12_results.json --output results/figure12.pdf
  ```

## Run Figure 14: GCN FLOPs and Memory Analysis (5 human-minutes + 5 compute-minutes)
Figure 14 analyzes computational efficiency and memory access patterns for GCN.

- Run the following commands:
  ```
  python3 scripts/process_figure14_metrics.py 2>&1 | tee logs/figure14.log
  ```

- Generate Figure 14 as a PDF:
  ```
  python3 scripts/plot_figure14.py
  ```
  - The script will create a plot at the location `/fuseflow-artifact/results/figure14.pdf`.

## Run Figure 15a: Sparsity Sweep (5 human-minutes + 1 compute-hour)
Figure 15a shows how performance varies with different sparsity levels.

- Run the sparsity sweep script:
  ```
  python3 scripts/run_figure15a_sweep.py 2>&1 | tee logs/figure15a.log
  ```
  - Results are saved to `figure15a_results.json`

- Generate Figure 15a as a PDF:
  ```
  python3 scripts/plot_figure15a.py
  ```
  - The script will create a plot at the location `/fuseflow-artifact/results/figure15a.pdf`.

## Run Figure 15b: Parallelism Sweep (5 human-minutes + 10 compute-minutes)
Figure 15b shows how performance scales with different parallelization factors.

- Run the parallelism sweep script:
  ```
  python3 scripts/run_figure15b_sweep.py 2>&1 | tee logs/figure15b.log
  ```
  - Results are saved to `figure15b_results.json`

- Generate Figure 15b as a PDF:
  ```
  python3 scripts/plot_figure15b.py
  ```
  - The script will create a plot at the location `/fuseflow-artifact/results/figure15b.pdf`.

## Run Figure 16: Block Size Comparison (5 human-minutes + 30 compute-minutes)
Figure 16 compares performance across different block sizes.

- Run the block size comparison script:
  ```
  python3 scripts/run_figure16.py 2>&1 | tee logs/figure16.log
  ```
  - Results are saved to `figure16_results.json`

- Generate Figure 16 as a PDF:
  ```
  python3 scripts/plot_figure16.py
  ```
  - The script will create a plot at the location `/fuseflow-artifact/results/figure16.pdf`.

## Run Figure 17: Dataflow Order Sweep (5 human-minutes + 5 compute-minutes)
Figure 17 evaluates different dataflow ordering strategies.

- Run the dataflow order sweep script:
  ```
  python3 scripts/run_figure17_sweep.py 2>&1 | tee logs/figure17.log
  ```
  - Results are saved to `figure17_results.json`

- Generate Figure 17 as a PDF:
  ```
  python3 scripts/plot_figure17.py
  ```
  - The script will create a plot at the location `/fuseflow-artifact/results/figure17.pdf`.

## Validate All Results

Since you used volume mounts, all results are already on your host machine at `~/fuseflow_data/results/`. Simply view them directly:

```bash
ls -lh ~/fuseflow_data/results/
```

You can open the PDFs directly from your host machine:
```bash
# On Linux
xdg-open ~/fuseflow_data/results/figure12.pdf

# On macOS
open ~/fuseflow_data/results/figure12.pdf
```

- Validate that the plot in `figure12.pdf` matches Figure 12 in the paper (performance comparison).
- Validate that the plot in `figure14.pdf` matches Figure 14 in the paper (GCN FLOPs and memory analysis).
- Validate that the plot in `figure15a.pdf` matches Figure 15a in the paper (sparsity sweep).
- Validate that the plot in `figure15b.pdf` matches Figure 15b in the paper (parallelism sweep).
- Validate that the plot in `figure16.pdf` matches Figure 16 in the paper (block size comparison).
- Validate that the plot in `figure17.pdf` matches Figure 17 in the paper (dataflow order sweep).

-----

## [Optional] How to Reuse Artifact Beyond the Paper
Please note that all active development beyond this paper is located in
the main repositories and not this artifact repository.

### The SAMML Compiler
The SAMML compiler transforms high-level MLIR (Linalg + SparseTensor) to SAM dataflow programs.
All compiler source files can be found in `/fuseflow-artifact/samml/`.

To compile MLIR to SAM dataflow:
```
cd /fuseflow-artifact

# Compile MLIR to SAMML dialect
./samml/build/tools/sam-opt --linalg-to-sam <input.mlir>

# Emit protobuf binary for simulator
./samml/build/tools/sam-opt --linalg-to-sam <input.mlir> | \
    ./samml/build/tools/sam-translate --emit-proto

# With parallelization
./samml/build/tools/sam-opt --linalg-to-sam \
    "--stream-parallelizer=stream-level=0 par-factor=4" <input.mlir> | \
    ./samml/build/tools/sam-translate --emit-proto

# With vectorization
./samml/build/tools/sam-opt --linalg-to-sam \
    "--stream-vectorizer=stream-shape=16" <input.mlir> | \
    ./samml/build/tools/sam-translate --emit-proto
```

Use `./samml/build/tools/sam-opt --help` for specific instructions on compiler passes.

### The Comal Simulator
The Comal simulator is a cycle-accurate dataflow simulator written in Rust with Python bindings.
Source files can be found in `/fuseflow-artifact/comal/`.

To use the simulator programmatically:
```python
import comal
# See scripts/run_end_to_end.py for usage examples
```

### Running Individual Benchmarks
For manual testing or debugging, use the end-to-end script directly:

```
# GCN example
python3 scripts/run_end_to_end.py \
    --infile samml/tests/models/gcn_unfused/gcn_sparse.mlir \
    --build samml/build \
    --sparsity 0.5 \
    --par 1

# GPT-3 with BigBird example
python3 scripts/run_end_to_end.py \
    --infile samml/tests/models/gpt-3/outLinear_layernorm_FFN_layernorm_QKVprojection.mlir \
    --build samml/build \
    --block 64 \
    --sparsity 0.9 \
    --outformat UNC
```

### Tortilla Visualizer
The Tortilla visualizer provides dataflow graph visualization.
Source files can be found in `/fuseflow-artifact/tortilla-visualizer/`.

## [Optional] Detailed Explanation of What the Top-Level Script Does

### Run and Validate Figure 12: Performance Comparison
The `run_figure12_benchmarks.py` script performs the following:

1. For each model (SAE, GCN, GraphSAGE, GPT-3) and dataset:
   - Compiles the MLIR representation using SAMML
   - Runs benchmark configurations
   - Collects cycle counts from the Comal simulator

2. Saves all results to `figure12_results.json`

3. The `plot_figure12.py` script:
   - Loads results from JSON
   - Generates bar charts showing performance comparison across models

**Expected results:** Performance variations across different models and datasets.

### Run and Validate Figure 14: GCN FLOPs and Memory Analysis
The `process_figure14_metrics.py` script:

1. Analyzes GCN workloads across different fusion configurations
2. Collects FLOPs (floating-point operations) counts
3. Collects memory access patterns and traffic

The `plot_figure14.py` script generates visualizations showing:
- Computational efficiency improvements
- Memory traffic reduction through fusion

**Expected results:** FuseFlow reduces memory traffic by exploiting spatial locality through fusion.

### Run and Validate Figure 15a: Sparsity Sweep
The `run_figure15a_sweep.py` script:

1. Sweeps across different sparsity levels
2. Records performance metrics for each sparsity level
3. Saves results to `figure15a_results.json`

The `plot_figure15a.py` script generates visualizations showing performance vs sparsity.

**Expected results:** Performance trends correlate with sparsity levels.

### Run and Validate Figure 15b: Parallelism Sweep
The `run_figure15b_sweep.py` script:

1. Sweeps parallelization factors (e.g., 1, 2, 4, 8, 16, 32, 64)
2. Records cycle counts for each configuration
3. Saves results to `figure15b_results.json`

The `plot_figure15b.py` script generates:
- Line plot showing cycles vs parallelization factor
- Demonstrates scaling behavior

**Expected results:** Near-linear scaling with parallelization factor up to hardware limits.

### Run and Validate Figure 16: Block Size Comparison
The `run_figure16.py` script:

1. Runs benchmarks with different block sizes
2. Collects performance metrics
3. Saves results to `figure16_results.json`

The `plot_figure16.py` script generates visualizations comparing block size performance.

**Expected results:** Performance varies with block size based on hardware characteristics.

### Run and Validate Figure 17: Dataflow Order Sweep
The `run_figure17_sweep.py` script:

1. Evaluates different dataflow ordering strategies
2. Measures performance impact of each ordering
3. Saves results to `figure17_results.json`

The `plot_figure17.py` script generates visualizations showing impact of dataflow ordering.

**Expected results:** Different dataflow orders exhibit varying performance characteristics.

## Directory Structure

```
fuseflow-artifact/
├── samml/               # SAMML compiler (MLIR-based)
│   ├── external/        # External dependencies (llvm-project, or-tools)
│   ├── lib/             # Compiler library sources
│   ├── tools/           # sam-opt, sam-translate binaries
│   └── tests/           # Test MLIR files and models
├── comal/               # Cycle-accurate dataflow simulator (Rust)
│   ├── src/             # Simulator source code
│   └── external/        # Ramulator2 DRAM wrapper
├── sam/                 # Sparse Abstract Machine library
├── scripts/             # Benchmark and plotting scripts
├── tortilla-visualizer/ # Dataflow graph visualization tool
├── results/             # Output directory (mounted to host)
├── logs/                # Log files (mounted to host)
├── checkpoints/         # Checkpoint files (mounted to host)
├── setup.sh             # Automated setup script
├── Dockerfile           # Docker build configuration
└── README.md            # This file
```

## Hardware Requirements

- **CPU**: x86_64 processor (tested on Intel/AMD)
- **RAM**: 32GB minimum (64GB recommended for LLVM build and full benchmarks)
- **Disk**: 50GB free space (LLVM build is large)
- **OS**: Linux (Ubuntu 20.04+ recommended)

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| LLVM build fails with OOM | Reduce parallelism: use `-j4` or `-j2` instead of `-j$(nproc)` |
| `protoc` version mismatch | Install protoc version 24.0 from GitHub releases |
| Rust compilation errors | Update Rust: `rustup update` |
| `maturin` build fails | Ensure virtual environment is activated |
| Out of memory during benchmarks | Use `--mode fast` or reduce parallelization |
| Missing `clang`/`lld` | Install: `sudo apt install clang lld` |
| Container deleted after crash | DO NOT use `--rm` flag when running container |
| Data lost on reboot | Ensure volume mounts are specified in `docker run` |

### Verifying Installation

```
# Check SAMML compiler
./samml/build/tools/sam-opt --help

# Check Comal Python bindings
python3 -c "import comal; print('Comal OK')"

# Check SAM
python3 -c "import sam; print('SAM OK')"
```

## Expected Results

The artifact should reproduce the following key findings from the paper:

1. **Figure 12**: Performance comparison across SAE, GCN, GraphSAGE, and GPT-3 models
2. **Figure 14**: FuseFlow reduces memory traffic by exploiting spatial locality through fusion
3. **Figure 15a**: Performance trends correlate with sparsity levels
4. **Figure 15b**: Near-linear scaling with parallelization factor up to hardware limits
5. **Figure 16**: Performance varies with block size based on hardware characteristics
6. **Figure 17**: Different dataflow orders exhibit varying performance characteristics

**Note:** Exact cycle counts may vary slightly due to:
- Random initialization of sparse tensors
- Different system configurations
- Floating-point non-determinism

The relative speedups and trends should match the paper's results.
