#!/bin/bash
# FuseFlow Artifact - Extract Results from Docker Container
# Run this script from your host machine (outside Docker) to copy results

set -e

echo "========================================"
echo "FuseFlow Artifact - Extracting Results"
echo "========================================"
echo ""

# Get the container ID
CONTAINER_ID=$(docker ps -q --filter ancestor=fuseflow-artifact)

if [ -z "$CONTAINER_ID" ]; then
    echo "Error: No running fuseflow-artifact container found."
    echo ""
    echo "Please start the container first:"
    echo "  docker run -d -it --rm fuseflow-artifact bash"
    echo "  docker attach <CONTAINER_ID>"
    exit 1
fi

echo "Found container: $CONTAINER_ID"
echo ""

# Create output directory
OUTPUT_DIR="output_figures"
mkdir -p "$OUTPUT_DIR"

echo "Extracting results to: $OUTPUT_DIR/"
echo ""

# Copy PDF figures
echo "Copying PDF figures..."
docker cp "$CONTAINER_ID:/fuseflow-artifact/results/figure12.pdf" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure12.pdf" || echo "  ✗ figure12.pdf (not found)"
docker cp "$CONTAINER_ID:/fuseflow-artifact/results/figure14.pdf" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure14.pdf" || echo "  ✗ figure14.pdf (not found)"
docker cp "$CONTAINER_ID:/fuseflow-artifact/results/figure15a.pdf" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure15a.pdf" || echo "  ✗ figure15a.pdf (not found)"
docker cp "$CONTAINER_ID:/fuseflow-artifact/results/figure15b.pdf" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure15b.pdf" || echo "  ✗ figure15b.pdf (not found)"
docker cp "$CONTAINER_ID:/fuseflow-artifact/results/figure16.pdf" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure16.pdf" || echo "  ✗ figure16.pdf (not found)"
docker cp "$CONTAINER_ID:/fuseflow-artifact/results/figure17.pdf" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure17.pdf" || echo "  ✗ figure17.pdf (not found)"

echo ""
echo "Copying JSON result files..."
docker cp "$CONTAINER_ID:/fuseflow-artifact/figure12_results.json" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure12_results.json" || echo "  ✗ figure12_results.json (not found)"
docker cp "$CONTAINER_ID:/fuseflow-artifact/figure15a_results.json" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure15a_results.json" || echo "  ✗ figure15a_results.json (not found)"
docker cp "$CONTAINER_ID:/fuseflow-artifact/figure15b_results.json" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure15b_results.json" || echo "  ✗ figure15b_results.json (not found)"
docker cp "$CONTAINER_ID:/fuseflow-artifact/figure16_results.json" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure16_results.json" || echo "  ✗ figure16_results.json (not found)"
docker cp "$CONTAINER_ID:/fuseflow-artifact/figure17_results.json" "$OUTPUT_DIR/" 2>/dev/null && echo "  ✓ figure17_results.json" || echo "  ✗ figure17_results.json (not found)"

echo ""
echo "========================================"
echo "Extraction Complete!"
echo "========================================"
echo ""
echo "Results are available in: $OUTPUT_DIR/"
echo ""
echo "View the PDFs to validate against the paper figures."
echo ""
