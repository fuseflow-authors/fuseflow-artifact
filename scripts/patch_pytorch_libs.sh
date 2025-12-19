#!/bin/bash
# Patch PyTorch libraries for weights_only compatibility
# This script fixes PyTorch 2.6+ breaking changes in OGB and PyTorch Geometric

set -e

echo "Patching PyTorch libraries for weights_only compatibility..."
echo ""

# Patch OGB linkproppred
OGB_LINK_FILE="/opt/fuseflow-venv/lib/python3.10/site-packages/ogb/linkproppred/dataset.py"
if [ -f "$OGB_LINK_FILE" ]; then
    cp "$OGB_LINK_FILE" "${OGB_LINK_FILE}.backup"
    sed -i "s/torch\.load(pre_processed_file_path, 'rb')/torch.load(pre_processed_file_path, 'rb', weights_only=False)/g" "$OGB_LINK_FILE"
    echo "✓ Patched OGB linkproppred/dataset.py"
fi

# Patch OGB nodeproppred
OGB_NODE_FILE="/opt/fuseflow-venv/lib/python3.10/site-packages/ogb/nodeproppred/dataset.py"
if [ -f "$OGB_NODE_FILE" ]; then
    cp "$OGB_NODE_FILE" "${OGB_NODE_FILE}.backup"
    sed -i "s/torch\.load(pre_processed_file_path)/torch.load(pre_processed_file_path, weights_only=False)/g" "$OGB_NODE_FILE"
    echo "✓ Patched OGB nodeproppred/dataset.py"
fi

echo ""
echo "✓ All patches applied successfully"
echo ""
echo "Note: These patches are only needed in the running container."
echo "If you rebuild the Docker image, add this script to the Dockerfile:"
echo "  RUN /fuseflow-artifact/scripts/patch_pytorch_libs.sh"
