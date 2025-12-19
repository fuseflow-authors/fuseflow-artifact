#!/bin/bash
# Patch PyTorch Geometric to disable weights_only for dataset loading
# This fixes EOFError and RuntimeError when loading datasets

PYG_FS_FILE="/opt/fuseflow-venv/lib/python3.10/site-packages/torch_geometric/io/fs.py"

echo "Patching PyTorch Geometric fs.py to disable weights_only..."

# Backup original file
cp "$PYG_FS_FILE" "${PYG_FS_FILE}.bak"

# Replace weights_only=True with weights_only=False
sed -i 's/weights_only=True/weights_only=False/g' "$PYG_FS_FILE"

# Verify the patch
if grep -q "weights_only=False" "$PYG_FS_FILE"; then
    echo "✓ Successfully patched PyTorch Geometric"
    echo "  Changed: weights_only=True -> weights_only=False"
else
    echo "✗ Patch failed! Restoring backup..."
    cp "${PYG_FS_FILE}.bak" "$PYG_FS_FILE"
    exit 1
fi

# Also clear any corrupted dataset cache
echo ""
echo "Clearing corrupted dataset cache..."
rm -rf /tmp/data/*
echo "✓ Dataset cache cleared"

echo ""
echo "Patch complete! You can now run benchmarks."
