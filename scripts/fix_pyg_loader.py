#!/usr/bin/env python3
"""
Fix PyTorch Geometric dataset loading with PyTorch 2.6+

PyTorch 2.6+ changed weights_only default from False to True for security.
This script patches PyTorch Geometric's fs.py to allowlist PyG classes.
"""

import os
import sys

PYG_FS_PATH = "/opt/fuseflow-venv/lib/python3.10/site-packages/torch_geometric/io/fs.py"

# Read the current file
with open(PYG_FS_PATH, 'r') as f:
    content = f.read()

# Backup
backup_path = PYG_FS_PATH + '.backup'
if not os.path.exists(backup_path):
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✓ Created backup at {backup_path}")

# New torch_load function that properly handles PyG classes
new_torch_load = '''def torch_load(path: str, map_location: Any = None) -> Any:
    import torch_geometric.data
    if torch_geometric.typing.WITH_PT24:
        try:
            with fsspec.open(path, 'rb') as f:
                # Allowlist PyTorch Geometric classes for weights_only=True
                with torch.serialization.safe_globals([
                    torch_geometric.data.data.Data,
                    torch_geometric.data.hetero_data.HeteroData,
                ]):
                    return torch.load(f, map_location, weights_only=True)
        except pickle.UnpicklingError as e:
            # Fallback to weights_only=False if safe loading fails
            with fsspec.open(path, 'rb') as f:
                return torch.load(f, map_location, weights_only=False)
        except Exception:
            with fsspec.open(path, 'rb') as f:
                return torch.load(f, map_location, weights_only=False)
    else:
        with fsspec.open(path, 'rb') as f:
            return torch.load(f, map_location)
'''

# Find and replace the torch_load function
import re
# Match the function definition and everything until the next function or end
pattern = r'def torch_load\(path: str, map_location: Any = None\) -> Any:.*?(?=\ndef [a-z_]|\Z)'
new_content = re.sub(pattern, new_torch_load, content, flags=re.DOTALL)

if new_content != content:
    with open(PYG_FS_PATH, 'w') as f:
        f.write(new_content)
    print(f"✓ Successfully patched {PYG_FS_PATH}")
    print("  - Added safe_globals for PyG Data classes")
    print("  - Added fallback to weights_only=False")
else:
    print("✗ Failed to patch - pattern not found")
    sys.exit(1)

print("\n✓ Fix complete! PyTorch Geometric should now load datasets properly.")
