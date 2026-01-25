"""
Reload updated modules after bug fixes.

Run this in your Colab/Jupyter notebook before running optimization:

    %run reload_modules.py

Or execute the code below directly in a cell.
"""

import sys
import importlib

# Remove cached modules
modules_to_reload = [
    'src.datasets',
    'src.optimization',
    'src.preprocessing',
    'src.adaptive_preprocessing',
    'src.training',
    'src.models',
    'src.evaluation',
    'src.robustness',
    'src.augmentations',
    'src.domain_adaptation',
    'src.cache_manager'
]

print("Reloading updated modules...")
for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"  Reloading: {module_name}")
        importlib.reload(sys.modules[module_name])
    else:
        print(f"  Not loaded yet: {module_name}")

print("\n✓ Modules reloaded successfully!")
print("\nYou can now run your optimization without the ValueError.")
