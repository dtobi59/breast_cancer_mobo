"""
Test script to diagnose import issues.
Run this from the project root to verify module imports work correctly.
"""

import sys
import os
from pathlib import Path

print("="*80)
print("IMPORT DIAGNOSTIC TEST")
print("="*80)

# Get current directory
current_dir = Path(__file__).parent.resolve()
print(f"\nCurrent directory: {current_dir}")

# Check if breast_cancer_detection exists
bc_dir = current_dir / "breast_cancer_detection"
print(f"\nBreast cancer detection dir exists: {bc_dir.exists()}")
print(f"Path: {bc_dir}")

# Check if src exists
src_dir = bc_dir / "src"
print(f"\nSrc dir exists: {src_dir.exists()}")
print(f"Path: {src_dir}")

# List contents of src
if src_dir.exists():
    print(f"\nContents of src/:")
    for item in sorted(src_dir.iterdir()):
        print(f"  - {item.name}")

# Add to path
print(f"\n" + "="*80)
print("TESTING IMPORTS")
print("="*80)

if str(bc_dir) not in sys.path:
    print(f"\nAdding to sys.path: {bc_dir}")
    sys.path.insert(0, str(bc_dir))
else:
    print(f"\nAlready in sys.path: {bc_dir}")

print(f"\nsys.path entries:")
for i, p in enumerate(sys.path[:5]):
    print(f"  [{i}] {p}")

# Try imports
print(f"\n" + "="*80)
print("ATTEMPTING IMPORTS")
print("="*80)

try:
    print("\n1. Importing src.preprocessing...")
    from src.preprocessing import MammographyPreprocessor
    print("   [OK] SUCCESS: src.preprocessing.MammographyPreprocessor")
except ImportError as e:
    print(f"   [FAIL] {e}")

try:
    print("\n2. Importing src.datasets...")
    from src.datasets import VinDRMammoBinaryDataset, create_breast_level_splits
    print("   [OK] SUCCESS: src.datasets")
except ImportError as e:
    print(f"   [FAIL] {e}")

try:
    print("\n3. Importing src.optimization...")
    from src.optimization import BreastCancerOptimizationProblem
    print("   [OK] SUCCESS: src.optimization.BreastCancerOptimizationProblem")
except ImportError as e:
    print(f"   [FAIL] {e}")

try:
    print("\n4. Importing src.surrogate_optimizer...")
    from src.surrogate_optimizer import MultiObjectiveGPSurrogate
    print("   [OK] SUCCESS: src.surrogate_optimizer.MultiObjectiveGPSurrogate")
except ImportError as e:
    print(f"   [FAIL] {e}")

try:
    print("\n5. Importing src.acquisition...")
    from src.acquisition import get_acquisition_function
    print("   [OK] SUCCESS: src.acquisition.get_acquisition_function")
except ImportError as e:
    print(f"   [FAIL] {e}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
