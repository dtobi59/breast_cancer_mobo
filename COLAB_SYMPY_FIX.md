# Fix: AttributeError: module 'sympy' has no attribute 'printing'

## Problem
PyTorch requires a compatible version of SymPy, but an incompatible version is installed.

## Quick Fix for Google Colab

Add this cell **before** importing PyTorch modules and run it:

```python
# Cell 1: Fix SymPy compatibility
!pip uninstall -y sympy
!pip install -q 'sympy>=1.12'

# Restart runtime after installation
print("✓ SymPy upgraded. IMPORTANT: Runtime > Restart runtime, then continue.")
```

**IMPORTANT:** After running this cell, you MUST restart the runtime:
- Click: **Runtime** → **Restart runtime**
- Then re-run all cells from the beginning

---

## Alternative: Force Reinstall in Single Cell

If the above doesn't work, use this more aggressive approach:

```python
# Force reinstall both PyTorch and SymPy
import sys
import subprocess

print("Fixing SymPy/PyTorch compatibility...")

# Uninstall
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "sympy"], check=False)

# Install compatible version
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy>=1.12"], check=True)

print("✓ Done! RESTART RUNTIME NOW: Runtime > Restart runtime")
```

---

## For the Updated Notebook

Add this as **Cell 2** (right after GPU check, before imports):

```python
# ============================================================================
# CELL 2: Fix SymPy Compatibility (Run once, then restart runtime)
# ============================================================================

import subprocess
import sys

# Check if fix is needed
try:
    import sympy
    has_printing = hasattr(sympy, 'printing')

    if not has_printing:
        print("[WARNING] SymPy version incompatible with PyTorch")
        print("Upgrading SymPy...")

        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "sympy"],
                      capture_output=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy>=1.12"],
                      check=True)

        print("✓ SymPy upgraded to compatible version")
        print("")
        print("="*80)
        print("IMPORTANT: Runtime > Restart runtime")
        print("Then re-run all cells")
        print("="*80)
    else:
        print(f"✓ SymPy {sympy.__version__} is compatible")

except ImportError:
    print("Installing SymPy...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy>=1.12"],
                  check=True)
    print("✓ SymPy installed")
```

---

## Permanent Fix in Notebook

Update the notebook's dependency installation cell:

### Original:
```python
!pip install -q pydicom opencv-python-headless scikit-image
!pip install -q pymoo
!pip install -q torch torchvision
```

### Fixed:
```python
# Install dependencies with compatible versions
!pip install -q pydicom opencv-python-headless scikit-image
!pip install -q pymoo
!pip install -q scikit-learn scipy
!pip install -q 'sympy>=1.12'  # ← Add this line BEFORE PyTorch
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Why This Happens

- **PyTorch 2.0+** requires **SymPy 1.12+** for its symbolic computation utilities
- Older SymPy versions (< 1.12) don't have the `sympy.printing` module structure PyTorch expects
- Google Colab sometimes has outdated SymPy cached

---

## Verification

After restarting runtime, verify the fix:

```python
import sympy
import torch

print(f"SymPy version: {sympy.__version__}")
print(f"Has 'printing': {hasattr(sympy, 'printing')}")
print(f"PyTorch version: {torch.__version__}")

# Test the problematic import
from torch.utils import _sympy
print("✓ torch.utils._sympy imported successfully")
```

Expected output:
```
SymPy version: 1.12 (or higher)
Has 'printing': True
PyTorch version: 2.x.x
✓ torch.utils._sympy imported successfully
```

---

## If Still Failing

Try clearing package cache:

```python
!pip cache purge
!pip uninstall -y sympy torch torchvision
!pip install -q 'sympy>=1.12'
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118

# RESTART RUNTIME
```

---

## Summary

1. **Add to notebook**: `!pip install -q 'sympy>=1.12'` before PyTorch installation
2. **Restart runtime**: Runtime → Restart runtime
3. **Re-run cells**: Start from the beginning

The error should be resolved!
