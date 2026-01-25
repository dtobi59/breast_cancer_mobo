# Local Setup Guide

## Problem: ModuleNotFoundError: No module named 'src'

This error occurs when Python cannot find the `src` module. This guide provides multiple solutions.

---

## Quick Fix

The issue is that Python needs to know where to find the `breast_cancer_detection` module. Add this at the top of any script that imports from `src`:

```python
import sys
from pathlib import Path

# Add breast_cancer_detection to path
project_root = Path(__file__).parent.resolve()
bc_path = project_root / "breast_cancer_detection"
sys.path.insert(0, str(bc_path))

# Now imports work
from src.preprocessing import MammographyPreprocessor
from src.datasets import VinDRMammoBinaryDataset
```

---

## Solution 1: Run Setup Script (Recommended)

```bash
cd "C:\Users\HP\Downloads\New Project"
python setup_local.py
```

This script will:
- Verify all paths are correct
- Check if dependencies are installed
- Test all module imports
- Provide installation instructions for missing packages

---

## Solution 2: Use Local Optimization Script

Instead of the Jupyter notebook (designed for Colab), use the local Python script:

```bash
# Demo run (5 pop × 3 gen, ~4 hours)
python run_local_optimization.py \
  --mode standard \
  --scale demo \
  --data_root /path/to/vindr/images \
  --csv_file /path/to/vindr/metadata.csv

# Surrogate-assisted (recommended, 70% faster)
python run_local_optimization.py \
  --mode surrogate \
  --scale recommended \
  --data_root /path/to/vindr/images \
  --csv_file /path/to/vindr/metadata.csv \
  --output_dir results/surrogate_nsga3
```

This script automatically configures paths and runs optimization without Jupyter.

---

## Solution 3: Fix Jupyter Notebook for Local Use

If you want to use the notebook locally (not in Colab), modify Cell 2:

### Original (Colab version):
```python
PROJECT_PATH = "/content/drive/MyDrive/breast_cancer_detection"
```

### Fixed (Local version):
```python
import sys
from pathlib import Path

# Get project root automatically
PROJECT_PATH = Path.cwd()  # If running from project root
# OR
PROJECT_PATH = r"C:\Users\HP\Downloads\New Project\breast_cancer_detection"

if PROJECT_PATH.exists():
    print(f"✓ Project found: {PROJECT_PATH}")
    sys.path.insert(0, str(PROJECT_PATH))
else:
    raise FileNotFoundError(f"Project not found at {PROJECT_PATH}")
```

---

## Solution 4: Install as Package (Advanced)

Create `setup.py` in the project root and install as editable package:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="breast_cancer_detection",
    version="1.0.0",
    packages=find_packages(),
)
```

Then install:
```bash
cd "C:\Users\HP\Downloads\New Project"
pip install -e breast_cancer_detection
```

After this, imports will work from anywhere:
```python
from src.preprocessing import MammographyPreprocessor  # Works!
```

---

## Verify Setup

Run the diagnostic test:

```bash
python test_imports.py
```

Expected output:
```
================================================================================
ATTEMPTING IMPORTS
================================================================================

1. Importing src.preprocessing...
   [OK] SUCCESS: src.preprocessing.MammographyPreprocessor

2. Importing src.datasets...
   [OK] SUCCESS: src.datasets

3. Importing src.optimization...
   [OK] SUCCESS: src.optimization.BreastCancerOptimizationProblem

4. Importing src.surrogate_optimizer...
   [OK] SUCCESS: src.surrogate_optimizer.MultiObjectiveGPSurrogate

5. Importing src.acquisition...
   [OK] SUCCESS: src.acquisition.get_acquisition_function
```

---

## Common Issues

### Issue: "No module named 'torch'"
**Solution:** Install PyTorch
```bash
pip install torch torchvision
```

### Issue: "No module named 'pymoo'"
**Solution:** Install pymoo
```bash
pip install pymoo
```

### Issue: "No module named 'cv2'"
**Solution:** Install OpenCV
```bash
pip install opencv-python
```

### Issue: Still getting ModuleNotFoundError
**Debug steps:**
1. Check current working directory: `print(os.getcwd())`
2. Check sys.path: `print(sys.path)`
3. Verify breast_cancer_detection exists: `ls breast_cancer_detection/`
4. Verify src exists: `ls breast_cancer_detection/src/`

---

## File Structure

Your project should look like this:

```
C:\Users\HP\Downloads\New Project\
├── breast_cancer_detection/
│   ├── __init__.py                    # ✓ Created
│   ├── src/
│   │   ├── __init__.py                # ✓ Exists
│   │   ├── preprocessing.py
│   │   ├── datasets.py
│   │   ├── optimization.py
│   │   ├── surrogate_optimizer.py     # ✓ New
│   │   ├── acquisition.py             # ✓ New
│   │   └── ...
│   └── scripts/
│       └── run_nsga3_surrogate.py     # ✓ New
├── setup_local.py                      # ✓ New
├── run_local_optimization.py          # ✓ New
├── test_imports.py                    # ✓ New
└── vindr_nsga3_optimization.ipynb     # ✓ Updated
```

---

## Next Steps

After fixing imports:

1. **Test imports**: `python test_imports.py`
2. **Run setup verification**: `python setup_local.py`
3. **Run optimization**:
   - For local Python: `python run_local_optimization.py --help`
   - For Colab: Upload updated notebook to Google Drive
   - For local Jupyter: Fix paths in Cell 2 as shown above

---

## Getting Help

If you still have issues:

1. Run: `python setup_local.py` and share the output
2. Run: `python test_imports.py` and share the output
3. Check: `python --version` (should be 3.8+)
4. Check: `pip list | grep -E "torch|pymoo|numpy"`

---

## Summary

**The root cause** is that Python doesn't know where `breast_cancer_detection` is located.

**The fix** is to add it to `sys.path`:
```python
sys.path.insert(0, str(Path("/path/to/breast_cancer_detection")))
```

**For convenience**, use the provided scripts:
- `setup_local.py` - Verify everything works
- `run_local_optimization.py` - Run optimization locally
- `test_imports.py` - Debug import issues
