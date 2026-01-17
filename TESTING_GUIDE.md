# Testing Guide - Breast Cancer Detection

This guide helps you verify the implementation is correct before running full experiments.

---

## ✅ Pre-Flight Checklist

### 1. Install Dependencies

```bash
cd breast_cancer_detection
pip install -r requirements.txt
```

### 2. Run Installation Test

```bash
python test_installation.py
```

**Expected output:**
```
Testing imports...
  ✓ PyTorch 2.x.x
  ✓ NumPy 1.x.x
  ✓ Pandas 2.x.x
  ✓ OpenCV 4.x.x
  ✓ pydicom 3.x.x
  ✓ pymoo available
  ✓ All project modules imported successfully

Testing preprocessing...
  ✓ Preprocessor initialized correctly

Testing model building...
  ✓ Feature extraction only: X.X% trainable
  ✓ Partial fine-tuning: X.X% trainable
  ✓ Full fine-tuning: 100.0% trainable
  ✓ All model configurations work correctly

Testing augmentation...
  ✓ Augmentation working for all strengths

Testing Noisy-OR aggregation...
  ✓ Noisy-OR formula verified

Testing optimization...
  ✓ Optimization problem class available

Test Summary
================================================================================
✓ PASS   Imports
✓ PASS   Preprocessing
✓ PASS   Model
✓ PASS   Augmentation
✓ PASS   Noisy-OR
✓ PASS   Optimization

Passed: 6/6

✓ All tests passed! Installation is correct.
```

---

## 🔬 Component Testing

### Test 1: Preprocessing Pipeline

```python
from src.preprocessing import MammographyPreprocessor

preprocessor = MammographyPreprocessor()

# Test on a DICOM file
img = preprocessor("/path/to/sample.dicom")

# Verify output
assert img.shape == (480, 720, 3), "Shape should be (480, 720, 3)"
assert img.dtype == np.uint8, "Should be uint8"
assert img.min() >= 0 and img.max() <= 255, "Values in [0, 255]"

print("✓ Preprocessing working correctly")
```

### Test 2: Dataset Loading

```python
from src.datasets import VinDRMammoBinaryDataset

dataset = VinDRMammoBinaryDataset(
    images_root="/path/to/vindr-mammo/images",
    csv_file="/path/to/metadata.csv",
    preprocessor=preprocessor
)

img, label = dataset[0]

assert img.shape == (3, 480, 720), "Should be (C, H, W)"
assert label in [0, 1], "Binary label"
assert 0.0 <= img.min() <= img.max() <= 1.0, "Normalized to [0, 1]"

print(f"✓ Dataset loaded: {len(dataset)} samples")
```

### Test 3: Model Forward Pass

```python
from src.models import build_resnet152
import torch

model = build_resnet152(pretrained=False, dropout=0.2, unfreeze_fraction=0.5)

# Test forward pass
batch = torch.randn(4, 3, 480, 720)
output = model(batch)

assert output.shape == (4,), f"Expected (4,), got {output.shape}"
print("✓ Model forward pass successful")
```

### Test 4: Noisy-OR Aggregation

```python
from src.evaluation import noisy_or_aggregation

# Test: p = 1 - (1-0.5)(1-0.5) = 1 - 0.25 = 0.75
result = noisy_or_aggregation([0.5, 0.5])
assert abs(result - 0.75) < 0.001, f"Expected 0.75, got {result}"

print("✓ Noisy-OR formula verified")
```

### Test 5: Training Pipeline (Quick)

```python
from src.training import train_model
from torch.utils.data import DataLoader, Subset

# Use small subset
small_dataset = Subset(dataset, range(50))

train_loader = DataLoader(small_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(small_dataset, batch_size=4)

model = build_resnet152(pretrained=False, dropout=0.2, unfreeze_fraction=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train for 2 epochs
model, metrics = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    val_dataset=small_dataset,
    device=device,
    learning_rate=1e-4,
    weight_decay=1e-4,
    patience=2,
    max_epochs=2,
    verbose=True
)

print(f"✓ Training completed - PR-AUC: {metrics['pr_auc']:.4f}")
```

---

## 🧪 Integration Tests

### Full Pipeline Test

Run the example training script:

```bash
python scripts/train_single_model.py \
    --lr 1e-4 \
    --wd 1e-4 \
    --dropout 0.2 \
    --aug_strength 0.5 \
    --unfreeze_frac 1.0 \
    --max_epochs 5 \
    --save_model \
    --model_name test_model.pth
```

**What to expect:**
- Training starts without errors
- Loss decreases over epochs
- Validation metrics are computed
- Model is saved to `checkpoints/test_model.pth`

### Small-Scale NSGA-III Test

```bash
python scripts/run_nsga3.py \
    --pop_size 3 \
    --n_gen 2 \
    --max_epochs 5 \
    --run_id "test"
```

**What to expect:**
- 3 × 2 = 6 evaluations total
- Each evaluation trains a model
- Results saved to `checkpoints/nsga3_results_test.pkl`
- Logs saved to `logs/nsga3_run_test.csv`

---

## 🐛 Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Solution:** Reduce batch size in `configs/config.py`:
```python
BATCH_SIZE = 2  # Instead of 4
```

### Issue 2: Dataset Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:** Update paths in `configs/config.py`:
```python
VINDR_IMAGES_ROOT = Path("/correct/path/to/vindr-mammo/images")
VINDR_CSV = Path("/correct/path/to/metadata.csv")
```

### Issue 3: Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Run from project root:
```bash
cd breast_cancer_detection
python scripts/train_single_model.py
```

Or add to Python path:
```python
import sys
sys.path.insert(0, "/path/to/breast_cancer_detection")
```

### Issue 4: Preprocessing Fails

**Error:** `RuntimeError: Failed preprocessing`

**Possible causes:**
1. Corrupted DICOM file - skip it
2. Missing DICOM tags - check with `pydicom.dcmread()`
3. Unusual image dimensions - verify source data

**Debug:**
```python
import pydicom
ds = pydicom.dcmread("problem_file.dicom")
print(ds)  # Check DICOM tags
```

### Issue 5: Early Stopping Too Early

**Symptom:** Training stops after 1-2 epochs

**Solution:** Increase patience:
```python
EARLY_STOPPING_PATIENCE = 20  # Instead of 10
```

---

## 📊 Expected Results

### Baseline (from notebook)

On VinDr-Mammo validation (image-level):
- **AUROC:** ~0.95
- **Training time:** 10 epochs with early stopping

### After Optimization

Expected range for Pareto solutions:
- **PR-AUC:** 0.85 - 0.95
- **AUROC:** 0.90 - 0.96
- **Brier Score:** 0.05 - 0.15
- **Robustness Degradation:** 0.01 - 0.10

### Zero-Shot on INbreast

Expect performance drop:
- **AUROC:** 0.80 - 0.90 (vs 0.90-0.96 on VinDr)
- This is normal for cross-dataset generalization

---

## ✅ Validation Checklist

Before running full experiments, verify:

- [ ] `test_installation.py` passes all tests
- [ ] Can load VinDr-Mammo dataset
- [ ] Can load INbreast dataset
- [ ] Preprocessing produces correct shape (480, 720, 3)
- [ ] Model forward pass works
- [ ] Single model training completes without errors
- [ ] NSGA-III runs (even with small scale)
- [ ] Results are saved correctly
- [ ] GPU is detected and used (if available)

---

## 🚀 Ready for Production

Once all tests pass, you can run:

```bash
# Full NSGA-III optimization (20-50 hours)
python scripts/run_nsga3.py \
    --pop_size 20 \
    --n_gen 50 \
    --max_epochs 100 \
    --run_id "production_001"

# Zero-shot evaluation
python scripts/evaluate_zeroshot.py \
    --results_file checkpoints/nsga3_results_production_001.pkl \
    --checkpoint_dir checkpoints/ \
    --run_id "production_001"
```

---

## 📝 Notes

1. **Random Seeds:** All fixed at 42 for reproducibility
2. **Data Split:** Single fixed 80/20 split, never changes
3. **Preprocessing:** Fixed pipeline, DO NOT modify
4. **BI-RADS Mapping:** 1-3=benign, 5-6=malignant, exclude 4
5. **Noisy-OR:** Always used for breast-level evaluation

---

## 🆘 Getting Help

If tests fail:
1. Check error messages carefully
2. Verify data paths are correct
3. Ensure all dependencies are installed
4. Check GPU memory (reduce batch size if needed)
5. Review `configs/config.py` settings

For issues, check:
- README.md for detailed documentation
- CLAUDE.md for research specifications
- Source code comments for implementation details
