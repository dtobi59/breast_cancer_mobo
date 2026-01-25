# Main Experiment Setup Guide

## Pre-Flight Checklist

### ✅ 1. Data Preparation

**VinDr-Mammo Dataset**
- [ ] Download complete VinDr-Mammo dataset
- [ ] Upload to Google Drive: `/MyDrive/vindr-mammo/images/`
- [ ] Verify CSV file: `/MyDrive/vindr-mammo/metadata/stratified_selection.csv`
- [ ] Confirm dataset size: ~15,000+ images
- [ ] Test loading a few samples to verify paths

**INbreast Dataset**
- [ ] Download INbreast DICOM files
- [ ] Upload to Google Drive: `/MyDrive/INbreast/AllDICOMs/`
- [ ] Verify CSV file: `/MyDrive/INbreast/INbreast.csv`
- [ ] Confirm dataset size: ~400 images
- [ ] Test loading a few samples to verify paths

### ✅ 2. Code Upload

**Project Structure**
```
/content/drive/MyDrive/breast_cancer_detection/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── adaptive_preprocessing.py
│   ├── domain_adaptation.py
│   ├── datasets.py
│   ├── models.py
│   ├── augmentations.py
│   ├── training.py
│   ├── evaluation.py
│   ├── robustness.py
│   ├── optimization.py
│   └── cache_manager.py
├── cache/ (will be created automatically)
├── checkpoints/ (will be created automatically)
├── optimization_checkpoints/ (will be created automatically)
└── logs/ (will be created automatically)
```

**Upload Checklist**
- [ ] Upload complete `breast_cancer_detection/` folder to Google Drive
- [ ] Verify all Python files are present in `src/`
- [ ] Check that `__init__.py` exists in `src/`
- [ ] Ensure no syntax errors in any files

### ✅ 3. Google Colab Setup

**Runtime Configuration**
- [ ] Open Google Colab: https://colab.research.google.com
- [ ] Upload `breast_cancer_colab_demo.ipynb`
- [ ] Change runtime type to **GPU** (Runtime > Change runtime type > GPU)
- [ ] Verify GPU is Tesla T4 or better (run `!nvidia-smi`)
- [ ] Check RAM: Should be at least 12 GB

**Google Colab Pro Recommendations** (for 50+ hour runs)
- [ ] Consider upgrading to Colab Pro for:
  - Longer runtime (24 hours vs 12 hours)
  - Better GPUs (A100, V100)
  - More RAM
  - Background execution
- [ ] Or use Colab Pro+ for unlimited runtime

**Keep-Alive Strategy** (for free Colab)
- [ ] Install browser extension to prevent disconnection
- [ ] Options:
  - "Colab Alive" Chrome extension
  - Manual: Keep browser tab active and interact every 30-60 mins
  - Use Colab Pro for unattended runs

### ✅ 4. Environment Configuration

**Python Dependencies**
All will be installed automatically by Section 1, but verify:
- PyTorch ≥ 2.0
- torchvision
- pydicom
- opencv-python-headless
- scikit-image
- scikit-learn
- pymoo
- numpy
- pandas
- matplotlib

### ✅ 5. Notebook Configuration for Production

**Section 3: Preprocessing Configuration**
```python
USE_ADAPTIVE_PREPROCESSING = True  # Enable domain adaptation
```

**Section 8: Dataset Selection**
```python
USE_FULL_DATASET = True  # Use entire dataset (CRITICAL for production)
```

**Section 11: NSGA-III Configuration**
```python
ENABLE_CHECKPOINTING = True  # MUST be enabled
RESUME_IF_AVAILABLE = True   # MUST be enabled
RUN_ID = "production_run_YYYYMMDD"  # Use unique ID with date
```

**Production Parameters** (modify in Section 11 code):
```python
# Change these values in the NSGA-III cell:
total_generations = 50  # Change from 3 to 50
algorithm = NSGA3(
    ref_dirs=ref_dirs,
    pop_size=20  # Change from 5 to 20
)
```

### ✅ 6. Pre-Experiment Tests

**Run Demo Mode First** (1-2 hours)
- [ ] Set `USE_FULL_DATASET = False`
- [ ] Set `total_generations = 3` in Section 11
- [ ] Run complete notebook end-to-end
- [ ] Verify all sections execute without errors
- [ ] Check that checkpointing works (stop and resume)
- [ ] Verify model saving works
- [ ] Confirm INbreast evaluation runs

**Entropy Cache Pre-Computation** (1-2 hours)
- [ ] Run Section 16 to compute entropy statistics
- [ ] Verify cache file created: `cache/entropy_stats_vindr_train.json`
- [ ] Check cache contains valid statistics (mean, std, etc.)
- [ ] This only needs to be done ONCE

### ✅ 7. Production Run Setup

**Expected Timeline**
- **NSGA-III Optimization**: 20-50 hours
  - 20 population × 50 generations = 1,000 evaluations
  - ~1-3 minutes per evaluation (depends on GPU)
  - With checkpointing: Can spread across multiple sessions

- **Full Pipeline**: ~24-60 hours total
  - Sections 1-10: ~2-4 hours
  - Section 11 (NSGA-III): ~20-50 hours
  - Sections 12-17: ~1-2 hours

**Resource Requirements**
- **GPU**: Tesla T4 or better
- **RAM**: 12-16 GB
- **Disk**: ~20 GB free on Google Drive
- **Time**: 24-60 hours (can pause/resume)

### ✅ 8. Monitoring & Logging

**What to Monitor**
- [ ] GPU utilization: Should be 80-100% during training
- [ ] Memory usage: Should not exceed 90%
- [ ] Generation time: Should be consistent (~2-5 min/gen)
- [ ] Checkpoint saves: Verify after each generation

**Check Progress**
- Run checkpoint management cell to see:
  - Current generation
  - Evaluations completed
  - Average time per generation
  - Estimated time remaining

### ✅ 9. Handling Interruptions

**If Colab Disconnects**
1. Reconnect to runtime
2. Mount Google Drive (Section 2)
3. Re-run Section 1-3 (setup)
4. Jump directly to Section 11 (NSGA-III)
5. Keep same `RUN_ID` and `RESUME_IF_AVAILABLE = True`
6. Optimization will automatically resume from last checkpoint

**Checkpoint Strategy**
- Checkpoints saved after every generation
- ~2-5 MB per checkpoint
- Automatically cleaned up when complete
- Final results saved separately

### ✅ 10. Post-Experiment Analysis

**After NSGA-III Completes**
- [ ] Section 12: Visualize Pareto Front
- [ ] Section 13-14: Load INbreast and evaluate zero-shot
- [ ] Section 15-16: Compare baseline vs adapted performance
- [ ] Section 17: Review summary

**Expected Outputs**
- Pareto front with ~5-20 non-dominated solutions
- Trade-offs between PR-AUC, AUROC, Brier, Robustness
- Zero-shot performance on INbreast
- Comparison: Standard vs Adaptive preprocessing

---

## Quick Start Commands

### Initial Setup (Run Once)
```bash
# 1. Upload data to Google Drive
# 2. Upload code to /MyDrive/breast_cancer_detection/
# 3. Open Colab, change runtime to GPU
# 4. Run Sections 1-3
```

### Pre-Compute Entropy Cache (Run Once, 1-2 hours)
```python
# Section 16: Run entropy computation
# This creates cache/entropy_stats_vindr_train.json
# Only needs to be done once, reused for all experiments
```

### Start Production Run
```python
# Section 8:
USE_FULL_DATASET = True

# Section 11:
RUN_ID = "production_YYYYMMDD_HH"
ENABLE_CHECKPOINTING = True
RESUME_IF_AVAILABLE = True
total_generations = 50  # Modify in code
pop_size = 20           # Modify in code
```

### Resume After Interruption
```python
# Just re-run Section 11 with same RUN_ID
# Will automatically detect checkpoint and continue
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` in Section 11 (try 2 instead of 4)
- Restart runtime and clear GPU memory
- Use smaller model or reduce image size (not recommended)

### "Runtime disconnected"
- Normal for free Colab after 12 hours
- Just reconnect and resume (checkpoints will save progress)
- Consider Colab Pro for 24-hour sessions

### "Checkpoint corrupted"
- Will automatically fall back to new optimization
- Previous work is lost only for current generation
- Manually delete corrupted checkpoint and restart

### "Entropy cache not found"
- Run Section 16 first to compute statistics
- Takes 1-2 hours for full dataset
- Only needs to be done once

### Slow training
- Verify GPU is being used (`!nvidia-smi`)
- Check GPU utilization (should be >80%)
- Reduce `num_workers` if CPU bottleneck
- Verify data is on mounted Google Drive

---

## Production Checklist

Before starting 20-50 hour run:

- [ ] ✅ All data uploaded to Google Drive
- [ ] ✅ All code uploaded to Google Drive
- [ ] ✅ GPU runtime selected in Colab
- [ ] ✅ Demo run completed successfully (1-2 hours)
- [ ] ✅ Entropy cache pre-computed (Section 16)
- [ ] ✅ `USE_FULL_DATASET = True`
- [ ] ✅ `ENABLE_CHECKPOINTING = True`
- [ ] ✅ `total_generations = 50`
- [ ] ✅ `pop_size = 20`
- [ ] ✅ Unique `RUN_ID` set
- [ ] ✅ Google Drive has 20+ GB free space
- [ ] ✅ Browser keep-alive strategy in place (if using free Colab)
- [ ] ✅ Backup plan for disconnections (know how to resume)

---

## Expected Results

### VinDr-Mammo Validation
- AUROC: 0.85-0.95
- PR-AUC: 0.80-0.92
- Brier: 0.10-0.20

### INbreast Zero-Shot (Baseline)
- AUROC: 0.50-0.60 (poor, domain shift)
- Prediction spread: 5-10% (collapsed)

### INbreast Zero-Shot (Adaptive)
- AUROC: 0.65-0.80 (improved)
- Prediction spread: 50-80% (restored)
- Improvement: +0.10 to +0.25 AUROC

### Pareto Front
- 5-20 non-dominated solutions
- Trade-offs visible between metrics
- Best solution depends on clinical priorities

---

## Support

If you encounter issues:
1. Check troubleshooting section above
2. Verify all checklist items completed
3. Run demo mode first to isolate issues
4. Check Google Drive permissions and storage
5. Verify GPU is available and being used

Good luck with your experiment! 🚀
