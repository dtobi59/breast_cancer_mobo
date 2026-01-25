# Breast Cancer Detection - Production Ready

## 🎯 Quick Start for Main Experiment

### 1. Open Notebook in Google Colab
1. Upload `breast_cancer_colab_demo.ipynb` to Google Colab
2. Change runtime to GPU: `Runtime > Change runtime type > GPU`
3. Verify GPU: Run `!nvidia-smi` - should show Tesla T4 or better

### 2. Upload Data to Google Drive
```
/MyDrive/vindr-mammo/
├── images/              # VinDr-Mammo DICOM files
└── metadata/
    └── stratified_selection.csv

/MyDrive/INbreast/
├── AllDICOMs/          # INbreast DICOM files
└── INbreast.csv
```

### 3. Upload Code to Google Drive
```
/MyDrive/breast_cancer_detection/
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
```

### 4. Configure for Production

**In the notebook, find the EXPERIMENT CONFIGURATION cell (after Section 2):**

```python
# Change this line:
EXPERIMENT_MODE = "DEMO"  # ← Change to "PRODUCTION"
```

**That's it!** All settings will automatically configure for production:
- ✅ Full dataset (not subset)
- ✅ 20 population × 50 generations
- ✅ 50 epochs per model
- ✅ Checkpointing enabled
- ✅ Adaptive preprocessing enabled
- ✅ Auto-resume on disconnection

### 5. Run the Experiment

**First Time: Pre-compute Entropy Cache (1-2 hours)**
```python
# Run Section 16 FIRST (only needed once)
# This computes entropy statistics from training set
# Creates: cache/entropy_stats_vindr_train.json
```

**Then: Run Full Pipeline**
```python
# Run Sections 1-17 sequentially
# Total time: 20-50 hours
# Will auto-save checkpoints every generation
```

**If Disconnected: Resume Automatically**
```python
# Re-run Sections 1-2 (mount drive)
# Jump to Section 11 (NSGA-III)
# Will detect checkpoint and continue automatically
```

---

## 📊 What to Expect

### Timeline
- **Demo Mode**: 1-2 hours (for testing)
- **Production Mode**: 20-50 hours (actual experiment)

### Resources
- **GPU**: Tesla T4 minimum (T4, V100, A100 all work)
- **RAM**: 12-16 GB
- **Storage**: 20 GB free on Google Drive
- **Colab**: Pro recommended for 24h sessions

### Outputs

**During Training:**
- Checkpoints saved after each generation: `optimization_checkpoints/`
- Model checkpoints: `checkpoints/`
- Progress logs in notebook output

**After Completion:**
- Pareto front with 5-20 non-dominated solutions
- Trade-offs between PR-AUC, AUROC, Brier, Robustness
- Zero-shot evaluation on INbreast
- Baseline vs Adaptive preprocessing comparison

### Expected Performance

**VinDr-Mammo Validation:**
- AUROC: 0.85-0.95
- PR-AUC: 0.80-0.92

**INbreast Zero-Shot (Baseline):**
- AUROC: 0.50-0.60 (poor - domain shift)
- Prediction spread: 5-10%

**INbreast Zero-Shot (Adaptive):**
- AUROC: 0.65-0.80 (improved!)
- Prediction spread: 50-80%
- **Improvement: +0.10 to +0.25 AUROC**

---

## 🛠️ Features

### ✅ One-Click Mode Switching
Change between DEMO and PRODUCTION with one variable:
```python
EXPERIMENT_MODE = "DEMO"       # Fast testing (1-2 hours)
EXPERIMENT_MODE = "PRODUCTION" # Main experiment (20-50 hours)
```

### ✅ Automatic Checkpointing
- Saves after every generation
- No work lost on disconnection
- Auto-resumes from last checkpoint
- Works across multiple Colab sessions

### ✅ Full Dataset Support
- Stratified 80/20 train/val split
- Maintains class balance
- Configurable batch size
- Optimized for GPU

### ✅ Entropy-Based Domain Adaptation
- Pre-computes statistics from training set
- Caches for reuse
- Applies to both validation and test sets
- No model retraining required

### ✅ Multi-Objective Optimization
- NSGA-III algorithm
- 4 objectives: PR-AUC, AUROC, Brier, Robustness
- Generates Pareto front of solutions
- Checkpoint-enabled for long runs

### ✅ Complete Pipeline
- Preprocessing (standard or adaptive)
- Model training with early stopping
- Breast-level aggregation (Noisy-OR)
- Robustness evaluation
- Zero-shot transfer to INbreast
- Comparative analysis

---

## 🔄 Workflow

### First Run (Complete Pipeline)
1. **Section 1-2**: Setup environment and mount drive
2. **Section 9**: Configure for PRODUCTION mode
3. **Section 16**: Pre-compute entropy cache (1-2 hours, once)
4. **Section 3-15**: Run pipeline (preprocessing → training → optimization → evaluation)
5. **Section 16-17**: Domain adaptation analysis and summary

### Resume After Disconnection
1. **Section 1-2**: Re-mount drive
2. **Section 11**: Jump to NSGA-III
   - Will detect checkpoint automatically
   - Continues from last generation
   - No manual intervention needed

### Check Progress Anytime
Run the **Checkpoint Management** cell (after Section 12):
```python
# Shows:
# - Current generation
# - Evaluations completed
# - Average time per generation
# - Estimated time remaining
```

---

## 📁 Output Files

All saved to Google Drive automatically:

```
/MyDrive/breast_cancer_detection/
├── cache/
│   └── entropy_stats_vindr_train.json    # Entropy statistics (reusable)
├── checkpoints/
│   ├── resnet152_vindr_*.pth             # Trained models
│   └── *_metrics.json                     # Training metrics
├── optimization_checkpoints/
│   ├── nsga3_*_checkpoint.pkl            # Resume checkpoints
│   └── nsga3_*_results.pkl               # Final Pareto front
└── logs/                                  # Training logs (optional)
```

---

## 🚨 Common Issues

### "Runtime disconnected"
**Solution:** Normal for free Colab after 12 hours. Just reconnect and re-run Section 11. Checkpoint will resume automatically.

### "CUDA out of memory"
**Solution:** Reduce batch size in CONFIG (try 4 instead of 8).

### "Entropy cache not found"
**Solution:** Run Section 16 first to compute statistics (takes 1-2 hours, only once).

### Slow training
**Solution:** Verify GPU is enabled with `!nvidia-smi`. Should show GPU utilization >80%.

### "Checkpoint corrupted"
**Solution:** Delete corrupted checkpoint and restart from last valid one. Only lose 1 generation of work.

---

## 📖 Documentation

- **EXPERIMENT_SETUP_GUIDE.md**: Complete pre-flight checklist and troubleshooting
- **CLAUDE.md**: Project requirements and specifications
- **REORGANIZATION_SUMMARY.md**: Notebook structure documentation
- **This file**: Quick start for production runs

---

## ⚙️ Advanced Configuration

Want to customize? Edit the CONFIG dictionary in the EXPERIMENT CONFIGURATION cell:

```python
CONFIG = {
    'use_full_dataset': True,           # Use full dataset
    'train_batch_size': 8,              # Batch size
    'max_epochs': 50,                   # Training epochs
    'nsga3_population': 20,             # Population size
    'nsga3_generations': 50,            # Number of generations
    'enable_checkpointing': True,       # Enable checkpointing
    'resume_if_available': True,        # Auto-resume
    'run_id': 'production_...',         # Unique run ID
}
```

---

## 🎓 Experiment Goals

1. **Primary**: Demonstrate multi-objective optimization (NSGA-III) for hyperparameter tuning
2. **Secondary**: Show entropy-based domain adaptation mitigates dataset shift
3. **Outcome**: Pareto front revealing trade-offs between metrics
4. **Application**: Zero-shot transfer from VinDr-Mammo to INbreast

---

## ✅ Ready to Run?

**Pre-Flight Checklist:**
- [ ] GPU runtime selected
- [ ] Data uploaded to Google Drive
- [ ] Code uploaded to Google Drive
- [ ] Paths verified in Section 2
- [ ] `EXPERIMENT_MODE = "PRODUCTION"` set
- [ ] Entropy cache pre-computed (Section 16)
- [ ] 20+ GB free on Google Drive
- [ ] Browser keep-alive strategy (if free Colab)

**All set?** Run Sections 1-17 sequentially!

---

## 📊 Monitoring

### Real-Time Progress
```
Gen 1: 20 evals | Time: 245.3s | Avg: 245.3s/gen | Checkpoint saved
Gen 2: 40 evals | Time: 238.7s | Avg: 242.0s/gen | Checkpoint saved
Gen 3: 60 evals | Time: 251.2s | Avg: 245.1s/gen | Checkpoint saved
...
```

### Check Anytime
Run Checkpoint Management cell to see:
- Current status
- Completion percentage
- Time remaining estimate

---

## 🏆 Success Criteria

**Optimization:**
- ✅ 1000 evaluations completed (20 × 50)
- ✅ Pareto front generated (5-20 solutions)
- ✅ Trade-offs visible between metrics

**Domain Adaptation:**
- ✅ INbreast AUROC improves by >0.10
- ✅ Prediction spread increases to >50%
- ✅ Domain shift mitigated without retraining

**Reproducibility:**
- ✅ All results saved and checkpointed
- ✅ Can resume from any generation
- ✅ Same run_id produces same results (with same seed)

---

## 🚀 Let's Go!

**Estimated Total Time:** 20-50 hours
**Can Pause/Resume:** Yes, anytime
**GPU Required:** Yes, Tesla T4 minimum
**Cost:** Free (Colab) or $10/month (Colab Pro)

**Start the experiment:**
1. Set `EXPERIMENT_MODE = "PRODUCTION"`
2. Run all cells sequentially
3. Monitor progress in real-time
4. Results will be saved automatically

Good luck! 🎉
