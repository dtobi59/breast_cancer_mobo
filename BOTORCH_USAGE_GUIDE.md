# BoTorch Multi-Objective Bayesian Optimization - Usage Guide

## Overview

Successfully migrated from NSGA-III to BoTorch-based multi-objective Bayesian optimization using qNEHVI (Quasi-Monte Carlo Noisy Expected Hypervolume Improvement) acquisition function.

## What Changed

### Files Removed (5 files)
- ❌ `breast_cancer_detection/scripts/run_nsga3.py`
- ❌ `breast_cancer_detection/scripts/run_nsga3_surrogate.py`
- ❌ `breast_cancer_detection/src/surrogate_optimizer.py`
- ❌ `breast_cancer_detection/src/acquisition.py`
- ❌ `breast_cancer_detection/src/optimization.py`

### Files Created (4 files)
- ✅ `breast_cancer_detection/src/evaluation_functions.py` - Standalone evaluation logic (refactored from optimization.py)
- ✅ `breast_cancer_detection/src/botorch_mobo.py` - GP models and qNEHVI acquisition
- ✅ `breast_cancer_detection/src/botorch_utils.py` - Transformations and checkpointing
- ✅ `breast_cancer_detection/scripts/run_botorch_mobo.py` - Main optimization script

### Dependencies Installed
- `botorch==0.16.1` - Multi-objective Bayesian optimization framework
- `gpytorch==1.15.1` - Gaussian Process library (required by BoTorch)

## Quick Start

### Basic Usage

```bash
python breast_cancer_detection/scripts/run_botorch_mobo.py \
    --data_root /path/to/vindr/images \
    --csv_file /path/to/vindr.csv \
    --n_initial 10 \
    --n_iterations 38 \
    --batch_size 5 \
    --output_dir results/botorch_mobo
```

### Small Test Run (Recommended First)

Test with minimal evaluations (~3.5 hours with max_epochs=5):

```bash
python breast_cancer_detection/scripts/run_botorch_mobo.py \
    --data_root /path/to/vindr/images \
    --csv_file /path/to/vindr.csv \
    --n_initial 3 \
    --n_iterations 2 \
    --batch_size 2 \
    --max_epochs 5 \
    --run_id test_small
```

This will:
- Run 3 initial Sobol samples
- Run 2 BO iterations with batch size 2
- Total: 3 + 2×2 = 7 evaluations

### Production Run

Full optimization with 200 evaluations (~4 days):

```bash
python breast_cancer_detection/scripts/run_botorch_mobo.py \
    --data_root /path/to/vindr/images \
    --csv_file /path/to/vindr.csv \
    --n_initial 10 \
    --n_iterations 38 \
    --batch_size 5 \
    --max_epochs 50 \
    --output_dir results/botorch_mobo \
    --run_id production_001
```

## Command-Line Arguments

### Data Parameters
- `--data_root` (required): Directory containing VinDr-Mammo images
- `--csv_file` (required): Path to VinDr-Mammo metadata CSV

### Optimization Parameters
- `--n_initial` (default: 10): Initial Sobol samples (2× dimensionality recommended)
- `--n_iterations` (default: 38): BO iterations after initial sampling
- `--batch_size` (default: 5): Candidates per BO iteration
- `--total_budget`: Total evaluation budget (overrides n_iterations)

### Acquisition Parameters
- `--acq_samples` (default: 128): MC samples for qNEHVI
- `--acq_restarts` (default: 20): Optimization restarts for acquisition
- `--ref_point_offset` (default: 0.1): Reference point offset (10%)

### Training Parameters
- `--train_batch_size` (default: 4): CNN training batch size
- `--max_epochs` (default: 50): Maximum CNN training epochs
- `--patience` (default: 10): Early stopping patience

### Checkpointing
- `--output_dir` (default: results/botorch_mobo): Output directory
- `--run_id`: Run identifier (default: timestamp)
- `--checkpoint_freq` (default: 5): Checkpoint every N iterations
- `--resume`: Resume from checkpoint path

### Hardware
- `--device` (default: cuda): Device (cuda or cpu)
- `--seed` (default: 42): Random seed

## Output Files

After running, the output directory contains:

```
results/botorch_mobo/[run_id]/
├── evaluations.csv           # All hyperparameters and objectives
├── checkpoint_iter5.pt       # Checkpoint at iteration 5
├── checkpoint_iter10.pt      # Checkpoint at iteration 10
├── ...
└── final_results.pkl         # Complete results with Pareto front
```

### evaluations.csv

CSV with all evaluated configurations:

```csv
learning_rate,weight_decay,dropout,aug_str,unfreeze,pr_auc,auroc,brier,robustness
0.000123,0.000456,0.25,0.5,0.5,0.8234,0.8567,0.1234,0.0345
...
```

### final_results.pkl

Pickle file containing:
- `X_all`: All evaluated hyperparameters (linear space)
- `Y_all`: All evaluated objectives
- `X_pareto`: Pareto-optimal hyperparameters
- `Y_pareto`: Pareto-optimal objectives
- `pareto_indices`: Indices of Pareto solutions
- `all_hyperparams`: List of hyperparameter dicts
- `gp_model`: Trained GP models
- `config`: Run configuration

## Resume from Checkpoint

If optimization is interrupted, resume from a checkpoint:

```bash
python breast_cancer_detection/scripts/run_botorch_mobo.py \
    --resume results/botorch_mobo/production_001/checkpoint_iter10.pt \
    --n_iterations 38  # Continue to iteration 38
```

The optimization will:
- Load GP models and training data
- Continue from iteration 11
- Preserve all previous evaluations

## Expected Performance

| Metric | Value |
|--------|-------|
| **Initial Sobol samples** | 10 |
| **BO iterations** | 38 |
| **Batch size** | 5 |
| **Total evaluations** | 200 |
| **Expected runtime** | ~4 days (30 min/eval on GPU) |

### Efficiency Gains

Compared to previous NSGA-III approaches:
- **vs NSGA-III (1000 evals):** 80% reduction
- **vs NSGA-III + Surrogate (295 evals):** 32% reduction

## Hyperparameters Optimized

1. **Learning rate:** [1e-5, 1e-3] (log10-scale)
2. **Weight decay:** [1e-6, 1e-2] (log10-scale)
3. **Dropout:** [0.0, 0.5]
4. **Augmentation strength:** [0.0, 1.0]
5. **Unfreeze fraction:** [0.0, 1.0]

## Objectives Minimized

1. **-PR-AUC** (maximize PR-AUC)
2. **-AUROC** (maximize AUROC)
3. **Brier score** (minimize calibration error)
4. **Robustness degradation** (minimize sensitivity to perturbations)

## Technical Details

### Gaussian Process Models
- **Kernel:** Matern 5/2 with ARD (Automatic Relevance Determination)
- **Model type:** Independent SingleTaskGP for each objective
- **Output standardization:** Automatic (zero mean, unit variance)

### Acquisition Function
- **Type:** qNEHVI (Quasi-Monte Carlo Noisy Expected Hypervolume Improvement)
- **Batch construction:** Sequential with fantasy models
- **MC samples:** 128 Sobol samples
- **Reference point:** Dynamic (worst + 10% buffer)

### Initialization
- **Method:** Sobol sequences (low-discrepancy)
- **Coverage:** 10 samples (2× dimensionality)

## Analyzing Results

### Load Results

```python
import pickle
import pandas as pd

# Load final results
with open('results/botorch_mobo/production_001/final_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract Pareto front
X_pareto = results['X_pareto']  # Hyperparameters
Y_pareto = results['Y_pareto']  # Objectives
hyperparams_pareto = [results['all_hyperparams'][i] for i in results['pareto_indices']]

# Display Pareto front
for i, (hp, y) in enumerate(zip(hyperparams_pareto, Y_pareto)):
    print(f"\nSolution {i+1}:")
    print(f"  LR: {hp['learning_rate']:.6f}, WD: {hp['weight_decay']:.6f}")
    print(f"  PR-AUC: {-y[0]:.4f}, AUROC: {-y[1]:.4f}")
    print(f"  Brier: {y[2]:.4f}, Robustness: {y[3]:.4f}")
```

### Load Evaluation History

```python
import pandas as pd

# Load all evaluations
df = pd.read_csv('results/botorch_mobo/production_001/evaluations.csv')

# Plot convergence
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(df['pr_auc'], label='PR-AUC')
plt.plot(df['auroc'], label='AUROC')
plt.xlabel('Evaluation')
plt.ylabel('Score')
plt.legend()
plt.title('Performance Metrics')

plt.subplot(1, 2, 2)
plt.plot(df['brier'], label='Brier Score')
plt.plot(df['robustness'], label='Robustness Degradation')
plt.xlabel('Evaluation')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Metrics')

plt.tight_layout()
plt.savefig('convergence.png')
```

## Troubleshooting

### Out of Memory (OOM) Errors

Reduce CNN batch size:
```bash
--train_batch_size 2
```

### qNEHVI Optimization Slow

Reduce acquisition optimization effort:
```bash
--acq_restarts 10
```

### GP Fitting Failures

Ensure at least 2 initial samples:
```bash
--n_initial 10  # Should be >= 2
```

## Validation Tests

Basic unit tests are automatically run during import:

```python
from src.botorch_utils import HyperparameterTransform
from src.botorch_mobo import MultiObjectiveGPModel, compute_reference_point

# All tests passed ✓
```

## Next Steps

1. **Run test optimization** (7 evaluations):
   ```bash
   python breast_cancer_detection/scripts/run_botorch_mobo.py \
       --data_root /path/to/vindr/images \
       --csv_file /path/to/vindr.csv \
       --n_initial 3 \
       --n_iterations 2 \
       --batch_size 2 \
       --max_epochs 5 \
       --run_id test_001
   ```

2. **Analyze results** using evaluation CSV and final results pickle

3. **Launch production run** (200 evaluations) if test is successful

4. **Zero-shot evaluation** on INbreast using Pareto solutions

## Support

For issues or questions, consult:
- **Plan file:** `.claude/plans/wondrous-discovering-bunny.md`
- **Code files:** `breast_cancer_detection/src/botorch_*.py`
- **BoTorch documentation:** https://botorch.org/

---

**Status:** Implementation complete, ready for testing ✅
**Test status:** Basic unit tests passed ✅
**Next action:** Run test optimization with real data
