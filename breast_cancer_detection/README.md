# Breast Cancer Detection under Dataset Shift

**Multi-Objective Optimization for Mammography Classification with Zero-Shot Generalization**

This project implements a research-grade medical imaging study that optimizes a deep learning model for breast cancer detection across multiple objectives while evaluating zero-shot generalization to an unseen dataset.

---

## Overview

### Objectives
Optimize a ResNet152 model for mammography classification across 4 competing objectives:
1. **Maximize PR-AUC** (Precision-Recall Area Under Curve)
2. **Maximize AUROC** (Receiver Operating Characteristic Area Under Curve)
3. **Minimize Brier Score** (calibration error)
4. **Minimize Robustness Degradation** (performance drop under perturbations)

### Datasets
- **Source**: VinDr-Mammo (training and validation)
  - 80/20 patient-wise split
  - Binary classification: Benign (BI-RADS 1-3) vs Malignant (BI-RADS 5-6)
  - BI-RADS 4 excluded

- **Target**: INbreast (zero-shot evaluation only)
  - No fine-tuning
  - No threshold tuning
  - Same preprocessing pipeline

### Key Features
- **Multi-objective optimization** using NSGA-III
- **Continuous hyperparameter search space** (5 dimensions)
- **Breast-level aggregation** using Noisy-OR formula
- **Partial fine-tuning control** for transfer learning
- **Intensity augmentation controller** (no geometric transforms)
- **Robustness evaluation** under mild perturbations

---

## Project Structure

```
breast_cancer_detection/
├── src/
│   ├── preprocessing.py      # MammographyPreprocessor pipeline
│   ├── datasets.py            # VinDr-Mammo and INbreast datasets
│   ├── augmentations.py       # Intensity augmentation controller
│   ├── models.py              # ResNet152 with partial fine-tuning
│   ├── training.py            # Training loop with early stopping
│   ├── evaluation.py          # Metrics and Noisy-OR aggregation
│   ├── robustness.py          # Robustness degradation measurement
│   └── optimization.py        # pymoo problem definition
├── scripts/
│   ├── run_nsga3.py          # NSGA-III optimization runner
│   └── evaluate_zeroshot.py  # Zero-shot INbreast evaluation
├── configs/
│   └── config.py             # Configuration parameters
├── checkpoints/              # Model checkpoints (created at runtime)
├── logs/                     # Training and optimization logs
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository or navigate to project directory:
```bash
cd breast_cancer_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure data paths in `configs/config.py`:
```python
DATA_ROOT = Path("/path/to/your/data")
VINDR_IMAGES_ROOT = DATA_ROOT / "vindr-mammo" / "images"
VINDR_CSV = DATA_ROOT / "vindr-mammo" / "metadata" / "stratified_selection.csv"
INBREAST_DICOM_DIR = DATA_ROOT / "INbreast" / "AllDICOMs"
INBREAST_CSV = DATA_ROOT / "INbreast" / "INbreast.csv"
```

---

## Usage

### 1. Run NSGA-III Optimization

Optimize hyperparameters across 4 objectives:

```bash
python scripts/run_nsga3.py \
    --pop_size 20 \
    --n_gen 50 \
    --n_partitions 12 \
    --max_epochs 100 \
    --run_id "experiment_001"
```

**Parameters:**
- `--pop_size`: Population size for NSGA-III (default: 20)
- `--n_gen`: Number of generations (default: 50)
- `--n_partitions`: Reference direction partitions (default: 12)
- `--max_epochs`: Max training epochs per evaluation (default: 100)
- `--run_id`: Identifier for this run

**Outputs:**
- `logs/nsga3_run_{run_id}.csv`: Hyperparameters and objectives for each evaluation
- `checkpoints/nsga3_results_{run_id}.pkl`: Pareto front solutions

**Estimated Runtime:**
- Each evaluation trains a full ResNet152 (10-60 minutes depending on early stopping)
- Total: ~20-50 hours for 20 pop × 50 gen = 1000 evaluations

### 2. Zero-Shot Evaluation on INbreast

Evaluate Pareto solutions on INbreast dataset:

```bash
python scripts/evaluate_zeroshot.py \
    --results_file checkpoints/nsga3_results_experiment_001.pkl \
    --checkpoint_dir checkpoints/ \
    --run_id "experiment_001"
```

**Outputs:**
- `logs/zeroshot_evaluation_{run_id}.csv`: Detailed results for all Pareto solutions

---

## Hyperparameter Search Space

The optimization explores 5 continuous hyperparameters:

| Hyperparameter           | Range         | Scale      |
|-------------------------|---------------|------------|
| Learning Rate           | [1e-5, 1e-3]  | Log        |
| Weight Decay            | [1e-6, 1e-2]  | Log        |
| Dropout Rate            | [0.0, 0.5]    | Linear     |
| Augmentation Strength   | [0.0, 1.0]    | Linear     |
| Unfreeze Fraction       | [0.0, 1.0]    | Linear     |

**Augmentation Strength** controls intensity-based augmentations:
- Brightness adjustment: ±20% × strength
- Contrast scaling: ±30% × strength
- Gaussian noise: 5% × strength

**Unfreeze Fraction** controls partial fine-tuning:
- 0.0 = Freeze all backbone (train head only)
- 1.0 = Unfreeze all layers
- 0.5 = Unfreeze last 50% of layers

---

## Preprocessing Pipeline

**Fixed pipeline** (DO NOT MODIFY):

1. **DICOM Loading**: Apply rescale slope/intercept
2. **Orientation Normalization**: Flip to left-oriented
3. **Breast Extraction**: Largest connected component
4. **Inferior Fold Removal**: Row density analysis
5. **Nipple Suppression**: Convex hull + morphology
6. **Aspect-Safe Resize**: To 720×480
7. **Color Mapping**: Magma colormap → RGB

Implemented in `src/preprocessing.py`.

---

## Evaluation Methodology

### Breast-Level Aggregation

Each breast has **CC + MLO views**. Predictions are aggregated using **Noisy-OR**:

```
p_breast = 1 - (1 - p_CC) × (1 - p_MLO)
```

### Metrics Reported

**For both VinDr-Mammo validation and INbreast:**
- PR-AUC (Precision-Recall Area Under Curve)
- AUROC (ROC Area Under Curve)
- Brier Score (calibration)
- Sensitivity & Specificity at transferred threshold

**Threshold Selection:**
- Find threshold achieving 90% specificity on VinDr-Mammo validation
- Transfer **unchanged** to INbreast (no tuning!)

### Robustness Degradation

Measures performance drop under mild perturbations:

```
R = PR-AUC(clean) - PR-AUC(perturbed)
```

Perturbations (at inference):
- Brightness: ±10%
- Contrast: ±10%
- Gaussian noise: σ = 0.02

---

## Training Protocol

### Fixed Components
- **Optimizer**: AdamW
- **Loss**: Binary Cross-Entropy with Logits
- **Batch Size**: 4
- **Early Stopping**: Monitor validation PR-AUC with patience=10

### Class Imbalance Handling
- Compute `pos_weight = n_benign / n_malignant`
- Apply to BCEWithLogitsLoss

### Data Split
- **Single fixed split** (random_state=42)
- 80% train, 20% validation
- Stratified by label
- **Same split for ALL experiments**

---

## Extending the Code

### Adding New Augmentations

Edit `src/augmentations.py`:
```python
# In IntensityAugmentation.forward()
# Add new intensity-based transform (NO geometric transforms!)
```

### Changing Model Architecture

Edit `src/models.py`:
```python
def build_custom_model(...):
    # Implement alternative architecture
    # Ensure single output node for binary classification
    pass
```

### Adding New Objectives

1. Add objective computation in `src/training.py`
2. Update `BreastCancerOptimizationProblem` in `src/optimization.py`
3. Increment `N_OBJECTIVES` in `configs/config.py`

---

## Citation

If you use this code, please cite:

```bibtex
@software{breast_cancer_nsga3,
  title={Breast Cancer Detection under Dataset Shift: Multi-Objective Optimization},
  author={Your Name},
  year={2025},
  note={Research implementation for mammography classification}
}
```

---

## License

This project is for research purposes only. Medical imaging datasets require appropriate institutional review and permissions.

---

## Acknowledgments

- **VinDr-Mammo**: Nguyen et al., "VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography"
- **INbreast**: Moreira et al., "INbreast: Toward a Full-field Digital Mammographic Database"
- **pymoo**: Blank & Deb, "pymoo: Multi-Objective Optimization in Python"

---

## Contact

For questions or issues, please open an issue in the repository.
