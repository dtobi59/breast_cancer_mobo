"""
Configuration file for breast cancer detection experiment
"""

import os
from pathlib import Path

# ===== Paths =====
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(r"C:\Users\HP\Downloads\New Project\data")  # Adjust to your data location

# VinDr-Mammo paths
VINDR_IMAGES_ROOT = DATA_ROOT / "vindr-mammo" / "images"
VINDR_CSV = DATA_ROOT / "vindr-mammo" / "metadata" / "stratified_selection.csv"

# INbreast paths
INBREAST_DICOM_DIR = DATA_ROOT / "INbreast" / "AllDICOMs"
INBREAST_CSV = DATA_ROOT / "INbreast" / "INbreast.csv"

# Output paths
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# ===== Data Split =====
TRAIN_VAL_SPLIT = 0.2  # 80% train, 20% validation
RANDOM_SEED = 42  # Fixed seed for reproducibility

# ===== BI-RADS Mapping =====
BIRADS_BENIGN = [1, 2, 3]  # BI-RADS 1-3 → benign
BIRADS_MALIGNANT = [5, 6]  # BI-RADS 5-6 → malignant
BIRADS_EXCLUDE = [4]       # Exclude BI-RADS 4

# ===== Preprocessing =====
TARGET_SIZE = (720, 480)  # (width, height)
ASPECT_RATIO = 1.5        # height/width must be <= 1.5

# ===== Model =====
MODEL_ARCHITECTURE = "resnet152"
PRETRAINED = True
NUM_CLASSES = 1  # Binary classification

# ===== Training (Fixed) =====
OPTIMIZER = "adamw"
LOSS_FUNCTION = "bce_with_logits"
BATCH_SIZE = 4
NUM_WORKERS = 2

# ===== Hyperparameter Search Space =====
# Continuous hyperparameters for NSGA-III
HYPERPARAMETER_BOUNDS = {
    "learning_rate": (1e-5, 1e-3),      # log-scale
    "weight_decay": (1e-6, 1e-2),       # log-scale
    "dropout": (0.0, 0.5),              # linear
    "augmentation_strength": (0.0, 1.0), # linear
    "unfreeze_fraction": (0.0, 1.0)     # linear, controls how many layers to fine-tune
}

# ===== Early Stopping =====
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_METRIC = "val_pr_auc"  # Monitor PR-AUC for early stopping
EARLY_STOPPING_MODE = "max"           # Maximize PR-AUC

# ===== NSGA-III Settings =====
NSGA3_POPULATION_SIZE = 20
NSGA3_N_GENERATIONS = 50
N_OBJECTIVES = 4  # PR-AUC, AUROC, Brier, Robustness degradation

# ===== Robustness Evaluation =====
ROBUSTNESS_PERTURBATIONS = {
    "brightness_delta": 0.1,    # +/- 10% brightness
    "contrast_factor": 0.1,     # +/- 10% contrast
    "noise_std": 0.02          # Gaussian noise std
}

# ===== Breast-level Aggregation =====
# Views per breast: CC + MLO
VIEWS_PER_BREAST = ["CC", "MLO"]

# ===== Evaluation Thresholds =====
TARGET_SPECIFICITY = 0.90  # For threshold selection
