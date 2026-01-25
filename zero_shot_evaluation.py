"""
Simple zero-shot evaluation on INbreast.
Run this after training your model.

This code assumes you have:
- model: trained PyTorch model
- device: torch.device
- val_dataset: VinDr-Mammo validation dataset
- preprocessor: MammographyPreprocessor instance
"""

import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, brier_score_loss

# ============================================================================
# STEP 1: SAVE THE TRAINED MODEL
# ============================================================================
print("="*80)
print("SAVING TRAINED MODEL")
print("="*80)

model_save_path = "trained_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"✓ Model saved to: {model_save_path}\n")

# ============================================================================
# STEP 2: LOAD INBREAST DATASET
# ============================================================================
print("="*80)
print("LOADING INBREAST DATASET")
print("="*80)

from src.datasets import INbreastDataset

# UPDATE THESE PATHS TO YOUR GOOGLE DRIVE LOCATIONS
INBREAST_DICOM_DIR = "/content/drive/MyDrive/INbreast/AllDICOMs"
INBREAST_CSV = "/content/drive/MyDrive/INbreast/INbreast.csv"

inbreast_dataset = INbreastDataset(
    dicom_dir=INBREAST_DICOM_DIR,
    csv_file=INBREAST_CSV,
    preprocessor=preprocessor,
    benign_birads=[2, 3],
    malignant_birads=[5, 6]
)

print(f"✓ INbreast loaded: {len(inbreast_dataset)} images")
print(f"✓ INbreast breasts: {len(inbreast_dataset.get_breast_groups())}\n")

# ============================================================================
# STEP 3: EVALUATE ON VINDR-MAMMO VALIDATION (SOURCE)
# ============================================================================
print("="*80)
print("EVALUATING ON VINDR-MAMMO VALIDATION (SOURCE DOMAIN)")
print("="*80)

from src.evaluation import aggregate_breast_level_predictions

y_true_vindr, y_probs_vindr = aggregate_breast_level_predictions(
    model, val_dataset, device
)

vindr_pr_auc = average_precision_score(y_true_vindr, y_probs_vindr)
vindr_auroc = roc_auc_score(y_true_vindr, y_probs_vindr)
vindr_brier = brier_score_loss(y_true_vindr, y_probs_vindr)

print(f"\nVinDr-Mammo Validation:")
print(f"  PR-AUC: {vindr_pr_auc:.4f}")
print(f"  AUROC:  {vindr_auroc:.4f}")
print(f"  Brier:  {vindr_brier:.4f}")

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y_true_vindr, y_probs_vindr)
youden = tpr - fpr
best_idx = np.argmax(youden)
optimal_threshold = thresholds[best_idx]

print(f"\nOptimal threshold: {optimal_threshold:.4f}")
print(f"  Sensitivity: {tpr[best_idx]:.4f}")
print(f"  Specificity: {1-fpr[best_idx]:.4f}\n")

# ============================================================================
# STEP 4: ZERO-SHOT EVALUATION ON INBREAST (TARGET)
# ============================================================================
print("="*80)
print("ZERO-SHOT EVALUATION ON INBREAST (TARGET DOMAIN)")
print("="*80)
print("✓ NO fine-tuning")
print("✓ NO threshold adjustment")
print("✓ Same preprocessing")
print("✓ Same Noisy-OR aggregation\n")

y_true_inbreast, y_probs_inbreast = aggregate_breast_level_predictions(
    model, inbreast_dataset, device
)

inbreast_pr_auc = average_precision_score(y_true_inbreast, y_probs_inbreast)
inbreast_auroc = roc_auc_score(y_true_inbreast, y_probs_inbreast)
inbreast_brier = brier_score_loss(y_true_inbreast, y_probs_inbreast)

# Apply threshold from VinDr-Mammo
y_pred_inbreast = (y_probs_inbreast >= optimal_threshold).astype(int)

# Calculate threshold-dependent metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score

tn, fp, fn, tp = confusion_matrix(y_true_inbreast, y_pred_inbreast).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = precision_score(y_true_inbreast, y_pred_inbreast, zero_division=0)
recall = recall_score(y_true_inbreast, y_pred_inbreast, zero_division=0)

print("="*80)
print("ZERO-SHOT RESULTS ON INBREAST")
print("="*80)

print(f"\nThreshold-Independent Metrics:")
print(f"  PR-AUC: {inbreast_pr_auc:.4f}")
print(f"  AUROC:  {inbreast_auroc:.4f}")
print(f"  Brier:  {inbreast_brier:.4f}")

print(f"\nThreshold-Dependent Metrics (threshold={optimal_threshold:.4f} from VinDr):")
print(f"  Sensitivity: {sensitivity:.4f}")
print(f"  Specificity: {specificity:.4f}")
print(f"  Precision:   {precision:.4f}")
print(f"  Recall:      {recall:.4f}")

print(f"\nConfusion Matrix:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")

# ============================================================================
# STEP 5: DOMAIN SHIFT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DOMAIN SHIFT ANALYSIS")
print("="*80)

print(f"\n{'Metric':<15} {'VinDr':<12} {'INbreast':<12} {'Delta':<10}")
print("-" * 50)
print(f"{'PR-AUC':<15} {vindr_pr_auc:<12.4f} {inbreast_pr_auc:<12.4f} {inbreast_pr_auc - vindr_pr_auc:<+10.4f}")
print(f"{'AUROC':<15} {vindr_auroc:<12.4f} {inbreast_auroc:<12.4f} {inbreast_auroc - vindr_auroc:<+10.4f}")
print(f"{'Brier':<15} {vindr_brier:<12.4f} {inbreast_brier:<12.4f} {inbreast_brier - vindr_brier:<+10.4f}")

performance_drop = vindr_auroc - inbreast_auroc
print(f"\nPerformance degradation: {performance_drop*100:.2f}%")

if performance_drop < 0.05:
    print("✓ EXCELLENT: Minimal domain shift (<5% drop)")
elif performance_drop < 0.10:
    print("✓ GOOD: Small domain shift (5-10% drop)")
elif performance_drop < 0.15:
    print("⚠ MODERATE: Noticeable domain shift (10-15% drop)")
else:
    print("⚠ SIGNIFICANT: Large domain shift (>15% drop)")

print("\n" + "="*80)
print("ZERO-SHOT EVALUATION COMPLETED!")
print("="*80)
