"""
Zero-shot evaluation script for a single trained model on INbreast.

Usage:
    python scripts/evaluate_zeroshot_single_model.py \
        --model_path checkpoints/model.pth \
        --dropout 0.2 \
        --unfreeze_frac 0.5

This script:
1. Loads a trained model from VinDr-Mammo
2. Evaluates on VinDr-Mammo validation to determine threshold
3. Performs zero-shot evaluation on INbreast (NO fine-tuning)
4. Reports metrics and domain shift analysis
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score,
    brier_score_loss, precision_recall_curve
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import MammographyPreprocessor
from src.datasets import VinDRMammoBinaryDataset, INbreastDataset, create_breast_level_splits
from src.models import build_resnet152
from src.evaluation import aggregate_breast_level_predictions, compute_metrics
from configs.config import (
    VINDR_IMAGES_ROOT,
    VINDR_CSV,
    INBREAST_DICOM_DIR,
    INBREAST_CSV,
    RANDOM_SEED,
    TRAIN_VAL_SPLIT
)


def load_trained_model(model_path, dropout, unfreeze_fraction, device):
    """Load a trained model from checkpoint."""
    print(f"\nLoading model from: {model_path}")

    # Build model architecture
    model = build_resnet152(
        pretrained=False,  # Don't load ImageNet weights, we have trained weights
        dropout=dropout,
        unfreeze_fraction=unfreeze_fraction
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    return model


def evaluate_zeroshot(args):
    """Main zero-shot evaluation function."""

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_trained_model(
        args.model_path,
        args.dropout,
        args.unfreeze_frac,
        device
    )

    # Setup preprocessor
    print("\nSetting up datasets...")
    preprocessor = MammographyPreprocessor()

    # Load VinDr-Mammo dataset and create validation split
    print("Loading VinDr-Mammo dataset...")
    vindr_dataset = VinDRMammoBinaryDataset(
        images_root=VINDR_IMAGES_ROOT,
        csv_file=VINDR_CSV,
        preprocessor=preprocessor,
        transform=None
    )

    # Create breast-level splits
    _, val_dataset = create_breast_level_splits(
        dataset=vindr_dataset,
        train_ratio=1.0 - TRAIN_VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=True
    )

    print(f"✓ VinDr-Mammo validation: {len(val_dataset)} images")

    # Load INbreast dataset
    print("Loading INbreast dataset...")
    inbreast_dataset = INbreastDataset(
        dicom_dir=INBREAST_DICOM_DIR,
        csv_file=INBREAST_CSV,
        preprocessor=preprocessor,
        benign_birads=[2, 3],
        malignant_birads=[5, 6]
    )

    print(f"✓ INbreast: {len(inbreast_dataset)} images")
    print(f"✓ INbreast breasts: {len(inbreast_dataset.get_breast_groups())}")

    print("\n" + "="*80)
    print("EVALUATING ON SOURCE DOMAIN (VinDr-Mammo)")
    print("="*80)

    # Evaluate on VinDr-Mammo validation
    y_true_vindr, y_probs_vindr = aggregate_breast_level_predictions(
        model, val_dataset, device
    )

    vindr_metrics = {
        'pr_auc': average_precision_score(y_true_vindr, y_probs_vindr),
        'auroc': roc_auc_score(y_true_vindr, y_probs_vindr),
        'brier': brier_score_loss(y_true_vindr, y_probs_vindr)
    }

    print(f"\nVinDr-Mammo Validation Performance:")
    print(f"  PR-AUC: {vindr_metrics['pr_auc']:.4f}")
    print(f"  AUROC:  {vindr_metrics['auroc']:.4f}")
    print(f"  Brier:  {vindr_metrics['brier']:.4f}")

    # Determine optimal threshold using Youden's index
    fpr, tpr, thresholds = roc_curve(y_true_vindr, y_probs_vindr)
    youden_index = tpr - fpr
    best_threshold_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[best_threshold_idx]

    print(f"\nOptimal threshold (Youden's index): {optimal_threshold:.4f}")
    print(f"  Sensitivity: {tpr[best_threshold_idx]:.4f}")
    print(f"  Specificity: {1 - fpr[best_threshold_idx]:.4f}")

    print("\n" + "="*80)
    print("ZERO-SHOT EVALUATION ON TARGET DOMAIN (INbreast)")
    print("="*80)
    print("  ✓ NO fine-tuning")
    print("  ✓ NO threshold adjustment")
    print("  ✓ Same preprocessing")
    print("  ✓ Same Noisy-OR aggregation")

    # Zero-shot evaluation on INbreast
    y_true_inbreast, y_probs_inbreast = aggregate_breast_level_predictions(
        model, inbreast_dataset, device
    )

    # Compute metrics with transferred threshold
    inbreast_metrics = compute_metrics(
        y_true_inbreast,
        y_probs_inbreast,
        threshold=optimal_threshold
    )

    print("\n" + "="*80)
    print("ZERO-SHOT RESULTS")
    print("="*80)

    print("\nThreshold-Independent Metrics:")
    print(f"  PR-AUC: {inbreast_metrics['pr_auc']:.4f}")
    print(f"  AUROC:  {inbreast_metrics['auroc']:.4f}")
    print(f"  Brier:  {inbreast_metrics['brier']:.4f}")

    print(f"\nThreshold-Dependent Metrics (threshold={optimal_threshold:.4f}):")
    print(f"  Sensitivity: {inbreast_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {inbreast_metrics['specificity']:.4f}")
    print(f"  Precision:   {inbreast_metrics['precision']:.4f}")
    print(f"  Recall:      {inbreast_metrics['recall']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN: {inbreast_metrics['tn']:<4} FP: {inbreast_metrics['fp']}")
    print(f"  FN: {inbreast_metrics['fn']:<4} TP: {inbreast_metrics['tp']}")

    # Domain shift analysis
    print("\n" + "="*80)
    print("DOMAIN SHIFT ANALYSIS")
    print("="*80)

    print(f"\n{'Metric':<20} {'VinDr-Mammo':<15} {'INbreast':<15} {'Δ':<10}")
    print("-" * 60)
    print(f"{'PR-AUC':<20} {vindr_metrics['pr_auc']:<15.4f} {inbreast_metrics['pr_auc']:<15.4f} {inbreast_metrics['pr_auc'] - vindr_metrics['pr_auc']:<10.4f}")
    print(f"{'AUROC':<20} {vindr_metrics['auroc']:<15.4f} {inbreast_metrics['auroc']:<15.4f} {inbreast_metrics['auroc'] - vindr_metrics['auroc']:<10.4f}")
    print(f"{'Brier Score':<20} {vindr_metrics['brier']:<15.4f} {inbreast_metrics['brier']:<15.4f} {inbreast_metrics['brier'] - vindr_metrics['brier']:<10.4f}")

    performance_drop = vindr_metrics['auroc'] - inbreast_metrics['auroc']
    print(f"\nPerformance degradation: {performance_drop*100:.2f}%")

    if performance_drop < 0.05:
        print("✓ Minimal domain shift impact")
    elif performance_drop < 0.10:
        print("✓ Moderate domain shift impact")
    else:
        print("⚠ Significant domain shift impact")

    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_results(
            y_true_vindr, y_probs_vindr,
            y_true_inbreast, y_probs_inbreast,
            vindr_metrics, inbreast_metrics,
            args.output_dir
        )

    # Save results
    if args.output_dir:
        save_results(
            vindr_metrics, inbreast_metrics,
            optimal_threshold,
            args.output_dir,
            args.run_id
        )

    print("\n✓ Zero-shot evaluation completed!")
    print("="*80)


def visualize_results(y_true_vindr, y_probs_vindr, y_true_inbreast, y_probs_inbreast,
                     vindr_metrics, inbreast_metrics, output_dir):
    """Generate and save visualization plots."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ROC Curves
    fpr_vindr, tpr_vindr, _ = roc_curve(y_true_vindr, y_probs_vindr)
    fpr_inbreast, tpr_inbreast, _ = roc_curve(y_true_inbreast, y_probs_inbreast)

    axes[0].plot(fpr_vindr, tpr_vindr, 'b-', linewidth=2,
                 label=f'VinDr-Mammo (AUC={vindr_metrics["auroc"]:.3f})')
    axes[0].plot(fpr_inbreast, tpr_inbreast, 'r-', linewidth=2,
                 label=f'INbreast Zero-Shot (AUC={inbreast_metrics["auroc"]:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curves: Source vs Target Domain', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curves
    precision_vindr, recall_vindr, _ = precision_recall_curve(y_true_vindr, y_probs_vindr)
    precision_inbreast, recall_inbreast, _ = precision_recall_curve(y_true_inbreast, y_probs_inbreast)

    axes[1].plot(recall_vindr, precision_vindr, 'b-', linewidth=2,
                 label=f'VinDr-Mammo (AP={vindr_metrics["pr_auc"]:.3f})')
    axes[1].plot(recall_inbreast, precision_inbreast, 'r-', linewidth=2,
                 label=f'INbreast Zero-Shot (AP={inbreast_metrics["pr_auc"]:.3f})')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curves: Source vs Target Domain', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'zeroshot_evaluation_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to: {plot_path}")

    plt.show()


def save_results(vindr_metrics, inbreast_metrics, threshold, output_dir, run_id):
    """Save results to CSV file."""
    import csv

    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'zeroshot_results_{run_id}.csv')

    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'PR-AUC', 'AUROC', 'Brier', 'Threshold',
                        'Sensitivity', 'Specificity', 'Precision', 'Recall'])

        # VinDr-Mammo row
        writer.writerow([
            'VinDr-Mammo',
            f"{vindr_metrics['pr_auc']:.4f}",
            f"{vindr_metrics['auroc']:.4f}",
            f"{vindr_metrics['brier']:.4f}",
            f"{threshold:.4f}",
            '-', '-', '-', '-'
        ])

        # INbreast row
        writer.writerow([
            'INbreast',
            f"{inbreast_metrics['pr_auc']:.4f}",
            f"{inbreast_metrics['auroc']:.4f}",
            f"{inbreast_metrics['brier']:.4f}",
            f"{threshold:.4f}",
            f"{inbreast_metrics['sensitivity']:.4f}",
            f"{inbreast_metrics['specificity']:.4f}",
            f"{inbreast_metrics['precision']:.4f}",
            f"{inbreast_metrics['recall']:.4f}"
        ])

    print(f"✓ Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation on INbreast"
    )

    # Model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate used during training"
    )
    parser.add_argument(
        "--unfreeze_frac",
        type=float,
        default=0.5,
        help="Unfreeze fraction used during training"
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="001",
        help="Run identifier for output files"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save visualization plots"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluate_zeroshot(args)


if __name__ == "__main__":
    main()
