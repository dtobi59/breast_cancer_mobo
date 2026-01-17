"""
Zero-shot evaluation on INbreast dataset.

Evaluates Pareto solutions from NSGA-III on INbreast:
- No retraining
- No threshold tuning
- Same preprocessing
- Same Noisy-OR aggregation
- Decision thresholds transferred from VinDr-Mammo validation
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import pickle
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import MammographyPreprocessor
from src.datasets import VinDRMammoBinaryDataset, INbreastDataset
from src.models import build_resnet152
from src.evaluation import (
    aggregate_breast_level_predictions,
    find_threshold_at_specificity,
    evaluate_with_threshold
)
from configs.config import (
    VINDR_IMAGES_ROOT,
    VINDR_CSV,
    INBREAST_DICOM_DIR,
    INBREAST_CSV,
    RANDOM_SEED,
    TRAIN_VAL_SPLIT,
    TARGET_SPECIFICITY,
    LOG_DIR
)
from sklearn.model_selection import train_test_split


def load_pareto_solutions(result_file):
    """
    Load Pareto solutions from optimization results.

    Args:
        result_file: Path to NSGA-III results pickle file

    Returns:
        Dictionary with X (hyperparameters) and F (objectives)
    """
    with open(result_file, "rb") as f:
        results = pickle.load(f)

    return results["X"], results["F"]


def setup_datasets():
    """
    Setup VinDr-Mammo validation and INbreast datasets.

    Returns:
        val_dataset, inbreast_dataset
    """
    print("Setting up datasets...")

    # Create preprocessor
    preprocessor = MammographyPreprocessor()

    # VinDr-Mammo validation dataset
    full_dataset = VinDRMammoBinaryDataset(
        images_root=VINDR_IMAGES_ROOT,
        csv_file=VINDR_CSV,
        preprocessor=preprocessor
    )

    # Get validation split (same as training)
    labels = [full_dataset.samples[i][-1] for i in range(len(full_dataset))]
    indices = np.arange(len(full_dataset))
    _, val_idx = train_test_split(
        indices,
        test_size=TRAIN_VAL_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    from torch.utils.data import Subset
    val_dataset = Subset(full_dataset, val_idx)

    # INbreast dataset
    inbreast_dataset = INbreastDataset(
        dicom_dir=INBREAST_DICOM_DIR,
        csv_file=INBREAST_CSV,
        preprocessor=preprocessor
    )

    print(f"VinDr-Mammo validation samples: {len(val_dataset)}")
    print(f"INbreast samples: {len(inbreast_dataset)}")

    return val_dataset, inbreast_dataset


def evaluate_solution(model, val_dataset, inbreast_dataset, device, solution_id):
    """
    Evaluate a single Pareto solution on both datasets.

    Args:
        model: Trained model
        val_dataset: VinDr-Mammo validation dataset
        inbreast_dataset: INbreast dataset
        device: torch.device
        solution_id: Solution identifier

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    results = {"solution_id": solution_id}

    # ===== 1. Validation set evaluation =====
    print(f"  Evaluating on VinDr-Mammo validation...")

    # Get breast-level predictions
    val_y_true, val_y_probs = aggregate_breast_level_predictions(
        model, val_dataset, device
    )

    # Find threshold at target specificity
    threshold = find_threshold_at_specificity(
        val_y_true, val_y_probs, target_specificity=TARGET_SPECIFICITY
    )

    if threshold is None:
        print(f"  Warning: Could not find threshold at {TARGET_SPECIFICITY} specificity")
        threshold = 0.5

    # Compute metrics
    val_metrics = evaluate_with_threshold(val_y_true, val_y_probs, threshold)

    results["val_threshold"] = threshold
    results["val_pr_auc"] = val_metrics["pr_auc"]
    results["val_auroc"] = val_metrics["auroc"]
    results["val_brier"] = val_metrics["brier"]
    results["val_sensitivity"] = val_metrics["sensitivity"]
    results["val_specificity"] = val_metrics["specificity"]

    print(f"    PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"    AUROC: {val_metrics['auroc']:.4f}")
    print(f"    Brier: {val_metrics['brier']:.4f}")
    print(f"    Threshold: {threshold:.4f}")
    print(f"    Sensitivity: {val_metrics['sensitivity']:.4f}")
    print(f"    Specificity: {val_metrics['specificity']:.4f}")

    # ===== 2. Zero-shot INbreast evaluation =====
    print(f"  Evaluating on INbreast (zero-shot)...")

    # Get breast-level predictions
    inb_y_true, inb_y_probs = aggregate_breast_level_predictions(
        model, inbreast_dataset, device
    )

    # Use TRANSFERRED threshold (no tuning!)
    inb_metrics = evaluate_with_threshold(inb_y_true, inb_y_probs, threshold)

    results["inb_pr_auc"] = inb_metrics["pr_auc"]
    results["inb_auroc"] = inb_metrics["auroc"]
    results["inb_brier"] = inb_metrics["brier"]
    results["inb_sensitivity"] = inb_metrics["sensitivity"]
    results["inb_specificity"] = inb_metrics["specificity"]

    print(f"    PR-AUC: {inb_metrics['pr_auc']:.4f}")
    print(f"    AUROC: {inb_metrics['auroc']:.4f}")
    print(f"    Brier: {inb_metrics['brier']:.4f}")
    print(f"    Sensitivity: {inb_metrics['sensitivity']:.4f}")
    print(f"    Specificity: {inb_metrics['specificity']:.4f}")

    return results


def evaluate_all_solutions(args):
    """
    Evaluate all Pareto solutions on INbreast.

    Args:
        args: Command line arguments
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Pareto solutions
    print(f"Loading Pareto solutions from: {args.results_file}")
    X, F = load_pareto_solutions(args.results_file)
    print(f"Found {len(X)} Pareto solutions")

    # Setup datasets
    val_dataset, inbreast_dataset = setup_datasets()

    # Decode hyperparameters helper
    def decode_hyperparameters(x):
        lr_log, wd_log, dropout, aug_strength, unfreeze_frac = x
        return {
            "learning_rate": 10 ** lr_log,
            "weight_decay": 10 ** wd_log,
            "dropout": dropout,
            "augmentation_strength": aug_strength,
            "unfreeze_fraction": unfreeze_frac
        }

    # Evaluate each solution
    all_results = []

    for i, (x, f) in enumerate(zip(X, F)):
        print(f"\n[Solution {i+1}/{len(X)}]")

        # Decode hyperparameters
        hyperparams = decode_hyperparameters(x)

        print(f"  Hyperparameters:")
        print(f"    LR: {hyperparams['learning_rate']:.6f}")
        print(f"    WD: {hyperparams['weight_decay']:.6f}")
        print(f"    Dropout: {hyperparams['dropout']:.4f}")
        print(f"    Aug Strength: {hyperparams['augmentation_strength']:.4f}")
        print(f"    Unfreeze Frac: {hyperparams['unfreeze_fraction']:.4f}")

        # Build model with these hyperparameters
        model = build_resnet152(
            pretrained=True,
            dropout=hyperparams["dropout"],
            unfreeze_fraction=hyperparams["unfreeze_fraction"]
        )
        model = model.to(device)

        # Load checkpoint if available
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"solution_{i}_model.pth"
        )

        if os.path.exists(checkpoint_path):
            print(f"  Loading checkpoint from: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"  WARNING: No checkpoint found at {checkpoint_path}")
            print(f"  Skipping solution {i}")
            continue

        # Evaluate
        results = evaluate_solution(
            model, val_dataset, inbreast_dataset, device, i
        )

        # Add hyperparameters and objectives
        results.update({
            "lr": hyperparams["learning_rate"],
            "wd": hyperparams["weight_decay"],
            "dropout": hyperparams["dropout"],
            "aug_strength": hyperparams["augmentation_strength"],
            "unfreeze_frac": hyperparams["unfreeze_fraction"],
            "obj_neg_pr_auc": f[0],
            "obj_neg_auroc": f[1],
            "obj_brier": f[2],
            "obj_robustness": f[3]
        })

        all_results.append(results)

    # Save results to CSV
    os.makedirs(LOG_DIR, exist_ok=True)
    output_file = os.path.join(LOG_DIR, f"zeroshot_evaluation_{args.run_id}.csv")

    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to: {output_file}")
    print(f"{'='*80}")

    # Print summary
    print("\nSummary Statistics:")
    print(f"Average INbreast PR-AUC: {df['inb_pr_auc'].mean():.4f} ± {df['inb_pr_auc'].std():.4f}")
    print(f"Average INbreast AUROC: {df['inb_auroc'].mean():.4f} ± {df['inb_auroc'].std():.4f}")
    print(f"Best INbreast PR-AUC: {df['inb_pr_auc'].max():.4f} (Solution {df['inb_pr_auc'].idxmax()})")
    print(f"Best INbreast AUROC: {df['inb_auroc'].max():.4f} (Solution {df['inb_auroc'].idxmax()})")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on INbreast")

    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to NSGA-III results pickle file")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing model checkpoints")
    parser.add_argument("--run_id", type=str, default="001",
                       help="Run identifier for logging")

    args = parser.parse_args()

    # Evaluate all solutions
    evaluate_all_solutions(args)


if __name__ == "__main__":
    main()
