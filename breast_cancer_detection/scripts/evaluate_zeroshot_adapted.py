"""
Zero-shot evaluation on INbreast with entropy-based domain adaptation.

Compares baseline (no adaptation) vs adapted zero-shot performance.

Scenarios evaluated:
1. Baseline: No adaptation on VinDr val or INbreast
2. INbreast Adapted: Adaptation on INbreast only
3. Both Adapted: Adaptation on both VinDr val and INbreast

Usage:
    python scripts/evaluate_zeroshot_adapted.py \\
        --model_path checkpoints/trained_model.pth \\
        --dropout 0.2 \\
        --unfreeze_frac 0.5
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score,
    brier_score_loss
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import MammographyPreprocessor
from src.datasets import (
    VinDRMammoBinaryDataset, INbreastDataset,
    create_breast_level_splits,
    create_vindr_dataset_with_adaptation,
    create_inbreast_dataset_with_adaptation
)
from src.models import build_resnet152
from src.evaluation import aggregate_breast_level_predictions, compute_metrics
from src.cache_manager import ensure_entropy_cache
from configs.config import (
    VINDR_IMAGES_ROOT, VINDR_CSV,
    INBREAST_DICOM_DIR, INBREAST_CSV,
    RANDOM_SEED, TRAIN_VAL_SPLIT
)


def load_model(model_path, dropout, unfreeze_fraction, device):
    """Load trained model from checkpoint."""
    print(f"\nLoading model from: {model_path}")

    model = build_resnet152(
        pretrained=False,
        dropout=dropout,
        unfreeze_fraction=unfreeze_fraction
    )

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    return model


def evaluate_scenario(model, val_dataset, inb_dataset, device, scenario_name):
    """Evaluate one scenario."""
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}")

    # Evaluate on VinDr validation
    print("\nEvaluating on VinDr-Mammo validation...")
    y_true_val, y_probs_val = aggregate_breast_level_predictions(
        model, val_dataset, device
    )

    val_metrics = {
        'pr_auc': average_precision_score(y_true_val, y_probs_val),
        'auroc': roc_auc_score(y_true_val, y_probs_val),
        'brier': brier_score_loss(y_true_val, y_probs_val),
        'prob_spread': y_probs_val.max() - y_probs_val.min()
    }

    # Evaluate on INbreast
    print("Evaluating on INbreast (zero-shot)...")
    y_true_inb, y_probs_inb = aggregate_breast_level_predictions(
        model, inb_dataset, device
    )

    inb_metrics = {
        'pr_auc': average_precision_score(y_true_inb, y_probs_inb),
        'auroc': roc_auc_score(y_true_inb, y_probs_inb),
        'brier': brier_score_loss(y_true_inb, y_probs_inb),
        'prob_spread': y_probs_inb.max() - y_probs_inb.min()
    }

    # Print results
    print(f"\nVinDr-Mammo Validation:")
    print(f"  PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"  AUROC:  {val_metrics['auroc']:.4f}")
    print(f"  Brier:  {val_metrics['brier']:.4f}")
    print(f"  Prob Spread: {val_metrics['prob_spread']:.4f}")

    print(f"\nINbreast (Zero-Shot):")
    print(f"  PR-AUC: {inb_metrics['pr_auc']:.4f}")
    print(f"  AUROC:  {inb_metrics['auroc']:.4f}")
    print(f"  Brier:  {inb_metrics['brier']:.4f}")
    print(f"  Prob Spread: {inb_metrics['prob_spread']:.4f}")

    return val_metrics, inb_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation with entropy adaptation"
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

    # Entropy cache
    parser.add_argument(
        "--entropy_cache",
        type=str,
        default="cache/entropy_stats_vindr_train.json",
        help="Path to entropy statistics cache"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(
        args.model_path,
        args.dropout,
        args.unfreeze_frac,
        device
    )

    # Ensure entropy cache exists
    print("\nChecking entropy cache...")
    try:
        ensure_entropy_cache("vindr_train", "cache")
        print(f"✓ Entropy cache found")
    except FileNotFoundError as e:
        print(e)
        return 1

    # Create datasets for all scenarios
    print("\n" + "="*80)
    print("CREATING DATASETS")
    print("="*80)

    # Load full VinDr dataset
    print("\nLoading VinDr-Mammo...")
    full_vindr = VinDRMammoBinaryDataset(
        images_root=VINDR_IMAGES_ROOT,
        csv_file=VINDR_CSV,
        preprocessor=MammographyPreprocessor(),
        transform=None
    )

    # Create validation split
    _, val_subset = create_breast_level_splits(
        dataset=full_vindr,
        train_ratio=1.0 - TRAIN_VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=True
    )

    # Scenario 1: Baseline (no adaptation)
    print("\n[Scenario 1] Creating baseline datasets (no adaptation)...")
    val_baseline = val_subset
    inb_baseline = INbreastDataset(
        dicom_dir=INBREAST_DICOM_DIR,
        csv_file=INBREAST_CSV,
        preprocessor=MammographyPreprocessor()
    )

    # Scenario 2: INbreast adapted only
    print("[Scenario 2] Creating INbreast with adaptation...")
    inb_adapted = create_inbreast_dataset_with_adaptation(
        dicom_dir=INBREAST_DICOM_DIR,
        csv_file=INBREAST_CSV,
        entropy_stats_path=args.entropy_cache
    )

    # Scenario 3: Both adapted (experimental per user requirement)
    print("[Scenario 3] Creating both VinDr val and INbreast with adaptation...")

    # For adapted VinDr validation, we need to create a new dataset
    # with adaptation, then get the validation subset
    full_vindr_adapted = create_vindr_dataset_with_adaptation(
        images_root=VINDR_IMAGES_ROOT,
        csv_file=VINDR_CSV,
        split='val',
        entropy_stats_path=args.entropy_cache,
        apply_adaptation=True,
        transform=None
    )

    # Get same validation indices
    from torch.utils.data import Subset
    val_adapted = Subset(full_vindr_adapted, val_subset.indices)

    # Run evaluations
    print("\n" + "="*80)
    print("RUNNING EVALUATIONS")
    print("="*80)

    results = {}

    # Scenario 1: Baseline
    val_m1, inb_m1 = evaluate_scenario(
        model, val_baseline, inb_baseline, device,
        "Baseline (No Adaptation)"
    )
    results['baseline'] = {'val': val_m1, 'inb': inb_m1}

    # Scenario 2: INbreast adapted
    val_m2, inb_m2 = evaluate_scenario(
        model, val_baseline, inb_adapted, device,
        "INbreast Adapted Only"
    )
    results['inb_adapted'] = {'val': val_m2, 'inb': inb_m2}

    # Scenario 3: Both adapted
    val_m3, inb_m3 = evaluate_scenario(
        model, val_adapted, inb_adapted, device,
        "Both VinDr Val + INbreast Adapted"
    )
    results['both_adapted'] = {'val': val_m3, 'inb': inb_m3}

    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    print(f"\n{'Scenario':<30} {'VinDr AUROC':<15} {'INb AUROC':<15} {'INb Spread':<15}")
    print("-" * 75)

    for scenario_name, scenario_data in [
        ("Baseline", results['baseline']),
        ("INbreast Adapted", results['inb_adapted']),
        ("Both Adapted", results['both_adapted'])
    ]:
        print(f"{scenario_name:<30} "
              f"{scenario_data['val']['auroc']:<15.4f} "
              f"{scenario_data['inb']['auroc']:<15.4f} "
              f"{scenario_data['inb']['prob_spread']:<15.4f}")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    baseline_auroc = results['baseline']['inb']['auroc']
    inb_adapted_auroc = results['inb_adapted']['inb']['auroc']
    improvement = inb_adapted_auroc - baseline_auroc

    print(f"\nINbreast AUROC Improvement (Baseline → INb Adapted):")
    print(f"  {baseline_auroc:.4f} → {inb_adapted_auroc:.4f} ({improvement:+.4f})")

    baseline_spread = results['baseline']['inb']['prob_spread']
    adapted_spread = results['inb_adapted']['inb']['prob_spread']

    print(f"\nINbreast Prediction Spread:")
    print(f"  Baseline: {baseline_spread:.4f} ({baseline_spread*100:.1f}% of [0,1])")
    print(f"  Adapted:  {adapted_spread:.4f} ({adapted_spread*100:.1f}% of [0,1])")

    # Success assessment
    print(f"\n{'='*80}")
    if improvement > 0.10 and adapted_spread > 0.50:
        print("✓ EXCELLENT: Significant improvement! Domain adaptation successful.")
    elif improvement > 0.05 and adapted_spread > 0.30:
        print("✓ GOOD: Moderate improvement. Domain adaptation helped.")
    elif improvement > 0.00:
        print("⚠ PARTIAL: Minor improvement. May need tuning or alternative methods.")
    else:
        print("✗ UNSUCCESSFUL: No improvement. Consider alternative adaptation strategies.")

    print(f"{'='*80}")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        import pandas as pd
        import json

        # Save detailed results
        results_file = os.path.join(args.output_dir, "zeroshot_adapted_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for scenario, data in results.items():
                json_results[scenario] = {
                    'val': {k: float(v) for k, v in data['val'].items()},
                    'inb': {k: float(v) for k, v in data['inb'].items()}
                }
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    exit(main())
