"""
Example script to train a single model with fixed hyperparameters.

This demonstrates the basic training pipeline without running
the full NSGA-III optimization.
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import MammographyPreprocessor
from src.datasets import VinDRMammoBinaryDataset
from src.augmentations import get_augmentation
from src.models import build_resnet152
from src.training import train_model
from src.evaluation import aggregate_breast_level_predictions, compute_metrics
from configs.config import (
    VINDR_IMAGES_ROOT,
    VINDR_CSV,
    RANDOM_SEED,
    TRAIN_VAL_SPLIT,
    BATCH_SIZE,
    NUM_WORKERS,
    EARLY_STOPPING_PATIENCE,
    CHECKPOINT_DIR
)


def main():
    parser = argparse.ArgumentParser(description="Train a single model")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--aug_strength", type=float, default=0.5, help="Augmentation strength")
    parser.add_argument("--unfreeze_frac", type=float, default=1.0, help="Fraction of layers to unfreeze")

    # Training settings
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")

    # Other
    parser.add_argument("--save_model", action="store_true", help="Save trained model")
    parser.add_argument("--model_name", type=str, default="single_model.pth", help="Model filename")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup datasets
    print("\nSetting up datasets...")
    preprocessor = MammographyPreprocessor()

    full_dataset = VinDRMammoBinaryDataset(
        images_root=VINDR_IMAGES_ROOT,
        csv_file=VINDR_CSV,
        preprocessor=preprocessor
    )

    # Split dataset
    labels = [full_dataset.samples[i][-1] for i in range(len(full_dataset))]
    indices = np.arange(len(full_dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=TRAIN_VAL_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Compute pos_weight
    train_labels = [labels[i] for i in train_idx]
    n_benign = sum(1 for l in train_labels if l == 0)
    n_malignant = sum(1 for l in train_labels if l == 1)
    pos_weight = n_benign / n_malignant

    print(f"Train: {len(train_dataset)} samples (Benign: {n_benign}, Malignant: {n_malignant})")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Positive class weight: {pos_weight:.3f}")

    # Setup augmentation
    augmentation = get_augmentation(args.aug_strength)
    full_dataset.transform = augmentation

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Build model
    print("\nBuilding model...")
    model = build_resnet152(
        pretrained=True,
        dropout=args.dropout,
        unfreeze_fraction=args.unfreeze_frac
    )
    model = model.to(device)

    # Print model info
    model_info = model.get_trainable_params_info()
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,} ({model_info['trainable_percentage']:.1f}%)")

    # Print hyperparameters
    print("\nHyperparameters:")
    print(f"  Learning rate: {args.lr:.6f}")
    print(f"  Weight decay: {args.wd:.6f}")
    print(f"  Dropout: {args.dropout:.4f}")
    print(f"  Augmentation strength: {args.aug_strength:.4f}")
    print(f"  Unfreeze fraction: {args.unfreeze_frac:.4f}")

    # Train model
    print("\nTraining model...")
    print("=" * 80)

    model, metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.wd,
        patience=args.patience,
        max_epochs=args.max_epochs,
        pos_weight=pos_weight,
        verbose=True
    )

    print("=" * 80)
    print("\nTraining complete!")

    # Print final metrics
    print("\nFinal Validation Metrics (Breast-Level):")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  Brier Score: {metrics['brier']:.4f}")
    print(f"  Robustness Degradation: {metrics['robustness_degradation']:.4f}")

    # Save model if requested
    if args.save_model:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        save_path = os.path.join(CHECKPOINT_DIR, args.model_name)
        torch.save(model.state_dict(), save_path)
        print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
