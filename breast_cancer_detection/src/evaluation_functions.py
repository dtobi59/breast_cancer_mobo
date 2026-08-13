"""
Evaluation functions for breast cancer detection optimization.

Contains standalone functions for hyperparameter evaluation without
framework dependencies (no pymoo, no sklearn surrogates).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .models import build_resnet152
from .augmentations import get_augmentation
from .training import train_model
from .datasets import VinDRMammoBinaryDataset


def decode_hyperparameters(x):
    """
    Decode decision variables to hyperparameters.

    Args:
        x: Array or tensor of shape (5,) with:
           [lr_log, wd_log, dropout, aug_strength, unfreeze_frac]
           - lr_log: log10(learning_rate) in [-5, -3]
           - wd_log: log10(weight_decay) in [-6, -2]
           - dropout: dropout rate in [0, 0.5]
           - aug_strength: augmentation strength in [0, 1]
           - unfreeze_frac: fraction of layers to unfreeze in [0, 1]

    Returns:
        Dictionary of hyperparameters with real values
    """
    # Convert to numpy if needed
    if torch.is_tensor(x):
        x = x.cpu().numpy()

    x = np.asarray(x)

    lr_log, wd_log, dropout, aug_strength, unfreeze_frac = x

    return {
        "learning_rate": 10 ** lr_log,
        "weight_decay": 10 ** wd_log,
        "dropout": float(dropout),
        "augmentation_strength": float(aug_strength),
        "unfreeze_fraction": float(unfreeze_frac)
    }


def train_and_evaluate(
    hyperparams,
    train_dataset,
    val_dataset,
    device,
    batch_size=4,
    num_workers=2,
    patience=10,
    max_epochs=100,
    pos_weight=None,
    random_seed=42,
    inbreast_calibration_dataset=None
):
    """
    Train model with given hyperparameters and evaluate.

    Args:
        hyperparams: Dictionary of hyperparameters with keys:
            - learning_rate: float
            - weight_decay: float
            - dropout: float
            - augmentation_strength: float
            - unfreeze_fraction: float
        train_dataset: Training dataset (can be Subset)
        val_dataset: Validation dataset (can be Subset)
        device: torch.device
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        patience: Early stopping patience
        max_epochs: Maximum training epochs
        pos_weight: Positive class weight for loss
        random_seed: Random seed for reproducibility
        inbreast_calibration_dataset: INbreast calibration dataset (for cross-dataset degradation)

    Returns:
        Dictionary of metrics:
            - pr_auc: Precision-recall AUC
            - auroc: ROC AUC
            - brier: Brier score
            - cross_dataset_degradation: Cross-dataset degradation metric
            - best_epoch: Epoch with best validation PR-AUC
    """
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create augmentation transform
    augmentation = get_augmentation(hyperparams["augmentation_strength"])

    # Handle augmentation for Subset datasets
    # Create a new dataset instance with augmentation, then wrap with same indices
    if isinstance(train_dataset, Subset):
        # Get the base dataset and indices
        base_dataset = train_dataset.dataset
        train_indices = train_dataset.indices

        # Create new dataset instance with augmentation
        if isinstance(base_dataset, VinDRMammoBinaryDataset):
            # Create dataset with augmentation, reusing samples from base dataset
            train_dataset_aug = VinDRMammoBinaryDataset(
                images_root=base_dataset.images_root,
                csv_file=None,
                preprocessor=base_dataset.preprocessor,
                transform=augmentation,
                samples=base_dataset.samples  # Pass samples directly
            )
            # Wrap with same indices
            train_dataset_aug = Subset(train_dataset_aug, train_indices)
        else:
            # Fallback: try to set transform on base dataset
            base_dataset.transform = augmentation
            train_dataset_aug = train_dataset
    else:
        # Not a Subset, set transform directly
        train_dataset.transform = augmentation
        train_dataset_aug = train_dataset

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Build model
    model = build_resnet152(
        pretrained=True,
        dropout=hyperparams["dropout"],
        unfreeze_fraction=hyperparams["unfreeze_fraction"]
    )
    model = model.to(device)

    # Train model
    model, metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        device=device,
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        patience=patience,
        max_epochs=max_epochs,
        pos_weight=pos_weight,
        verbose=True,  # Show training progress
        inbreast_calibration_dataset=inbreast_calibration_dataset
    )

    return metrics


def create_evaluation_function(
    train_dataset,
    val_dataset,
    device,
    batch_size=4,
    num_workers=2,
    patience=10,
    max_epochs=100,
    pos_weight=None,
    random_seed=42,
    inbreast_calibration_dataset=None
):
    """
    Factory function to create a configured evaluation function.

    Returns a function that takes hyperparameters and returns metrics.
    This is useful for BoTorch optimization where we need a simple
    callable interface.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: torch.device
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        patience: Early stopping patience
        max_epochs: Maximum training epochs
        pos_weight: Positive class weight for loss
        random_seed: Random seed
        inbreast_calibration_dataset: INbreast calibration dataset (for cross-dataset degradation)

    Returns:
        Function with signature: evaluate_fn(hyperparams) -> metrics_dict
    """
    def evaluate_fn(hyperparams):
        """Evaluate a single hyperparameter configuration."""
        return train_and_evaluate(
            hyperparams=hyperparams,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            patience=patience,
            max_epochs=max_epochs,
            pos_weight=pos_weight,
            random_seed=random_seed,
            inbreast_calibration_dataset=inbreast_calibration_dataset
        )

    return evaluate_fn
