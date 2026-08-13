"""
Training pipeline with early stopping.
"""

import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .evaluation import aggregate_breast_level_predictions, compute_metrics
from .robustness import RobustnessTester

def evaluate_image_level_predictions(
    model,
    dataset,
    device,
    batch_size=8
):
    """
    Evaluate a model directly at image level.

    Returns:
        y_true: numpy array of binary labels
        y_probs: numpy array of predicted probabilities
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            # Model outputs logits because training uses
            # BCEWithLogitsLoss
            probabilities = torch.sigmoid(
                logits
            )

            # Flatten safely regardless of (B,) or (B,1)
            probabilities = (
                probabilities
                .view(-1)
                .cpu()
                .numpy()
            )

            labels = (
                labels
                .view(-1)
                .cpu()
                .numpy()
            )

            all_probs.extend(
                probabilities.tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    return (
        np.asarray(all_labels),
        np.asarray(all_probs)
    )
    
class EarlyStopping:
    """
    Early stopping to stop training when monitored metric stops improving.
    """

    def __init__(self, patience=10, mode="max", min_delta=0.0):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            mode: "max" to maximize metric, "min" to minimize
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, metric_value, model):
        """
        Check if training should stop.

        Args:
            metric_value: Current metric value
            model: Current model (for saving best state)

        Returns:
            True if should stop, False otherwise
        """
        if self.mode == "max":
            score = metric_value
        else:
            score = -metric_value

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

        return self.early_stop

    def load_best_model(self, model):
        """Load best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class Trainer:
    """
    PyTorch trainer with early stopping and comprehensive evaluation.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        val_dataset,
        optimizer,
        criterion,
        device,
        patience=10,
        max_epochs=100,
        log_dir=None,
        inbreast_calibration_dataset=None
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (for image-level evaluation)
            val_dataset: Validation Dataset (for breast-level evaluation)
            optimizer: PyTorch optimizer
            criterion: Loss function
            device: torch.device
            patience: Early stopping patience
            max_epochs: Maximum number of epochs
            log_dir: Directory for saving logs and checkpoints
            inbreast_calibration_dataset: INbreast calibration dataset (for cross-dataset degradation)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.max_epochs = max_epochs
        self.log_dir = log_dir
        self.inbreast_calibration_dataset = inbreast_calibration_dataset

        self.early_stopping = EarlyStopping(patience=patience, mode="max")
        self.history = {
            "train_loss": [],
            "val_pr_auc": [],
            "val_auroc": [],
            "val_brier": [],
            "cross_dataset_degradation": []
        }

    def train_one_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0

        for imgs, labels in self.train_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.float().to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            logits = self.model(imgs)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        """
        Validate model with breast-level aggregation.

        Returns:
            Dictionary of validation metrics
        """
        # Collect breast-level predictions using Noisy-OR
        y_true, y_probs = aggregate_breast_level_predictions(
            self.model, self.val_dataset, self.device
        )

        # Compute metrics
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

        metrics = {
            "pr_auc": average_precision_score(y_true, y_probs),
            "auroc": roc_auc_score(y_true, y_probs),
            "brier": brier_score_loss(y_true, y_probs)
        }

        return metrics

    def evaluate_cross_dataset_degradation(self):
        """
        Compute cross-dataset degradation at IMAGE LEVEL.

        Objective 4:

            degradation =
                PR-AUC(VinDr validation images)
                -
                PR-AUC(INbreast calibration images)

        Both sides are evaluated at the same unit (image level),
        making the cross-dataset comparison consistent.

        Lower degradation is better.
        """

        if self.inbreast_calibration_dataset is None:
            return 0.0

        from sklearn.metrics import (
            average_precision_score
        )

        # ======================================================
        # VinDr validation: IMAGE LEVEL
        # ======================================================

        vindr_true, vindr_probs = (
            evaluate_image_level_predictions(
                model=self.model,
                dataset=self.val_dataset,
                device=self.device
            )
        )

        vindr_image_pr_auc = (
            average_precision_score(
                vindr_true,
                vindr_probs
            )
        )

        # ======================================================
        # INbreast calibration: IMAGE LEVEL
        # ======================================================

        inbreast_true, inbreast_probs = (
            evaluate_image_level_predictions(
                model=self.model,
                dataset=self.inbreast_calibration_dataset,
                device=self.device
            )
        )

        inbreast_image_pr_auc = (
            average_precision_score(
                inbreast_true,
                inbreast_probs
            )
        )

        # ======================================================
        # Cross-dataset degradation
        # ======================================================

        degradation = (
            vindr_image_pr_auc
            -
            inbreast_image_pr_auc
        )

        return degradation

    def train(self, verbose=True):
        """
        Full training loop with early stopping.

        Returns:
            Dictionary with final metrics and best epoch
        """
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_one_epoch()

            # Validate
            val_metrics = self.validate()

            # Evaluate cross-dataset degradation (expensive, so maybe skip some epochs)
            if epoch % 2 == 0 or epoch == self.max_epochs - 1:
                cross_dataset_deg = (self.evaluate_cross_dataset_degradation())
            else:
                cross_dataset_deg = 0.0

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_pr_auc"].append(val_metrics["pr_auc"])
            self.history["val_auroc"].append(val_metrics["auroc"])
            self.history["val_brier"].append(val_metrics["brier"])
            self.history["cross_dataset_degradation"].append(cross_dataset_deg)

            if verbose:
                # Show cross-dataset degradation if it was computed this epoch
                if epoch % 2 == 0 or epoch == self.max_epochs - 1:
                    print(
                        f"Epoch {epoch+1}/{self.max_epochs} | "
                        f"Loss: {train_loss:.4f} | "
                        f"PR-AUC: {val_metrics['pr_auc']:.4f} | "
                        f"AUROC: {val_metrics['auroc']:.4f} | "
                        f"Brier: {val_metrics['brier']:.4f} | "
                        f"Cross-dataset: {cross_dataset_deg:.4f}"
                    )
                else:
                    # Skip epochs don't compute degradation, so don't show it
                    print(
                        f"Epoch {epoch+1}/{self.max_epochs} | "
                        f"Loss: {train_loss:.4f} | "
                        f"PR-AUC: {val_metrics['pr_auc']:.4f} | "
                        f"AUROC: {val_metrics['auroc']:.4f} | "
                        f"Brier: {val_metrics['brier']:.4f}"
                    )

            # Early stopping check
            if self.early_stopping(val_metrics["pr_auc"], self.model):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        self.early_stopping.load_best_model(self.model)

        # Final evaluation
        final_metrics = self.validate()
        final_cross_dataset_deg = (self.evaluate_cross_dataset_degradation())

        return {
            "pr_auc": final_metrics["pr_auc"],
            "auroc": final_metrics["auroc"],
            "brier": final_metrics["brier"],
            "cross_dataset_degradation": final_cross_dataset_deg,
            "best_epoch": len(self.history["val_pr_auc"]) - self.early_stopping.patience
        }

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]


def train_model(
    model,
    train_loader,
    val_loader,
    val_dataset,
    device,
    learning_rate,
    weight_decay,
    patience=10,
    max_epochs=100,
    pos_weight=None,
    verbose=True,
    inbreast_calibration_dataset=None
):
    """
    Convenience function to train a model.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        val_dataset: Validation Dataset
        device: torch.device
        learning_rate: Learning rate
        weight_decay: Weight decay
        patience: Early stopping patience
        max_epochs: Maximum epochs
        pos_weight: Positive class weight for BCEWithLogitsLoss
        verbose: Print progress
        inbreast_calibration_dataset: INbreast calibration dataset (for cross-dataset degradation)

    Returns:
        Trained model and metrics dictionary
    """
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Setup loss function
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=patience,
        max_epochs=max_epochs,
        inbreast_calibration_dataset=inbreast_calibration_dataset
    )

    # Train
    metrics = trainer.train(verbose=verbose)

    return model, metrics
