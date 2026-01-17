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
        log_dir=None
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

        self.early_stopping = EarlyStopping(patience=patience, mode="max")
        self.history = {
            "train_loss": [],
            "val_pr_auc": [],
            "val_auroc": [],
            "val_brier": [],
            "val_robustness": []
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

    def evaluate_robustness(self):
        """Evaluate robustness degradation."""
        tester = RobustnessTester()
        results = tester.evaluate_robustness(
            self.model, self.val_loader, self.device
        )
        return results["degradation"]

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

            # Evaluate robustness (expensive, so maybe skip some epochs)
            if epoch % 5 == 0 or epoch == self.max_epochs - 1:
                robustness_deg = self.evaluate_robustness()
            else:
                robustness_deg = 0.0

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_pr_auc"].append(val_metrics["pr_auc"])
            self.history["val_auroc"].append(val_metrics["auroc"])
            self.history["val_brier"].append(val_metrics["brier"])
            self.history["val_robustness"].append(robustness_deg)

            if verbose:
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
        final_robustness = self.evaluate_robustness()

        return {
            "pr_auc": final_metrics["pr_auc"],
            "auroc": final_metrics["auroc"],
            "brier": final_metrics["brier"],
            "robustness_degradation": final_robustness,
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
    verbose=True
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
        max_epochs=max_epochs
    )

    # Train
    metrics = trainer.train(verbose=verbose)

    return model, metrics
