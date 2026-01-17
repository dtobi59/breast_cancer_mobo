"""
Evaluation utilities including Noisy-OR aggregation and metrics.
"""

import torch
import numpy as np
from torch.utils.data import Subset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_score,
    recall_score,
    confusion_matrix
)


def collect_predictions(model, dataloader, device, perturb_fn=None):
    """
    Collect predictions from model on a dataset.

    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: torch.device
        perturb_fn: Optional perturbation function for robustness testing

    Returns:
        y_true: Array of ground truth labels
        y_probs: Array of predicted probabilities
    """
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Apply perturbation if provided
            if perturb_fn is not None:
                imgs = perturb_fn(imgs)

            # Forward pass
            logits = model(imgs)  # (B,)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)

    return y_true, y_probs


def noisy_or_aggregation(probs):
    """
    Noisy-OR aggregation for breast-level prediction.

    Formula: p_breast = 1 - ∏(1 - p_i) for all views i

    Args:
        probs: List or array of image-level probabilities for same breast

    Returns:
        Aggregated breast-level probability
    """
    probs = np.array(probs)
    return 1.0 - np.prod(1.0 - probs)


def aggregate_breast_level_predictions(model, dataset, device, perturb_fn=None):
    """
    Collect breast-level predictions using Noisy-OR aggregation.

    Args:
        model: PyTorch model
        dataset: Dataset with get_breast_groups() method (or Subset wrapping such dataset)
        device: torch.device
        perturb_fn: Optional perturbation function

    Returns:
        y_true: Array of breast-level ground truth labels
        y_probs: Array of breast-level predicted probabilities
    """
    model.eval()

    # Handle Subset wrapper
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        subset_indices = set(dataset.indices)
    else:
        base_dataset = dataset
        subset_indices = None

    breast_groups = base_dataset.get_breast_groups()

    breast_labels = []
    breast_probs = []

    with torch.no_grad():
        for group in breast_groups:
            image_indices = group["image_indices"]
            label = group["label"]

            # If using a subset, filter to only images in subset
            if subset_indices is not None:
                # Get intersection of breast images and subset indices
                valid_indices = [idx for idx in image_indices if idx in subset_indices]

                # Skip if no images from this breast are in the subset
                if len(valid_indices) == 0:
                    continue

                # Use only the valid indices (may be incomplete breast group)
                image_indices = valid_indices

            # Collect predictions for all images in this breast
            image_probs = []

            for idx in image_indices:
                img, _ = base_dataset[idx]
                img = img.unsqueeze(0).to(device, non_blocking=True)

                # Apply perturbation if provided
                if perturb_fn is not None:
                    img = perturb_fn(img)

                logits = model(img)
                prob = torch.sigmoid(logits).item()
                image_probs.append(prob)

            # Aggregate using Noisy-OR
            breast_prob = noisy_or_aggregation(image_probs)

            breast_labels.append(label)
            breast_probs.append(breast_prob)

    y_true = np.array(breast_labels)
    y_probs = np.array(breast_probs)

    # Sanity check
    if len(y_true) == 0:
        raise ValueError(
            f"No breast groups found in the dataset/subset. "
            f"Dataset has {len(base_dataset)} images. "
            f"Subset has {len(subset_indices) if subset_indices else 'N/A'} indices. "
            f"Total breast groups in base dataset: {len(breast_groups)}."
        )

    return y_true, y_probs


def compute_metrics(y_true, y_probs, threshold=0.5):
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        threshold: Decision threshold

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # AUC metrics
    metrics["auroc"] = roc_auc_score(y_true, y_probs)
    metrics["pr_auc"] = average_precision_score(y_true, y_probs)

    # Brier score (calibration)
    metrics["brier"] = brier_score_loss(y_true, y_probs)

    # Threshold-based metrics
    y_pred = (y_probs >= threshold).astype(int)
    metrics["threshold"] = threshold

    if len(np.unique(y_pred)) > 1:
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    else:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)

    # Sensitivity and specificity
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def find_threshold_at_specificity(y_true, y_probs, target_specificity=0.90):
    """
    Find decision threshold that achieves target specificity.

    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        target_specificity: Desired specificity level

    Returns:
        Threshold value, or None if not achievable
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    specificity = 1 - fpr

    # Find thresholds that achieve at least target specificity
    valid = np.where(specificity >= target_specificity)[0]

    if len(valid) == 0:
        return None

    # Among valid thresholds, pick the one with highest sensitivity
    best_idx = valid[np.argmax(tpr[valid])]

    return thresholds[best_idx]


def evaluate_with_threshold(y_true, y_probs, threshold):
    """
    Evaluate at a specific threshold.

    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        threshold: Decision threshold

    Returns:
        Dictionary of metrics
    """
    return compute_metrics(y_true, y_probs, threshold=threshold)


class MetricsTracker:
    """
    Track metrics across training epochs.
    """

    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_auroc": [],
            "val_pr_auc": [],
            "val_brier": []
        }

    def update(self, epoch, train_loss=None, val_auroc=None, val_pr_auc=None, val_brier=None):
        """Update metrics for current epoch."""
        if train_loss is not None:
            self.history["train_loss"].append(train_loss)
        if val_auroc is not None:
            self.history["val_auroc"].append(val_auroc)
        if val_pr_auc is not None:
            self.history["val_pr_auc"].append(val_pr_auc)
        if val_brier is not None:
            self.history["val_brier"].append(val_brier)

    def get_best_epoch(self, metric="val_pr_auc", mode="max"):
        """Get epoch with best metric value."""
        values = self.history.get(metric, [])
        if not values:
            return None

        if mode == "max":
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        return best_idx

    def get_best_value(self, metric="val_pr_auc", mode="max"):
        """Get best metric value."""
        values = self.history.get(metric, [])
        if not values:
            return None

        if mode == "max":
            return max(values)
        else:
            return min(values)
