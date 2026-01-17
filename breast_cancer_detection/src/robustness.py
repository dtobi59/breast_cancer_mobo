"""
Robustness degradation measurement.

Evaluates model robustness by measuring performance drop under
mild intensity perturbations at inference time.
"""

import torch
import numpy as np
from sklearn.metrics import average_precision_score


class RobustnessTester:
    """
    Test model robustness to intensity perturbations.

    Measures:
    R = PR-AUC(standard) - PR-AUC(perturbed)
    """

    def __init__(
        self,
        brightness_delta=0.1,
        contrast_factor=0.1,
        noise_std=0.02
    ):
        """
        Args:
            brightness_delta: Max brightness shift (±)
            contrast_factor: Max contrast scaling factor (±)
            noise_std: Gaussian noise standard deviation
        """
        self.brightness_delta = brightness_delta
        self.contrast_factor = contrast_factor
        self.noise_std = noise_std

    def perturb_batch(self, imgs):
        """
        Apply mild perturbations to a batch of images.

        Args:
            imgs: Tensor of shape (B, C, H, W) with values in [0, 1]

        Returns:
            Perturbed tensor of shape (B, C, H, W)
        """
        imgs = imgs.clone()

        # Random brightness (same for all images in batch for consistency)
        delta = (torch.rand(1).item() - 0.5) * 2 * self.brightness_delta
        imgs = imgs + delta

        # Random contrast
        factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.contrast_factor
        mean = imgs.mean(dim=(2, 3), keepdim=True)
        imgs = (imgs - mean) * factor + mean

        # Gaussian noise (per-pixel)
        noise = torch.randn_like(imgs) * self.noise_std
        imgs = imgs + noise

        # Clamp to valid range
        imgs = torch.clamp(imgs, 0.0, 1.0)

        return imgs

    def evaluate_robustness(self, model, dataloader, device):
        """
        Measure robustness degradation on validation set.

        Returns:
            Dictionary with:
            - pr_auc_standard: PR-AUC on clean images
            - pr_auc_perturbed: PR-AUC on perturbed images
            - degradation: pr_auc_standard - pr_auc_perturbed
        """
        model.eval()

        all_labels = []
        all_probs_clean = []
        all_probs_perturbed = []

        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Standard inference
                logits_clean = model(imgs)
                probs_clean = torch.sigmoid(logits_clean)

                # Perturbed inference
                imgs_perturbed = self.perturb_batch(imgs)
                logits_perturbed = model(imgs_perturbed)
                probs_perturbed = torch.sigmoid(logits_perturbed)

                all_labels.append(labels.cpu().numpy())
                all_probs_clean.append(probs_clean.cpu().numpy())
                all_probs_perturbed.append(probs_perturbed.cpu().numpy())

        # Concatenate results
        y_true = np.concatenate(all_labels)
        y_probs_clean = np.concatenate(all_probs_clean)
        y_probs_perturbed = np.concatenate(all_probs_perturbed)

        # Compute PR-AUC for both
        pr_auc_standard = average_precision_score(y_true, y_probs_clean)
        pr_auc_perturbed = average_precision_score(y_true, y_probs_perturbed)

        # Compute degradation
        degradation = pr_auc_standard - pr_auc_perturbed

        return {
            "pr_auc_standard": pr_auc_standard,
            "pr_auc_perturbed": pr_auc_perturbed,
            "degradation": degradation
        }

    def evaluate_robustness_breast_level(self, model, dataset, device):
        """
        Measure robustness degradation with breast-level aggregation.

        Args:
            model: PyTorch model
            dataset: Dataset with get_breast_groups() method
            device: torch.device

        Returns:
            Dictionary with robustness metrics at breast level
        """
        from .evaluation import noisy_or_aggregation

        model.eval()

        breast_groups = dataset.get_breast_groups()

        breast_labels = []
        breast_probs_clean = []
        breast_probs_perturbed = []

        with torch.no_grad():
            for group in breast_groups:
                image_indices = group["image_indices"]
                label = group["label"]

                image_probs_clean = []
                image_probs_perturbed = []

                for idx in image_indices:
                    img, _ = dataset[idx]
                    img = img.unsqueeze(0).to(device, non_blocking=True)

                    # Clean inference
                    logits_clean = model(img)
                    prob_clean = torch.sigmoid(logits_clean).item()
                    image_probs_clean.append(prob_clean)

                    # Perturbed inference
                    img_perturbed = self.perturb_batch(img)
                    logits_perturbed = model(img_perturbed)
                    prob_perturbed = torch.sigmoid(logits_perturbed).item()
                    image_probs_perturbed.append(prob_perturbed)

                # Aggregate using Noisy-OR
                breast_prob_clean = noisy_or_aggregation(image_probs_clean)
                breast_prob_perturbed = noisy_or_aggregation(image_probs_perturbed)

                breast_labels.append(label)
                breast_probs_clean.append(breast_prob_clean)
                breast_probs_perturbed.append(breast_prob_perturbed)

        # Convert to arrays
        y_true = np.array(breast_labels)
        y_probs_clean = np.array(breast_probs_clean)
        y_probs_perturbed = np.array(breast_probs_perturbed)

        # Compute PR-AUC
        pr_auc_standard = average_precision_score(y_true, y_probs_clean)
        pr_auc_perturbed = average_precision_score(y_true, y_probs_perturbed)

        # Compute degradation
        degradation = pr_auc_standard - pr_auc_perturbed

        return {
            "pr_auc_standard": pr_auc_standard,
            "pr_auc_perturbed": pr_auc_perturbed,
            "degradation": degradation
        }


def compute_robustness_degradation(
    model,
    dataloader,
    device,
    brightness_delta=0.1,
    contrast_factor=0.1,
    noise_std=0.02
):
    """
    Convenience function to compute robustness degradation.

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        device: torch.device
        brightness_delta: Brightness perturbation strength
        contrast_factor: Contrast perturbation strength
        noise_std: Noise perturbation strength

    Returns:
        Robustness degradation value (R)
    """
    tester = RobustnessTester(
        brightness_delta=brightness_delta,
        contrast_factor=contrast_factor,
        noise_std=noise_std
    )

    results = tester.evaluate_robustness(model, dataloader, device)

    return results["degradation"]
