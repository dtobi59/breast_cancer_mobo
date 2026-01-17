"""
Augmentation strength controller for intensity-based augmentations.

NO geometric transforms - only intensity augmentations:
- Brightness adjustment
- Contrast scaling
- Additive Gaussian noise
"""

import torch
import torch.nn as nn


class IntensityAugmentation(nn.Module):
    """
    Intensity-based augmentation with scalar strength controller.

    Augmentation strength ∈ [0, 1] controls the intensity of:
    - Brightness perturbation
    - Contrast scaling
    - Gaussian noise addition

    Applied ONLY during training.
    """

    def __init__(self, strength=0.5):
        """
        Args:
            strength: Augmentation strength in [0, 1]
                     0 = no augmentation
                     1 = maximum augmentation
        """
        super().__init__()
        self.strength = max(0.0, min(1.0, strength))  # Clamp to [0, 1]

    def forward(self, img):
        """
        Apply intensity augmentations to image tensor.

        Args:
            img: Tensor of shape (C, H, W) with values in [0, 1]

        Returns:
            Augmented tensor of shape (C, H, W)
        """
        if self.strength == 0.0:
            return img

        # 1. Brightness adjustment
        # Range: [-strength * 0.2, +strength * 0.2]
        brightness_delta = (torch.rand(1).item() - 0.5) * 2 * self.strength * 0.2
        img = img + brightness_delta

        # 2. Contrast scaling
        # Range: [1 - strength * 0.3, 1 + strength * 0.3]
        contrast_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.strength * 0.3
        mean = img.mean()
        img = (img - mean) * contrast_factor + mean

        # 3. Additive Gaussian noise
        # Std: strength * 0.05
        noise_std = self.strength * 0.05
        noise = torch.randn_like(img) * noise_std
        img = img + noise

        # Clamp to valid range
        img = torch.clamp(img, 0.0, 1.0)

        return img


def get_augmentation(strength):
    """
    Factory function to create augmentation transform.

    Args:
        strength: Augmentation strength in [0, 1]

    Returns:
        IntensityAugmentation module
    """
    return IntensityAugmentation(strength=strength)


class RobustnessPerturb(nn.Module):
    """
    Mild intensity perturbations for robustness evaluation.

    Applied at INFERENCE time only to measure robustness degradation.
    """

    def __init__(self, brightness_delta=0.1, contrast_factor=0.1, noise_std=0.02):
        """
        Args:
            brightness_delta: Maximum brightness shift (±)
            contrast_factor: Maximum contrast scaling factor (±)
            noise_std: Gaussian noise standard deviation
        """
        super().__init__()
        self.brightness_delta = brightness_delta
        self.contrast_factor = contrast_factor
        self.noise_std = noise_std

    def forward(self, img):
        """
        Apply mild perturbations for robustness testing.

        Args:
            img: Tensor of shape (C, H, W) with values in [0, 1]

        Returns:
            Perturbed tensor of shape (C, H, W)
        """
        # Random brightness
        delta = (torch.rand(1).item() - 0.5) * 2 * self.brightness_delta
        img = img + delta

        # Random contrast
        factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.contrast_factor
        mean = img.mean()
        img = (img - mean) * factor + mean

        # Gaussian noise
        noise = torch.randn_like(img) * self.noise_std
        img = img + noise

        # Clamp to valid range
        img = torch.clamp(img, 0.0, 1.0)

        return img
