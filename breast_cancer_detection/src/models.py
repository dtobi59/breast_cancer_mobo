"""
ResNet152 model with partial fine-tuning control.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet152Binary(nn.Module):
    """
    ResNet152 for binary classification with controllable fine-tuning.

    Features:
    - ImageNet pretrained backbone
    - Single output node with sigmoid activation
    - Dropout before final layer
    - Continuous control over which layers to fine-tune
    """

    def __init__(
        self,
        pretrained=True,
        dropout=0.0,
        unfreeze_fraction=1.0
    ):
        """
        Args:
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before final layer [0, 1]
            unfreeze_fraction: Fraction of backbone layers to unfreeze [0, 1]
                              0.0 = freeze all backbone (only train head)
                              1.0 = unfreeze all layers
                              0.5 = unfreeze last 50% of layers
        """
        super().__init__()

        # Load pretrained ResNet152
        if pretrained:
            weights = models.ResNet152_Weights.IMAGENET1K_V1
            self.backbone = models.resnet152(weights=weights)
        else:
            self.backbone = models.resnet152(weights=None)

        # Get number of features from final layer
        in_features = self.backbone.fc.in_features

        # Replace final FC layer with dropout + binary classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1)  # Single output for binary classification
        )

        # Apply partial fine-tuning
        self._set_trainable_layers(unfreeze_fraction)

    def _set_trainable_layers(self, unfreeze_fraction):
        """
        Freeze/unfreeze layers based on unfreeze_fraction.

        Strategy:
        - Always keep final FC layer trainable
        - Freeze/unfreeze backbone layers from bottom to top
        - unfreeze_fraction=0.0: freeze all backbone
        - unfreeze_fraction=1.0: unfreeze all backbone
        - unfreeze_fraction=0.5: freeze first 50%, unfreeze last 50%
        """
        unfreeze_fraction = max(0.0, min(1.0, unfreeze_fraction))

        # Get all named parameters in backbone (excluding fc)
        backbone_params = []
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc"):
                backbone_params.append((name, param))

        # Calculate how many to unfreeze (from the end)
        n_params = len(backbone_params)
        n_unfreeze = int(n_params * unfreeze_fraction)

        # Freeze all backbone parameters first
        for name, param in backbone_params:
            param.requires_grad = False

        # Unfreeze the last n_unfreeze parameters
        for name, param in backbone_params[-n_unfreeze:]:
            param.requires_grad = True

        # Always keep final FC layer trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Logits of shape (B,) - use with BCEWithLogitsLoss
        """
        return self.backbone(x).squeeze(1)

    def get_trainable_params_info(self):
        """
        Get information about trainable parameters.

        Returns:
            dict with counts and percentages
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params
        }


def build_resnet152(pretrained=True, dropout=0.0, unfreeze_fraction=1.0):
    """
    Factory function to create ResNet152 binary classifier.

    Args:
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate before final layer
        unfreeze_fraction: Fraction of backbone layers to unfreeze [0, 1]

    Returns:
        ResNet152Binary model
    """
    model = ResNet152Binary(
        pretrained=pretrained,
        dropout=dropout,
        unfreeze_fraction=unfreeze_fraction
    )

    return model
