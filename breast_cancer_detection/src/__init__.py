"""
Breast Cancer Detection Source Package
"""

from .preprocessing import MammographyPreprocessor
from .datasets import VinDRMammoBinaryDataset, INbreastDataset
from .augmentations import IntensityAugmentation, get_augmentation, RobustnessPerturb
from .models import ResNet152Binary, build_resnet152
from .training import Trainer, train_model, EarlyStopping
from .evaluation import (
    collect_predictions,
    noisy_or_aggregation,
    aggregate_breast_level_predictions,
    compute_metrics,
    find_threshold_at_specificity,
    MetricsTracker
)
from .robustness import RobustnessTester, compute_robustness_degradation
from .optimization import BreastCancerOptimizationProblem, HyperparameterLogger

__all__ = [
    # Preprocessing
    "MammographyPreprocessor",

    # Datasets
    "VinDRMammoBinaryDataset",
    "INbreastDataset",

    # Augmentations
    "IntensityAugmentation",
    "get_augmentation",
    "RobustnessPerturb",

    # Models
    "ResNet152Binary",
    "build_resnet152",

    # Training
    "Trainer",
    "train_model",
    "EarlyStopping",

    # Evaluation
    "collect_predictions",
    "noisy_or_aggregation",
    "aggregate_breast_level_predictions",
    "compute_metrics",
    "find_threshold_at_specificity",
    "MetricsTracker",

    # Robustness
    "RobustnessTester",
    "compute_robustness_degradation",

    # Optimization
    "BreastCancerOptimizationProblem",
    "HyperparameterLogger",
]
