"""
Breast Cancer Detection Source Package
"""

from .preprocessing import MammographyPreprocessor
from .datasets import (
    VinDRMammoBinaryDataset,
    INbreastDataset,
    create_breast_level_splits,
    create_vindr_dataset_with_adaptation,
    create_inbreast_dataset_with_adaptation
)
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
from .domain_adaptation import (
    calculate_shannon_entropy,
    adaptive_entropy_transform,
    EntropyStatistics,
    compute_dataset_entropy_stats
)
from .adaptive_preprocessing import AdaptiveMammographyPreprocessor
from .cache_manager import EntropyCache, ensure_entropy_cache

__all__ = [
    # Preprocessing
    "MammographyPreprocessor",
    "AdaptiveMammographyPreprocessor",

    # Datasets
    "VinDRMammoBinaryDataset",
    "INbreastDataset",
    "create_breast_level_splits",
    "create_vindr_dataset_with_adaptation",
    "create_inbreast_dataset_with_adaptation",

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

    # Domain Adaptation
    "calculate_shannon_entropy",
    "adaptive_entropy_transform",
    "EntropyStatistics",
    "compute_dataset_entropy_stats",
    "EntropyCache",
    "ensure_entropy_cache",
]
