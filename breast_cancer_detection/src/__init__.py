"""
Breast Cancer Detection Source Package
"""

from .preprocessing import MammographyPreprocessor
from .datasets import (
    VinDRMammoBinaryDataset,
    INbreastDataset,
    create_breast_level_splits,
    create_inbreast_calibration_test_splits,
    create_vindr_dataset_with_adaptation,
    create_inbreast_dataset_with_adaptation
)
from .inbreast_xls_loader import INbreastDatasetFromXLS, load_inbreast_from_xls
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
from .evaluation_functions import decode_hyperparameters, train_and_evaluate, create_evaluation_function
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
    "INbreastDatasetFromXLS",
    "load_inbreast_from_xls",
    "create_breast_level_splits",
    "create_inbreast_calibration_test_splits",
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

    # Evaluation Functions
    "decode_hyperparameters",
    "train_and_evaluate",
    "create_evaluation_function",

    # Domain Adaptation
    "calculate_shannon_entropy",
    "adaptive_entropy_transform",
    "EntropyStatistics",
    "compute_dataset_entropy_stats",
    "EntropyCache",
    "ensure_entropy_cache",
]
