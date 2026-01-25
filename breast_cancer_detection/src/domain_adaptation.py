"""
Domain adaptation utilities for entropy-based style transfer.

This module implements entropy-based domain adaptation to mitigate
domain shift between source (VinDr-Mammo) and target (INbreast) datasets.
"""

import os
import json
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime
from tqdm import tqdm


def calculate_shannon_entropy(image: np.ndarray) -> float:
    """
    Compute Shannon entropy of grayscale image.

    Shannon entropy measures the information content/unpredictability
    of the intensity distribution. Higher entropy indicates more uniform
    distribution, lower entropy indicates concentrated distribution.

    Formula: H(X) = -Σ p(x) log₂ p(x) where p(x) is probability of intensity x

    Args:
        image: Grayscale image, uint8 [0, 255], shape (H, W)

    Returns:
        Entropy value in bits (range [0, 8] for 8-bit images)

    Examples:
        >>> # Uniform image (all same intensity) → entropy = 0
        >>> img = np.ones((100, 100), dtype=np.uint8) * 128
        >>> calculate_shannon_entropy(img)
        0.0

        >>> # Binary image (two intensities) → entropy ≈ 1 bit
        >>> img = np.zeros((100, 100), dtype=np.uint8)
        >>> img[:50, :] = 255
        >>> calculate_shannon_entropy(img)
        1.0
    """
    # Compute histogram with 256 bins (one per intensity level)
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

    # Convert to probability distribution
    probabilities = histogram / histogram.sum()

    # Remove zero probabilities (log(0) is undefined)
    probabilities = probabilities[probabilities > 0]

    # Calculate Shannon entropy: H = -Σ p * log2(p)
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)


def adaptive_entropy_transform(
    image: np.ndarray,
    target_mean: float,
    target_std: float,
    max_iterations: int = 50,
    tolerance: float = 0.1
) -> Tuple[np.ndarray, Dict]:
    """
    Iteratively adjust image to match target entropy range.

    This function applies iterative squaring/square-root transformations
    to adjust image entropy to fall within the target range [mean-std, mean+std].

    - Squaring increases contrast → increases entropy
    - Square root decreases contrast → decreases entropy

    Args:
        image: Grayscale image, uint8 [0, 255], shape (H, W)
        target_mean: Target entropy mean from source domain
        target_std: Target entropy std from source domain
        max_iterations: Maximum iterations (prevents infinite loops)
        tolerance: Early stopping tolerance in bits

    Returns:
        adapted_image: Transformed image, uint8 [0, 255]
        metrics: Dictionary containing:
            - initial_entropy: Entropy before transformation
            - final_entropy: Entropy after transformation
            - iterations: Number of iterations performed
            - converged: Boolean indicating if target range was reached

    Algorithm:
        1. Calculate current entropy
        2. Define target range: [mean - std, mean + std]
        3. While entropy outside range AND iterations < max:
            a. If entropy < mean - std:
               - Square image: img² → increases contrast
               - Renormalize to [0, 255]
            b. If entropy > mean + std:
               - Square root: √img → decreases contrast
               - Renormalize to [0, 255]
            c. Recalculate entropy
            d. Check convergence
        4. Return adapted image + metrics

    Example:
        >>> img = np.random.randint(120, 140, (100, 100), dtype=np.uint8)
        >>> adapted, metrics = adaptive_entropy_transform(
        ...     img, target_mean=6.5, target_std=0.5
        ... )
        >>> print(f"Converged: {metrics['converged']}")
        >>> print(f"Initial entropy: {metrics['initial_entropy']:.4f}")
        >>> print(f"Final entropy: {metrics['final_entropy']:.4f}")
    """
    # Ensure image is float for transformations
    img = image.astype(np.float64)

    # Calculate initial entropy
    initial_entropy = calculate_shannon_entropy(image)

    # Define target range
    target_min = target_mean - target_std
    target_max = target_mean + target_std

    current_entropy = initial_entropy
    iteration = 0
    converged = False

    # Track last few entropy values to detect oscillation
    entropy_history = []

    while iteration < max_iterations:
        # Check if within target range
        if target_min <= current_entropy <= target_max:
            converged = True
            break

        # Check for oscillation (bouncing around target)
        if len(entropy_history) >= 3:
            recent = entropy_history[-3:]
            if (max(recent) - min(recent)) < tolerance:
                # Entropy stopped changing significantly
                break

        # Apply transformation based on current entropy
        if current_entropy < target_min:
            # Entropy too low → increase contrast via squaring
            # Normalize to [0, 1] first to prevent overflow
            img_normalized = img / 255.0
            img = img_normalized ** 2

        elif current_entropy > target_max:
            # Entropy too high → decrease contrast via square root
            # Normalize to [0, 1] first
            img_normalized = img / 255.0
            img = np.sqrt(img_normalized)

        # Renormalize to [0, 255]
        img_min = img.min()
        img_max = img.max()

        if img_max > img_min:  # Avoid division by zero
            img = (img - img_min) / (img_max - img_min) * 255.0

        # Convert to uint8 for entropy calculation
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

        # Recalculate entropy
        current_entropy = calculate_shannon_entropy(img_uint8)
        entropy_history.append(current_entropy)

        iteration += 1

    # Final image as uint8
    adapted_image = np.clip(img, 0, 255).astype(np.uint8)
    final_entropy = calculate_shannon_entropy(adapted_image)

    # Compile metrics
    metrics = {
        'initial_entropy': initial_entropy,
        'final_entropy': final_entropy,
        'iterations': iteration,
        'converged': converged,
        'target_range': (target_min, target_max)
    }

    return adapted_image, metrics


class EntropyStatistics:
    """
    Container for entropy statistics from source domain.

    Stores mean, std, min, max entropy values computed from
    training dataset for use in domain adaptation.

    Attributes:
        mean: Mean entropy across training images
        std: Standard deviation of entropy
        min_entropy: Minimum observed entropy
        max_entropy: Maximum observed entropy
        n_samples: Number of samples used for calculation
        computed_date: Date statistics were computed
        dataset_name: Name of source dataset
    """

    def __init__(
        self,
        mean: float,
        std: float,
        min_entropy: float,
        max_entropy: float,
        n_samples: int,
        computed_date: Optional[str] = None,
        dataset_name: str = "unknown"
    ):
        """
        Initialize entropy statistics.

        Args:
            mean: Mean entropy value
            std: Standard deviation of entropy
            min_entropy: Minimum entropy value
            max_entropy: Maximum entropy value
            n_samples: Number of samples
            computed_date: Date computed (ISO format)
            dataset_name: Name of dataset
        """
        self.mean = mean
        self.std = std
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.n_samples = n_samples
        self.computed_date = computed_date or datetime.now().isoformat()
        self.dataset_name = dataset_name

    def save(self, filepath: str):
        """
        Save statistics to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'mean': float(self.mean),
            'std': float(self.std),
            'min_entropy': float(self.min_entropy),
            'max_entropy': float(self.max_entropy),
            'n_samples': int(self.n_samples),
            'computed_date': self.computed_date,
            'dataset_name': self.dataset_name
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'EntropyStatistics':
        """
        Load statistics from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            EntropyStatistics instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            mean=data['mean'],
            std=data['std'],
            min_entropy=data['min_entropy'],
            max_entropy=data['max_entropy'],
            n_samples=data['n_samples'],
            computed_date=data.get('computed_date'),
            dataset_name=data.get('dataset_name', 'unknown')
        )

    def __repr__(self):
        return (
            f"EntropyStatistics(dataset={self.dataset_name}, "
            f"mean={self.mean:.4f}, std={self.std:.4f}, "
            f"n={self.n_samples})"
        )


def compute_dataset_entropy_stats(
    dataset,
    preprocessor,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> EntropyStatistics:
    """
    Compute entropy statistics across entire dataset.

    This function iterates through all images in the dataset,
    computes entropy for each, and aggregates statistics.

    Args:
        dataset: Dataset object with __len__ and __getitem__
        preprocessor: Preprocessor instance (must stop before colormap)
        save_path: Optional path to save statistics
        verbose: Show progress bar

    Returns:
        EntropyStatistics object

    Notes:
        - This is a ONE-TIME computation (can take 1-2 hours for large datasets)
        - Results should be cached for reuse
        - Requires preprocessor that can return grayscale before colormap

    Example:
        >>> from src.preprocessing import MammographyPreprocessor
        >>> from src.datasets import VinDRMammoBinaryDataset
        >>>
        >>> preprocessor = MammographyPreprocessor()
        >>> dataset = VinDRMammoBinaryDataset(...)
        >>>
        >>> stats = compute_dataset_entropy_stats(
        ...     dataset, preprocessor,
        ...     save_path='cache/entropy_stats.json',
        ...     verbose=True
        ... )
        >>> print(f"Mean entropy: {stats.mean:.4f}")
    """
    entropy_values = []

    iterator = range(len(dataset))
    if verbose:
        iterator = tqdm(iterator, desc="Computing entropy")

    for idx in iterator:
        try:
            # Get sample from dataset
            # Note: We need grayscale before colormap
            # This requires special handling
            sample_info = dataset.samples[idx]

            # Construct path based on dataset type
            if hasattr(dataset, 'images_root'):
                # VinDr-Mammo dataset
                study_id, image_id = sample_info[0], sample_info[1]
                from pathlib import Path
                dicom_path = Path(dataset.images_root) / str(study_id) / f"{image_id}.dicom"
            else:
                # INbreast dataset
                dicom_path = sample_info[0]

            # Preprocess to get grayscale
            # IMPORTANT: This assumes preprocessor is AdaptiveMammographyPreprocessor
            # or has a method to get grayscale before colormap
            if hasattr(preprocessor, 'resize_grayscale'):
                # AdaptiveMammographyPreprocessor
                img = preprocessor.load_dicom(str(dicom_path))
                img = preprocessor.ensure_left(img)
                roi = preprocessor.extract_single_breast(img)
                roi = preprocessor.remove_inferior_fold_adaptive(roi)
                roi = preprocessor.suppress_nipple(roi)
                roi = preprocessor.resize_grayscale(roi)
            else:
                # Regular MammographyPreprocessor - need workaround
                # We'll replicate preprocessing up to grayscale
                img = preprocessor.load_dicom(str(dicom_path))
                img = preprocessor.ensure_left(img)
                roi = preprocessor.extract_single_breast(img)
                roi = preprocessor.remove_inferior_fold_adaptive(roi)
                roi = preprocessor.suppress_nipple(roi)

                # Resize manually (replicate resize logic without colormap)
                import cv2
                h, w = roi.shape
                aspect_ratio = preprocessor.aspect_ratio
                target_width, target_height = preprocessor.target_width, preprocessor.target_height

                if h / w > aspect_ratio:
                    new_w = int(np.ceil(h / aspect_ratio))
                    pad = new_w - w
                    roi = np.pad(roi, pad_width=((0, 0), (0, pad)),
                                mode="constant", constant_values=0)

                roi = cv2.resize(roi, (target_width, target_height),
                                interpolation=cv2.INTER_LINEAR)

            # Calculate entropy
            entropy = calculate_shannon_entropy(roi)
            entropy_values.append(entropy)

        except Exception as e:
            if verbose:
                print(f"\nWarning: Failed to process sample {idx}: {e}")
            continue

    if len(entropy_values) == 0:
        raise ValueError("No valid entropy values computed!")

    # Compute statistics
    entropy_array = np.array(entropy_values)

    stats = EntropyStatistics(
        mean=float(np.mean(entropy_array)),
        std=float(np.std(entropy_array)),
        min_entropy=float(np.min(entropy_array)),
        max_entropy=float(np.max(entropy_array)),
        n_samples=len(entropy_values),
        dataset_name="training_set"
    )

    # Save if path provided
    if save_path:
        stats.save(save_path)
        if verbose:
            print(f"\nSaved entropy statistics to: {save_path}")

    return stats
