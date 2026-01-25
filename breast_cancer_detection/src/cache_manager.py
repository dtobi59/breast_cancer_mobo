"""
Cache management for entropy statistics.

Provides utilities for caching entropy statistics to avoid
recomputation on subsequent runs.
"""

import os
from pathlib import Path
from typing import Optional

from .domain_adaptation import EntropyStatistics


class EntropyCache:
    """
    Manage entropy statistics cache files.

    Features:
    - Automatic cache directory creation
    - Cache validation
    - Simple load/save interface

    Usage:
        >>> cache = EntropyCache(cache_dir="cache")
        >>>
        >>> # Check if cache exists
        >>> if cache.exists("vindr_train"):
        >>>     stats = cache.load("vindr_train")
        >>> else:
        >>>     # Compute stats...
        >>>     cache.save(stats, "vindr_train")
    """

    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files (relative to project root)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, dataset_name: str) -> Path:
        """
        Get cache file path for dataset.

        Args:
            dataset_name: Name identifier for dataset (e.g., "vindr_train")

        Returns:
            Path object for cache file
        """
        return self.cache_dir / f"entropy_stats_{dataset_name}.json"

    def exists(self, dataset_name: str) -> bool:
        """
        Check if cache exists for dataset.

        Args:
            dataset_name: Name identifier for dataset

        Returns:
            True if cache file exists, False otherwise
        """
        return self.get_cache_path(dataset_name).exists()

    def load(self, dataset_name: str) -> Optional[EntropyStatistics]:
        """
        Load cached statistics for dataset.

        Args:
            dataset_name: Name identifier for dataset

        Returns:
            EntropyStatistics object if cache exists, None otherwise

        Raises:
            FileNotFoundError: If cache file doesn't exist
            json.JSONDecodeError: If cache file is corrupted
        """
        path = self.get_cache_path(dataset_name)

        if not path.exists():
            raise FileNotFoundError(
                f"Cache file not found: {path}\n"
                f"Run compute_entropy_stats.py to generate cache."
            )

        return EntropyStatistics.load(str(path))

    def save(self, stats: EntropyStatistics, dataset_name: str):
        """
        Save statistics to cache.

        Args:
            stats: EntropyStatistics object to save
            dataset_name: Name identifier for dataset
        """
        path = self.get_cache_path(dataset_name)
        stats.save(str(path))

    def invalidate(self, dataset_name: str):
        """
        Delete cache file for dataset.

        Args:
            dataset_name: Name identifier for dataset

        Returns:
            True if file was deleted, False if didn't exist
        """
        path = self.get_cache_path(dataset_name)

        if path.exists():
            path.unlink()
            return True
        return False

    def list_cached_datasets(self):
        """
        List all cached datasets.

        Returns:
            List of dataset names that have cached entropy statistics
        """
        cached = []

        for file in self.cache_dir.glob("entropy_stats_*.json"):
            # Extract dataset name from filename
            # Format: entropy_stats_{dataset_name}.json
            dataset_name = file.stem.replace("entropy_stats_", "")
            cached.append(dataset_name)

        return cached

    def get_cache_info(self, dataset_name: str) -> dict:
        """
        Get information about cached statistics.

        Args:
            dataset_name: Name identifier for dataset

        Returns:
            Dictionary with cache info (file size, modified time, stats summary)

        Raises:
            FileNotFoundError: If cache doesn't exist
        """
        path = self.get_cache_path(dataset_name)

        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")

        # Load statistics
        stats = self.load(dataset_name)

        # Get file info
        stat_info = path.stat()

        return {
            'dataset_name': dataset_name,
            'file_path': str(path),
            'file_size_bytes': stat_info.st_size,
            'modified_time': stat_info.st_mtime,
            'mean_entropy': stats.mean,
            'std_entropy': stats.std,
            'n_samples': stats.n_samples,
            'computed_date': stats.computed_date
        }


def ensure_entropy_cache(
    dataset_name: str = "vindr_train",
    cache_dir: str = "cache"
) -> EntropyStatistics:
    """
    Ensure entropy cache exists, prompting user if not.

    Utility function to check for cache and provide helpful error
    message if not found.

    Args:
        dataset_name: Name identifier for dataset
        cache_dir: Cache directory

    Returns:
        EntropyStatistics object

    Raises:
        FileNotFoundError: If cache doesn't exist with instructions

    Example:
        >>> # In evaluation scripts
        >>> try:
        >>>     stats = ensure_entropy_cache("vindr_train")
        >>> except FileNotFoundError as e:
        >>>     print(e)
        >>>     exit(1)
    """
    cache = EntropyCache(cache_dir=cache_dir)

    if not cache.exists(dataset_name):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"ERROR: Entropy cache not found for '{dataset_name}'\n"
            f"{'='*60}\n\n"
            f"The entropy statistics cache is required for domain adaptation.\n\n"
            f"To generate the cache, run:\n\n"
            f"  cd breast_cancer_detection\n"
            f"  python scripts/compute_entropy_stats.py\n\n"
            f"Expected cache location:\n"
            f"  {cache.get_cache_path(dataset_name)}\n\n"
            f"This is a ONE-TIME computation that takes ~1-2 hours.\n"
            f"{'='*60}\n"
        )

    return cache.load(dataset_name)
