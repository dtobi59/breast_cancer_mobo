"""
Compute and cache entropy statistics from VinDr-Mammo training set.

This is a ONE-TIME script that computes Shannon entropy for all images
in the training set and caches the statistics (mean, std) for use in
domain adaptation.

Expected runtime: ~1-2 hours for full dataset

Usage:
    cd breast_cancer_detection
    python scripts/compute_entropy_stats.py

Output:
    cache/entropy_stats_vindr_train.json
"""

import os
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import MammographyPreprocessor
from src.adaptive_preprocessing import AdaptiveMammographyPreprocessor
from src.datasets import VinDRMammoBinaryDataset, create_breast_level_splits
from src.domain_adaptation import compute_dataset_entropy_stats
from src.cache_manager import EntropyCache
from configs.config import (
    VINDR_IMAGES_ROOT,
    VINDR_CSV,
    RANDOM_SEED,
    TRAIN_VAL_SPLIT
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute entropy statistics from VinDr-Mammo training set"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cache exists"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory for cache files"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="vindr_train",
        help="Name for cached dataset"
    )

    args = parser.parse_args()

    print("="*80)
    print("COMPUTING ENTROPY STATISTICS FOR DOMAIN ADAPTATION")
    print("="*80)

    # Initialize cache
    cache = EntropyCache(cache_dir=args.cache_dir)

    # Check if cache already exists
    if cache.exists(args.dataset_name) and not args.force:
        print(f"\nCache already exists at: {cache.get_cache_path(args.dataset_name)}")
        print("\nLoading existing statistics...")

        stats = cache.load(args.dataset_name)

        print(f"\nEntropy Statistics:")
        print(f"  Mean:     {stats.mean:.4f} bits")
        print(f"  Std:      {stats.std:.4f} bits")
        print(f"  Min:      {stats.min_entropy:.4f} bits")
        print(f"  Max:      {stats.max_entropy:.4f} bits")
        print(f"  Samples:  {stats.n_samples}")
        print(f"  Computed: {stats.computed_date}")

        print(f"\nTarget range for adaptation: [{stats.mean - stats.std:.4f}, {stats.mean + stats.std:.4f}]")

        print("\nTo recompute, run with --force flag:")
        print(f"  python scripts/compute_entropy_stats.py --force")

        return

    # Load VinDr-Mammo dataset
    print("\n[1/3] Loading VinDr-Mammo dataset...")
    print(f"  Images root: {VINDR_IMAGES_ROOT}")
    print(f"  CSV file: {VINDR_CSV}")

    # Create full dataset
    full_dataset = VinDRMammoBinaryDataset(
        images_root=VINDR_IMAGES_ROOT,
        csv_file=VINDR_CSV,
        preprocessor=MammographyPreprocessor(),  # Use regular preprocessor
        transform=None
    )

    # Create breast-level splits to get training subset
    print(f"\n[2/3] Creating breast-level splits (train/val = {1-TRAIN_VAL_SPLIT:.0%}/{TRAIN_VAL_SPLIT:.0%})...")

    train_subset, val_subset = create_breast_level_splits(
        dataset=full_dataset,
        train_ratio=1.0 - TRAIN_VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=True
    )

    print(f"\nTraining set: {len(train_subset.indices)} images")
    print(f"Validation set: {len(val_subset.indices)} images")

    # Compute entropy statistics on TRAINING SET ONLY
    print(f"\n[3/3] Computing entropy statistics on training set...")
    print(f"  This will take ~1-2 hours for full dataset")
    print(f"  Progress will be shown below:\n")

    # Create a temporary dataset with only training indices
    # We need to use AdaptiveMammographyPreprocessor to access grayscale
    adaptive_preprocessor = AdaptiveMammographyPreprocessor(
        apply_adaptation=False  # No adaptation, just need grayscale access
    )

    # Compute statistics
    # Note: We pass the full dataset but will only process training indices
    # The compute function will need access to dataset.samples
    class TrainingOnlyDataset:
        """Wrapper to expose only training samples."""
        def __init__(self, base_dataset, train_indices):
            self.base_dataset = base_dataset
            self.train_indices = list(train_indices)
            self.images_root = base_dataset.images_root

            # Filter samples to training indices only
            self.samples = [base_dataset.samples[i] for i in train_indices]

        def __len__(self):
            return len(self.train_indices)

        def __getitem__(self, idx):
            # Map to original index
            orig_idx = self.train_indices[idx]
            return self.base_dataset[orig_idx]

    train_only_dataset = TrainingOnlyDataset(full_dataset, train_subset.indices)

    # Compute entropy statistics
    entropy_stats = compute_dataset_entropy_stats(
        dataset=train_only_dataset,
        preprocessor=adaptive_preprocessor,
        save_path=None,  # Will cache via cache manager
        verbose=True
    )

    # Set dataset name
    entropy_stats.dataset_name = "VinDr-Mammo-Train"

    # Save to cache
    print(f"\nSaving statistics to cache...")
    cache.save(entropy_stats, args.dataset_name)

    print(f"\n{'='*80}")
    print("COMPUTATION COMPLETE!")
    print(f"{'='*80}")

    print(f"\nEntropy Statistics:")
    print(f"  Mean:     {entropy_stats.mean:.4f} bits")
    print(f"  Std:      {entropy_stats.std:.4f} bits")
    print(f"  Min:      {entropy_stats.min_entropy:.4f} bits")
    print(f"  Max:      {entropy_stats.max_entropy:.4f} bits")
    print(f"  Samples:  {entropy_stats.n_samples}")

    print(f"\nTarget range for adaptation:")
    print(f"  [{entropy_stats.mean - entropy_stats.std:.4f}, {entropy_stats.mean + entropy_stats.std:.4f}] bits")

    print(f"\nCache saved to:")
    print(f"  {cache.get_cache_path(args.dataset_name)}")

    print(f"\nNext steps:")
    print(f"  1. Run zero-shot evaluation with adaptation:")
    print(f"     python scripts/evaluate_zeroshot_adapted.py")
    print(f"\n  2. Or use in your own scripts:")
    print(f"     from src.datasets import create_inbreast_dataset_with_adaptation")
    print(f"     dataset = create_inbreast_dataset_with_adaptation(...)")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
