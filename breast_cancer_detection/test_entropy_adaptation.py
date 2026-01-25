"""
Quick test script to verify entropy adaptation implementation.

Tests:
1. Shannon entropy calculation
2. Adaptive entropy transform
3. EntropyStatistics save/load
4. Cache manager
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.domain_adaptation import (
    calculate_shannon_entropy,
    adaptive_entropy_transform,
    EntropyStatistics
)
from src.cache_manager import EntropyCache


def test_shannon_entropy():
    """Test Shannon entropy calculation."""
    print("="*60)
    print("TEST 1: Shannon Entropy Calculation")
    print("="*60)

    # Test case 1: Uniform image (maximum entropy)
    uniform_img = np.full((100, 100), 128, dtype=np.uint8)
    entropy_uniform = calculate_shannon_entropy(uniform_img)
    print(f"\nUniform image (all pixels = 128):")
    print(f"  Entropy: {entropy_uniform:.4f} bits")
    print(f"  Expected: 0.0 bits (single value)")
    assert entropy_uniform == 0.0, "Uniform image should have 0 entropy"
    print("  ✓ PASS")

    # Test case 2: Binary image (low entropy)
    binary_img = np.random.choice([0, 255], size=(100, 100)).astype(np.uint8)
    entropy_binary = calculate_shannon_entropy(binary_img)
    print(f"\nBinary image (only 0 and 255):")
    print(f"  Entropy: {entropy_binary:.4f} bits")
    print(f"  Expected: ~1.0 bits (2 values)")
    assert 0.8 < entropy_binary < 1.2, "Binary image should have ~1 bit entropy"
    print("  ✓ PASS")

    # Test case 3: Random image (high entropy)
    random_img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    entropy_random = calculate_shannon_entropy(random_img)
    print(f"\nRandom image (all values 0-255):")
    print(f"  Entropy: {entropy_random:.4f} bits")
    print(f"  Expected: >7.0 bits (many values)")
    assert entropy_random > 7.0, "Random image should have high entropy"
    print("  ✓ PASS")

    print("\n✓ All entropy calculation tests passed!\n")


def test_adaptive_transform():
    """Test adaptive entropy transform."""
    print("="*60)
    print("TEST 2: Adaptive Entropy Transform")
    print("="*60)

    # Create test image with known entropy
    test_img = np.random.randint(50, 150, size=(100, 100), dtype=np.uint8)
    initial_entropy = calculate_shannon_entropy(test_img)

    print(f"\nTest image:")
    print(f"  Initial entropy: {initial_entropy:.4f} bits")
    print(f"  Shape: {test_img.shape}")
    print(f"  Range: [{test_img.min()}, {test_img.max()}]")

    # Test 1: Transform to higher entropy
    target_mean = initial_entropy + 0.5
    target_std = 0.2

    print(f"\nTest 1: Increase entropy to {target_mean:.4f} ± {target_std:.4f}")
    adapted_img, metrics = adaptive_entropy_transform(
        test_img.copy(),
        target_mean=target_mean,
        target_std=target_std,
        max_iterations=50,
        tolerance=0.1
    )

    print(f"  Initial entropy: {metrics['initial_entropy']:.4f}")
    print(f"  Final entropy: {metrics['final_entropy']:.4f}")
    print(f"  Iterations: {metrics['iterations']}")
    print(f"  Converged: {metrics['converged']}")

    # Verify output
    assert adapted_img.shape == test_img.shape, "Shape should be preserved"
    assert adapted_img.dtype == np.uint8, "Dtype should be uint8"
    assert adapted_img.min() >= 0 and adapted_img.max() <= 255, "Range should be [0, 255]"
    assert abs(metrics['final_entropy'] - target_mean) <= target_std, "Should converge to target"
    print("  ✓ PASS")

    # Test 2: Transform to lower entropy
    test_img2 = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    initial_entropy2 = calculate_shannon_entropy(test_img2)
    target_mean2 = initial_entropy2 - 0.5

    print(f"\nTest 2: Decrease entropy from {initial_entropy2:.4f} to {target_mean2:.4f}")
    adapted_img2, metrics2 = adaptive_entropy_transform(
        test_img2.copy(),
        target_mean=target_mean2,
        target_std=0.2,
        max_iterations=50,
        tolerance=0.1
    )

    print(f"  Final entropy: {metrics2['final_entropy']:.4f}")
    print(f"  Iterations: {metrics2['iterations']}")
    assert metrics2['final_entropy'] < initial_entropy2, "Entropy should decrease"
    print("  ✓ PASS")

    print("\n✓ All adaptive transform tests passed!\n")


def test_entropy_statistics():
    """Test EntropyStatistics save/load."""
    print("="*60)
    print("TEST 3: EntropyStatistics Save/Load")
    print("="*60)

    # Create test statistics
    stats = EntropyStatistics(
        mean=6.85,
        std=0.42,
        min_entropy=5.92,
        max_entropy=7.78,
        n_samples=1000,
        dataset_name="Test Dataset"
    )

    print(f"\nCreated EntropyStatistics:")
    print(f"  Mean: {stats.mean:.4f}")
    print(f"  Std: {stats.std:.4f}")
    print(f"  Min: {stats.min_entropy:.4f}")
    print(f"  Max: {stats.max_entropy:.4f}")
    print(f"  N: {stats.n_samples}")
    print(f"  Dataset: {stats.dataset_name}")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        print(f"\nSaving to: {temp_path}")
        stats.save(temp_path)
        print("  ✓ Saved")

        # Load back
        print(f"Loading from: {temp_path}")
        loaded_stats = EntropyStatistics.load(temp_path)
        print("  ✓ Loaded")

        # Verify
        assert loaded_stats.mean == stats.mean, "Mean should match"
        assert loaded_stats.std == stats.std, "Std should match"
        assert loaded_stats.min_entropy == stats.min_entropy, "Min should match"
        assert loaded_stats.max_entropy == stats.max_entropy, "Max should match"
        assert loaded_stats.n_samples == stats.n_samples, "N samples should match"
        assert loaded_stats.dataset_name == stats.dataset_name, "Dataset name should match"

        print("\nVerification:")
        print(f"  Mean: {loaded_stats.mean:.4f} == {stats.mean:.4f} ✓")
        print(f"  Std: {loaded_stats.std:.4f} == {stats.std:.4f} ✓")
        print(f"  Dataset: {loaded_stats.dataset_name} == {stats.dataset_name} ✓")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n✓ All save/load tests passed!\n")


def test_cache_manager():
    """Test EntropyCache."""
    print("="*60)
    print("TEST 4: EntropyCache Manager")
    print("="*60)

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary cache dir: {temp_dir}")

        # Create cache
        cache = EntropyCache(cache_dir=temp_dir)
        print("  ✓ Cache created")

        # Check non-existent dataset
        assert not cache.exists("test_dataset"), "Should not exist yet"
        print("  ✓ exists() works for missing dataset")

        # Create and save statistics
        stats = EntropyStatistics(
            mean=6.5,
            std=0.4,
            min_entropy=5.8,
            max_entropy=7.2,
            n_samples=500,
            dataset_name="Test"
        )

        cache.save(stats, "test_dataset")
        print("  ✓ Saved to cache")

        # Check existence
        assert cache.exists("test_dataset"), "Should exist now"
        print("  ✓ exists() works for saved dataset")

        # Load from cache
        loaded_stats = cache.load("test_dataset")
        assert loaded_stats.mean == stats.mean, "Loaded stats should match"
        print("  ✓ Loaded from cache")

        # List cached datasets
        cached_list = cache.list_cached_datasets()
        assert "test_dataset" in cached_list, "Should be in list"
        print(f"  ✓ Listed cached datasets: {cached_list}")

        # Get cache info
        info = cache.get_cache_info("test_dataset")
        print(f"\nCache info:")
        print(f"  Dataset: {info['dataset_name']}")
        print(f"  Mean entropy: {info['mean_entropy']:.4f}")
        print(f"  N samples: {info['n_samples']}")
        print(f"  File size: {info['file_size_bytes']} bytes")
        print("  ✓ Cache info retrieved")

        # Invalidate cache
        assert cache.invalidate("test_dataset"), "Should successfully delete"
        assert not cache.exists("test_dataset"), "Should not exist after invalidation"
        print("  ✓ Cache invalidation works")

    print("\n✓ All cache manager tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ENTROPY ADAPTATION IMPLEMENTATION TESTS")
    print("="*60 + "\n")

    try:
        test_shannon_entropy()
        test_adaptive_transform()
        test_entropy_statistics()
        test_cache_manager()

        print("="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nImplementation verified successfully!")
        print("\nNext steps:")
        print("  1. Run compute_entropy_stats.py to generate cache from VinDr training set")
        print("  2. Run evaluate_zeroshot_adapted.py to compare baseline vs adapted")
        print("  3. Check breast_cancer_colab_demo.ipynb for full demo")
        print("\n" + "="*60 + "\n")

        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
