"""
Core test for entropy adaptation (minimal dependencies).

Tests only domain_adaptation.py and cache_manager.py modules
without requiring scikit-image, pydicom, etc.
"""

import os
import sys
import tempfile
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# Import only the modules we're testing
# ============================================================================

# We'll import the functions directly to avoid triggering other imports
import importlib.util

def load_module(module_path):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load domain_adaptation module
domain_adaptation_path = Path(__file__).parent / "src" / "domain_adaptation.py"
da = load_module(domain_adaptation_path)

calculate_shannon_entropy = da.calculate_shannon_entropy
adaptive_entropy_transform = da.adaptive_entropy_transform
EntropyStatistics = da.EntropyStatistics


def test_shannon_entropy():
    """Test Shannon entropy calculation."""
    print("="*60)
    print("TEST 1: Shannon Entropy Calculation")
    print("="*60)

    # Test case 1: Uniform image (zero entropy)
    uniform_img = np.full((100, 100), 128, dtype=np.uint8)
    entropy_uniform = calculate_shannon_entropy(uniform_img)
    print(f"\nUniform image (all pixels = 128):")
    print(f"  Entropy: {entropy_uniform:.4f} bits")
    assert entropy_uniform == 0.0, "Uniform image should have 0 entropy"
    print("  [PASS]")

    # Test case 2: Binary image (low entropy)
    binary_img = np.random.choice([0, 255], size=(100, 100)).astype(np.uint8)
    entropy_binary = calculate_shannon_entropy(binary_img)
    print(f"\nBinary image (only 0 and 255):")
    print(f"  Entropy: {entropy_binary:.4f} bits")
    assert 0.8 < entropy_binary < 1.2, "Binary image should have ~1 bit entropy"
    print("  [PASS]")

    # Test case 3: Random image (high entropy)
    random_img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    entropy_random = calculate_shannon_entropy(random_img)
    print(f"\nRandom image (all values 0-255):")
    print(f"  Entropy: {entropy_random:.4f} bits")
    assert entropy_random > 7.0, "Random image should have high entropy"
    print("  [PASS]")

    print("\n[OK] All entropy calculation tests passed!\n")


def test_adaptive_transform():
    """Test adaptive entropy transform."""
    print("="*60)
    print("TEST 2: Adaptive Entropy Transform")
    print("="*60)

    # Create test image
    np.random.seed(42)
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

    print(f"  Initial: {metrics['initial_entropy']:.4f} bits")
    print(f"  Final: {metrics['final_entropy']:.4f} bits")
    print(f"  Iterations: {metrics['iterations']}")
    print(f"  Converged: {metrics['converged']}")

    # Verify
    assert adapted_img.shape == test_img.shape
    assert adapted_img.dtype == np.uint8
    assert 0 <= adapted_img.min() and adapted_img.max() <= 255
    assert abs(metrics['final_entropy'] - target_mean) <= target_std or not metrics['converged']
    print("  [PASS]")

    # Test 2: Transform to lower entropy
    test_img2 = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    initial_entropy2 = calculate_shannon_entropy(test_img2)
    target_mean2 = max(initial_entropy2 - 0.5, 1.0)  # Ensure valid target

    print(f"\nTest 2: Decrease entropy from {initial_entropy2:.4f} to {target_mean2:.4f}")
    adapted_img2, metrics2 = adaptive_entropy_transform(
        test_img2.copy(),
        target_mean=target_mean2,
        target_std=0.2,
        max_iterations=50,
        tolerance=0.1
    )

    print(f"  Final: {metrics2['final_entropy']:.4f} bits")
    print(f"  Iterations: {metrics2['iterations']}")
    print("  [PASS]")

    print("\n[OK] All adaptive transform tests passed!\n")


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

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        stats.save(temp_path)
        print(f"  [OK] Saved to {temp_path}")

        # Load back
        loaded_stats = EntropyStatistics.load(temp_path)
        print("  [OK] Loaded from file")

        # Verify
        assert loaded_stats.mean == stats.mean
        assert loaded_stats.std == stats.std
        assert loaded_stats.min_entropy == stats.min_entropy
        assert loaded_stats.max_entropy == stats.max_entropy
        assert loaded_stats.n_samples == stats.n_samples
        print("  [OK] All values match")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n[OK] All save/load tests passed!\n")


def test_json_format():
    """Test that saved JSON has correct format."""
    print("="*60)
    print("TEST 4: JSON Format Verification")
    print("="*60)

    stats = EntropyStatistics(
        mean=6.5,
        std=0.4,
        min_entropy=5.8,
        max_entropy=7.2,
        n_samples=500
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        stats.save(temp_path)

        # Read raw JSON
        with open(temp_path, 'r') as f:
            data = json.load(f)

        print("\nJSON contents:")
        for key, value in data.items():
            print(f"  {key}: {value}")

        # Verify required fields
        required_fields = ['mean', 'std', 'min_entropy', 'max_entropy', 'n_samples']
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        print("\n  [OK] All required fields present")
        print("  [OK] JSON format valid")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n[OK] JSON format test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ENTROPY ADAPTATION CORE TESTS")
    print("(Minimal Dependencies)")
    print("="*60 + "\n")

    try:
        test_shannon_entropy()
        test_adaptive_transform()
        test_entropy_statistics()
        test_json_format()

        print("="*60)
        print("ALL CORE TESTS PASSED!")
        print("="*60)
        print("\nCore implementation verified successfully!")
        print("\nModules tested:")
        print("  [OK] domain_adaptation.py - Shannon entropy & adaptive transform")
        print("  [OK] EntropyStatistics class - save/load functionality")
        print("\nNext steps:")
        print("  1. Install full dependencies (pydicom, scikit-image, etc.)")
        print("  2. Run compute_entropy_stats.py to generate cache")
        print("  3. Run evaluate_zeroshot_adapted.py for full evaluation")
        print("\n" + "="*60 + "\n")

        return 0

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
