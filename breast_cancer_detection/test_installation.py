"""
Quick test script to verify installation and basic functionality.

Run this script to ensure all components are working correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        # Core libraries
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import cv2
        import pydicom
        from sklearn.metrics import roc_auc_score
        import pymoo

        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ NumPy {np.__version__}")
        print(f"  ✓ Pandas {pd.__version__}")
        print(f"  ✓ OpenCV {cv2.__version__}")
        print(f"  ✓ pydicom {pydicom.__version__}")
        print(f"  ✓ pymoo available")

        # Project modules
        from src.preprocessing import MammographyPreprocessor
        from src.datasets import VinDRMammoBinaryDataset, INbreastDataset
        from src.augmentations import IntensityAugmentation, get_augmentation
        from src.models import build_resnet152
        from src.training import Trainer, train_model
        from src.evaluation import noisy_or_aggregation
        from src.robustness import RobustnessTester
        from src.optimization import BreastCancerOptimizationProblem

        print("  ✓ All project modules imported successfully")

        return True

    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_preprocessing():
    """Test preprocessing module."""
    print("\nTesting preprocessing...")

    try:
        from src.preprocessing import MammographyPreprocessor

        preprocessor = MammographyPreprocessor()
        assert preprocessor.target_width == 720
        assert preprocessor.target_height == 480

        print("  ✓ Preprocessor initialized correctly")
        return True

    except Exception as e:
        print(f"  ✗ Preprocessing test failed: {e}")
        return False


def test_model():
    """Test model building."""
    print("\nTesting model building...")

    try:
        import torch
        from src.models import build_resnet152

        # Test different configurations
        configs = [
            (0.0, "Feature extraction only"),
            (0.5, "Partial fine-tuning"),
            (1.0, "Full fine-tuning")
        ]

        for unfreeze_frac, desc in configs:
            model = build_resnet152(
                pretrained=False,  # Don't download weights for test
                dropout=0.2,
                unfreeze_fraction=unfreeze_frac
            )

            info = model.get_trainable_params_info()
            print(f"  ✓ {desc}: {info['trainable_percentage']:.1f}% trainable")

            # Test forward pass
            test_input = torch.randn(2, 3, 480, 720)
            output = model(test_input)
            assert output.shape == (2,), f"Expected shape (2,), got {output.shape}"

        print("  ✓ All model configurations work correctly")
        return True

    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_augmentation():
    """Test augmentation."""
    print("\nTesting augmentation...")

    try:
        import torch
        from src.augmentations import get_augmentation

        img = torch.rand(3, 480, 720)

        for strength in [0.0, 0.5, 1.0]:
            aug = get_augmentation(strength)
            img_aug = aug(img.clone())

            assert img_aug.shape == img.shape
            assert img_aug.min() >= 0.0
            assert img_aug.max() <= 1.0

        print("  ✓ Augmentation working for all strengths")
        return True

    except Exception as e:
        print(f"  ✗ Augmentation test failed: {e}")
        return False


def test_noisy_or():
    """Test Noisy-OR aggregation."""
    print("\nTesting Noisy-OR aggregation...")

    try:
        import numpy as np
        from src.evaluation import noisy_or_aggregation

        # Test formula: p = 1 - (1-p1)(1-p2)
        test_cases = [
            ([0.2, 0.3], 0.44),
            ([0.5, 0.5], 0.75),
            ([0.0, 0.0], 0.0),
            ([1.0, 1.0], 1.0),
        ]

        for probs, expected in test_cases:
            result = noisy_or_aggregation(probs)
            assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"

        print("  ✓ Noisy-OR formula verified")
        return True

    except Exception as e:
        print(f"  ✗ Noisy-OR test failed: {e}")
        return False


def test_optimization_problem():
    """Test optimization problem setup."""
    print("\nTesting optimization problem...")

    try:
        from src.optimization import BreastCancerOptimizationProblem

        # Just test that we can create the problem
        # (won't run actual optimization without data)
        print("  ✓ Optimization problem class available")
        print("  ℹ Full test requires dataset (skipped)")
        return True

    except Exception as e:
        print(f"  ✗ Optimization test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("Breast Cancer Detection - Installation Test")
    print("="*80)

    tests = [
        ("Imports", test_imports),
        ("Preprocessing", test_preprocessing),
        ("Model", test_model),
        ("Augmentation", test_augmentation),
        ("Noisy-OR", test_noisy_or),
        ("Optimization", test_optimization_problem),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))

    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Installation is correct.")
        print("\nNext steps:")
        print("1. Configure data paths in configs/config.py")
        print("2. Run: python scripts/train_single_model.py (test)")
        print("3. Run: python scripts/run_nsga3.py (full optimization)")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
