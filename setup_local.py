"""
Local Setup Script for Breast Cancer Detection Project

Run this once to ensure all paths and dependencies are configured correctly
for local development (outside of Google Colab).

Usage:
    python setup_local.py
"""

import sys
import os
from pathlib import Path

def setup_project_paths():
    """Add project directories to Python path."""

    print("="*80)
    print("BREAST CANCER DETECTION - LOCAL SETUP")
    print("="*80)

    # Get project root (where this script is located)
    project_root = Path(__file__).parent.resolve()
    print(f"\nProject root: {project_root}")

    # Path to breast_cancer_detection module
    bc_detection_path = project_root / "breast_cancer_detection"

    if not bc_detection_path.exists():
        print(f"\n[ERROR] breast_cancer_detection directory not found!")
        print(f"Expected at: {bc_detection_path}")
        return False

    print(f"breast_cancer_detection: {bc_detection_path}")

    # Check src directory
    src_path = bc_detection_path / "src"
    if not src_path.exists():
        print(f"\n[ERROR] src directory not found!")
        print(f"Expected at: {src_path}")
        return False

    print(f"src: {src_path}")

    # Add to path
    bc_path_str = str(bc_detection_path)
    if bc_path_str not in sys.path:
        sys.path.insert(0, bc_path_str)
        print(f"\n[OK] Added to sys.path: {bc_path_str}")
    else:
        print(f"\n[OK] Already in sys.path: {bc_path_str}")

    return True


def verify_imports():
    """Verify that all required modules can be imported."""

    print("\n" + "="*80)
    print("VERIFYING IMPORTS")
    print("="*80)

    modules_to_test = [
        ("src.preprocessing", "MammographyPreprocessor"),
        ("src.datasets", "VinDRMammoBinaryDataset"),
        ("src.optimization", "BreastCancerOptimizationProblem"),
        ("src.surrogate_optimizer", "MultiObjectiveGPSurrogate"),
        ("src.acquisition", "get_acquisition_function"),
        ("src.models", "build_resnet152"),
        ("src.training", "train_model"),
        ("src.evaluation", "compute_metrics"),
    ]

    all_success = True

    for module_name, class_name in modules_to_test:
        try:
            print(f"\n  Importing {module_name}.{class_name}...", end=" ")
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print("[OK]")
        except ImportError as e:
            print(f"[FAIL]")
            print(f"    Error: {e}")
            all_success = False
        except AttributeError as e:
            print(f"[FAIL]")
            print(f"    Error: {class_name} not found in {module_name}")
            all_success = False

    return all_success


def check_dependencies():
    """Check if required dependencies are installed."""

    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)

    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "sklearn",
        "pymoo",
        "pydicom",
        "cv2",
        "PIL",
        "scipy",
    ]

    missing = []

    for package in required_packages:
        try:
            print(f"\n  {package}...", end=" ")
            if package == "cv2":
                __import__("cv2")
            elif package == "PIL":
                __import__("PIL")
            elif package == "sklearn":
                __import__("sklearn")
            else:
                __import__(package)
            print("[OK]")
        except ImportError:
            print("[MISSING]")
            missing.append(package)

    if missing:
        print(f"\n[WARNING] Missing packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print("  pip install torch torchvision")
        print("  pip install numpy pandas scikit-learn pymoo")
        print("  pip install pydicom opencv-python pillow scipy scikit-image")
        return False

    return True


def main():
    """Run setup and verification."""

    # Setup paths
    if not setup_project_paths():
        print("\n[FAILED] Path setup failed")
        return False

    # Check dependencies
    deps_ok = check_dependencies()

    # Verify imports
    imports_ok = verify_imports()

    print("\n" + "="*80)
    print("SETUP COMPLETE")
    print("="*80)

    if deps_ok and imports_ok:
        print("\n[SUCCESS] All checks passed!")
        print("\nYou can now run optimization scripts:")
        print("  python run_local_optimization.py")
        return True
    else:
        print("\n[INCOMPLETE] Some checks failed")
        if not deps_ok:
            print("  - Install missing dependencies")
        if not imports_ok:
            print("  - Fix import errors")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
