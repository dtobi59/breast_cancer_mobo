"""
Fix for SymPy/PyTorch compatibility issue.

Error: AttributeError: module 'sympy' has no attribute 'printing'

This script diagnoses and fixes the version incompatibility.
"""

import subprocess
import sys

def check_versions():
    """Check current versions of PyTorch and SymPy."""
    print("="*80)
    print("CHECKING VERSIONS")
    print("="*80)

    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
    except ImportError:
        print("\nPyTorch: NOT INSTALLED")

    try:
        import sympy
        print(f"SymPy version: {sympy.__version__}")
        print(f"SymPy has 'printing': {hasattr(sympy, 'printing')}")
    except ImportError:
        print("\nSymPy: NOT INSTALLED")

    print("\n" + "="*80)


def fix_sympy():
    """Upgrade SymPy to compatible version."""
    print("\nFIXING SYMPY COMPATIBILITY")
    print("="*80)

    print("\nUpgrading SymPy to latest compatible version...")

    commands = [
        # Uninstall old version
        [sys.executable, "-m", "pip", "uninstall", "-y", "sympy"],
        # Install compatible version (1.12+ works with PyTorch 2.0+)
        [sys.executable, "-m", "pip", "install", "sympy>=1.12"],
    ]

    for cmd in commands:
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        else:
            print("Success!")

    return True


def verify_fix():
    """Verify that the fix worked."""
    print("\n" + "="*80)
    print("VERIFYING FIX")
    print("="*80)

    # Force reimport
    if 'sympy' in sys.modules:
        del sys.modules['sympy']
    if 'torch' in sys.modules:
        del sys.modules['torch']

    try:
        print("\n1. Testing SymPy import...")
        import sympy
        print(f"   SymPy version: {sympy.__version__}")
        print(f"   Has 'printing': {hasattr(sympy, 'printing')}")

        if not hasattr(sympy, 'printing'):
            print("   [FAIL] Still no 'printing' module")
            return False

        print("   [OK]")

        print("\n2. Testing PyTorch import...")
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print("   [OK]")

        print("\n3. Testing torch.utils._sympy...")
        from torch.utils import _sympy
        print("   [OK]")

        print("\n" + "="*80)
        print("[SUCCESS] Fix verified!")
        print("="*80)
        return True

    except Exception as e:
        print(f"\n[FAIL] {e}")
        return False


def main():
    """Main execution."""
    print("="*80)
    print("SYMPY/PYTORCH COMPATIBILITY FIX")
    print("="*80)

    # Check current state
    check_versions()

    # Apply fix
    if fix_sympy():
        # Verify
        if verify_fix():
            print("\n[COMPLETE] You can now import PyTorch modules")
            print("\nNext steps:")
            print("  - Restart your Python kernel/runtime")
            print("  - Re-run your notebook")
            return True

    print("\n[FAILED] Manual intervention required")
    print("\nTry manually:")
    print("  pip uninstall sympy")
    print("  pip install 'sympy>=1.12'")
    print("  # Then restart Python")

    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
