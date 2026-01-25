"""
Reorganize breast_cancer_colab_demo.ipynb to have a logical flow.
"""

import json
from pathlib import Path

# Read the notebook
notebook_path = Path("breast_cancer_colab_demo.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Logical order for the notebook
logical_order = [
    # Header
    "# Breast Cancer Detection - Google Colab Demo",

    # Section 1: Environment Setup
    "## 1. Environment Setup",
    "!nvidia-smi",
    "!pip install",
    "import os",

    # Section 2: Mount Drive and Paths
    "## 2. Mount Google Drive",
    "from google.colab import drive",
    "PROJECT_PATH =",
    "VINDR_IMAGES_ROOT =",

    # Section 3: Preprocessing Configuration
    "## 3. Preprocessing Configuration",
    "USE_ADAPTIVE_PREPROCESSING =",

    # Section 4: Create Preprocessor
    "## 4. Test Preprocessing Pipeline",
    "if USE_ADAPTIVE_PREPROCESSING:",
    "sample_dicom =",

    # Section 5: Dataset Loading
    "## 5. Test Dataset Loading",
    "from src.datasets import VinDRMammoBinaryDataset",
    "train_dataset, val_dataset = create_breast_level_splits",
    "img, label = dataset[0]",

    # Section 6: Model Building
    "## 6. Test Model Building",
    "from src.models import build_resnet152",

    # Section 7: Augmentation
    "## 7. Test Augmentation",
    "from src.augmentations import get_augmentation",

    # Section 8: Training Pipeline
    "## 8. Test Training Pipeline",
    "from src.training import train_model",
    "model, metrics = train_model",

    # Section 9: Breast-Level Aggregation
    "## 9. Test Breast-Level Aggregation",
    "from src.evaluation import noisy_or_aggregation",
    "y_true_breast, y_probs_breast",

    # Section 10: Robustness
    "## 10. Test Robustness Evaluation",
    "from src.robustness import RobustnessTester",

    # Section 11: NSGA-III
    "## 11. NSGA-III Optimization",
    "from src.optimization import BreastCancerOptimizationProblem",
    "res = minimize",
    "print(f\"{'ID':<5}",

    # Section 12: Visualize Pareto
    "## 12. Visualize Pareto Front",
    "from mpl_toolkits.mplot3d import Axes3D",

    # Section 13: INbreast Loading
    "## 13. Test INbreast Dataset Loading",
    "from src.datasets import INbreastDataset",

    # Section 14: Summary (first summary)
    "## 14. Summary & Next Steps",

    # Section 15: Zero-Shot Evaluation
    "## 15. Zero-Shot Evaluation on INbreast",
    "print(\"=\"*80)\nprint(\"ZERO-SHOT EVALUATION",
    "# Visualize zero-shot performance comparison",

    # Section 16: Entropy Statistics
    "## 16. Entropy Statistics",
    "from src.domain_adaptation import compute_dataset_entropy_stats",

    # Section 17: Comparative Analysis
    "from src.datasets import create_inbreast_dataset_with_adaptation",
    "# Visualize baseline vs adapted",

    # Section 18: Final Summary
    "## 17. Final Summary",
]

# Helper function to identify cells
def get_cell_key(cell):
    """Get identifying key for a cell."""
    if cell['cell_type'] == 'markdown':
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        # Return first non-empty line
        for line in source.split('\n'):
            if line.strip():
                return line.strip()[:70]
    else:  # code
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        # Return first meaningful line
        for line in source.split('\n'):
            if line.strip() and not line.strip().startswith('#'):
                return line.strip()[:70]
    return ""

# Print current structure
print("="*80)
print("CURRENT NOTEBOOK STRUCTURE")
print("="*80)
for i, cell in enumerate(cells):
    key = get_cell_key(cell)
    cell_type = "MD" if cell['cell_type'] == 'markdown' else "CODE"
    print(f"{i:3d}. [{cell_type:4s}] {key}")

print("\n" + "="*80)
print(f"Total cells: {len(cells)}")
print("="*80)

# Identify duplicates
seen = {}
duplicates = []
for i, cell in enumerate(cells):
    key = get_cell_key(cell)
    if key and key in seen:
        duplicates.append((i, seen[key], key))
    else:
        seen[key] = i

if duplicates:
    print("\n" + "="*80)
    print("DUPLICATE CELLS FOUND")
    print("="*80)
    for curr_idx, orig_idx, key in duplicates:
        print(f"Cell {curr_idx} duplicates cell {orig_idx}:")
        print(f"  {key}")
