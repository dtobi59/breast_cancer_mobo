"""
Add missing code cells to Section 5 (Test Dataset Loading).
"""

import json
from pathlib import Path
import shutil

# Read the notebook
notebook_path = Path("breast_cancer_colab_demo.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Create backup
backup_path = notebook_path.with_suffix('.ipynb.backup2')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Find Section 5 markdown cell (index 14)
section5_idx = None
for i, cell in enumerate(cells):
    source = cell['source']
    if isinstance(source, list):
        source = ''.join(source)
    if '## 5. Test Dataset Loading' in source:
        section5_idx = i
        break

if section5_idx is None:
    print("[ERROR] Could not find Section 5")
    exit(1)

print(f"[OK] Found Section 5 at index {section5_idx}")

# Create the missing code cell
dataset_loading_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from src.datasets import VinDRMammoBinaryDataset, create_breast_level_splits\n",
        "\n",
        "# Load dataset with configured preprocessor\n",
        "print(\"Loading VinDr-Mammo dataset...\")\n",
        "print(f\"Preprocessor: {'Adaptive' if USE_ADAPTIVE_PREPROCESSING else 'Standard'}\")\n",
        "\n",
        "dataset = VinDRMammoBinaryDataset(\n",
        "    images_root=VINDR_IMAGES_ROOT,\n",
        "    csv_file=VINDR_CSV,\n",
        "    preprocessor=preprocessor\n",
        ")\n",
        "\n",
        "print(f\"\\n[OK] Dataset loaded: {len(dataset)} samples\")\n",
        "\n",
        "# Get dataset statistics\n",
        "labels = []\n",
        "for i in range(len(dataset)):\n",
        "    _, label = dataset[i]\n",
        "    labels.append(label.item())\n",
        "\n",
        "n_benign = sum([1 for l in labels if l == 0])\n",
        "n_malignant = sum([1 for l in labels if l == 1])\n",
        "\n",
        "print(f\"\\nDataset Statistics:\")\n",
        "print(f\"  Benign: {n_benign}\")\n",
        "print(f\"  Malignant: {n_malignant}\")\n",
        "print(f\"  Class ratio: {n_benign/n_malignant:.2f}:1\")"
    ]
}

# Insert after Section 5 header
cells.insert(section5_idx + 1, dataset_loading_cell)

# Update notebook
nb['cells'] = cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Added dataset loading code to Section 5")
print(f"[OK] Notebook updated with {len(cells)} cells (was {len(cells)-1})")
print(f"[OK] Backup: {backup_path}")
