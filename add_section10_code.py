"""
Add robustness evaluation code to Section 10.
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
backup_path = notebook_path.with_suffix('.ipynb.backup7')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Find Section 10
section10_idx = None
for i, cell in enumerate(cells):
    source = cell['source']
    if isinstance(source, list):
        source = ''.join(source)
    if '## 10. Test Robustness' in source:
        section10_idx = i
        break

print(f"[OK] Found Section 10 at index {section10_idx}")

# Create robustness code cell
robustness_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from src.robustness import RobustnessTester\n",
        "\n",
        "print(\"Testing robustness to intensity perturbations...\\n\")\n",
        "\n",
        "# Create robustness tester\n",
        "tester = RobustnessTester(\n",
        "    brightness_delta=0.1,\n",
        "    contrast_factor=0.1,\n",
        "    noise_std=0.02\n",
        ")\n",
        "\n",
        "# Evaluate robustness on validation set\n",
        "results = tester.evaluate_robustness(model, val_loader, device)\n",
        "\n",
        "print(f\"\\nRobustness Evaluation Results:\")\n",
        "print(f\"  PR-AUC (clean):       {results['pr_auc_standard']:.4f}\")\n",
        "print(f\"  PR-AUC (perturbed):   {results['pr_auc_perturbed']:.4f}\")\n",
        "print(f\"  Degradation:          {results['degradation']:.4f}\")\n",
        "\n",
        "print(f\"\\nInterpretation:\")\n",
        "if results['degradation'] < 0.05:\n",
        "    print(f\"  [EXCELLENT] Model is highly robust to intensity perturbations\")\n",
        "elif results['degradation'] < 0.10:\n",
        "    print(f\"  [GOOD] Model shows good robustness\")\n",
        "elif results['degradation'] < 0.15:\n",
        "    print(f\"  [MODERATE] Model has moderate robustness\")\n",
        "else:\n",
        "    print(f\"  [POOR] Model is sensitive to perturbations\")\n",
        "\n",
        "print(f\"\\nNote: Lower degradation = more robust model\")"
    ]
}

# Insert after Section 10 header
cells.insert(section10_idx + 1, robustness_code)

# Update notebook
nb['cells'] = cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Added robustness code to Section 10")
print(f"[OK] Notebook updated with {len(cells)} cells")
print(f"[OK] Backup: {backup_path}")
