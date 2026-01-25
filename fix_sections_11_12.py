"""
Fix sections 11 and 12:
- Remove duplicate import in Section 11
- Add proper NSGA-III optimization code
- Add proper Pareto front visualization code
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
backup_path = notebook_path.with_suffix('.ipynb.backup8')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Remove duplicate at cell 29
print(f"[OK] Removing duplicate cell at index 29")
del cells[29]

# Now cells[28] is Section 11 header
# cells[29] is Section 12 header

# Create NSGA-III code
nsga3_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from src.optimization import BreastCancerOptimizationProblem\n",
        "from pymoo.algorithms.moo.nsga3 import NSGA3\n",
        "from pymoo.optimize import minimize\n",
        "from pymoo.util.ref_dirs import get_reference_directions\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"NSGA-III OPTIMIZATION (DEMO VERSION)\")\n",
        "print(\"=\"*80)\n",
        "print(\"\\nWarning: Full optimization takes 20-50 hours!\")\n",
        "print(\"This demo uses: 5 population x 3 generations = 15 evaluations\")\n",
        "print(\"For production, use: 20 population x 50 generations = 1000 evaluations\\n\")\n",
        "\n",
        "# Create optimization problem\n",
        "problem = BreastCancerOptimizationProblem(\n",
        "    train_dataset=small_train_dataset,\n",
        "    val_dataset=small_val_dataset,\n",
        "    device=device,\n",
        "    batch_size=4,\n",
        "    num_workers=2,\n",
        "    patience=3,\n",
        "    max_epochs=5,  # Reduced for demo\n",
        "    pos_weight=n_benign/n_malignant,\n",
        "    random_seed=42\n",
        ")\n",
        "\n",
        "# Generate reference directions for 4 objectives\n",
        "ref_dirs = get_reference_directions(\"das-dennis\", 4, n_partitions=3)\n",
        "\n",
        "# Create NSGA-III algorithm\n",
        "algorithm = NSGA3(\n",
        "    ref_dirs=ref_dirs,\n",
        "    pop_size=5  # DEMO: 5 instead of 20\n",
        ")\n",
        "\n",
        "# Run optimization\n",
        "print(\"Starting optimization...\\n\")\n",
        "\n",
        "res = minimize(\n",
        "    problem,\n",
        "    algorithm,\n",
        "    ('n_gen', 3),  # DEMO: 3 instead of 50\n",
        "    verbose=True,\n",
        "    save_history=False\n",
        ")\n",
        "\n",
        "print(f\"\\n[OK] Optimization complete!\")\n",
        "print(f\"  Evaluations: {res.algorithm.n_eval}\")\n",
        "print(f\"  Pareto solutions: {len(res.F)}\")"
    ]
}

# Create Pareto visualization code
pareto_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import matplotlib.pyplot as plt\n",
        "from mpl_toolkits.mplot3d import Axes3D\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"PARETO FRONT SOLUTIONS\")\n",
        "print(\"=\"*80 + \"\\n\")\n",
        "\n",
        "# Display table\n",
        "print(f\"{'ID':<5} {'PR-AUC':>8} {'AUROC':>8} {'Brier':>8} {'Robust':>8}\")\n",
        "print(\"-\" * 45)\n",
        "\n",
        "for i, f in enumerate(res.F):\n",
        "    pr_auc = -f[0]  # Convert back from minimization\n",
        "    auroc = -f[1]\n",
        "    brier = f[2]\n",
        "    robust = f[3]\n",
        "    \n",
        "    print(f\"{i:<5} {pr_auc:>8.4f} {auroc:>8.4f} {brier:>8.4f} {robust:>8.4f}\")\n",
        "\n",
        "# Visualize 2D projections\n",
        "pr_auc = -res.F[:, 0]\n",
        "auroc = -res.F[:, 1]\n",
        "brier = res.F[:, 2]\n",
        "robust = res.F[:, 3]\n",
        "\n",
        "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
        "\n",
        "# PR-AUC vs AUROC\n",
        "axes[0, 0].scatter(pr_auc, auroc, c='blue', s=100)\n",
        "axes[0, 0].set_xlabel('PR-AUC')\n",
        "axes[0, 0].set_ylabel('AUROC')\n",
        "axes[0, 0].set_title('PR-AUC vs AUROC')\n",
        "axes[0, 0].grid(True)\n",
        "\n",
        "# PR-AUC vs Brier\n",
        "axes[0, 1].scatter(pr_auc, brier, c='red', s=100)\n",
        "axes[0, 1].set_xlabel('PR-AUC')\n",
        "axes[0, 1].set_ylabel('Brier Score')\n",
        "axes[0, 1].set_title('PR-AUC vs Brier')\n",
        "axes[0, 1].grid(True)\n",
        "\n",
        "# PR-AUC vs Robustness\n",
        "axes[0, 2].scatter(pr_auc, robust, c='green', s=100)\n",
        "axes[0, 2].set_xlabel('PR-AUC')\n",
        "axes[0, 2].set_ylabel('Robustness Degradation')\n",
        "axes[0, 2].set_title('PR-AUC vs Robustness')\n",
        "axes[0, 2].grid(True)\n",
        "\n",
        "# AUROC vs Brier\n",
        "axes[1, 0].scatter(auroc, brier, c='purple', s=100)\n",
        "axes[1, 0].set_xlabel('AUROC')\n",
        "axes[1, 0].set_ylabel('Brier Score')\n",
        "axes[1, 0].set_title('AUROC vs Brier')\n",
        "axes[1, 0].grid(True)\n",
        "\n",
        "# AUROC vs Robustness\n",
        "axes[1, 1].scatter(auroc, robust, c='orange', s=100)\n",
        "axes[1, 1].set_xlabel('AUROC')\n",
        "axes[1, 1].set_ylabel('Robustness Degradation')\n",
        "axes[1, 1].set_title('AUROC vs Robustness')\n",
        "axes[1, 1].grid(True)\n",
        "\n",
        "# Brier vs Robustness\n",
        "axes[1, 2].scatter(brier, robust, c='brown', s=100)\n",
        "axes[1, 2].set_xlabel('Brier Score')\n",
        "axes[1, 2].set_ylabel('Robustness Degradation')\n",
        "axes[1, 2].set_title('Brier vs Robustness')\n",
        "axes[1, 2].grid(True)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\nNote: This is a DEMO with reduced scale.\")\n",
        "print(\"For production, use scripts/run_nsga3.py with full parameters.\")"
    ]
}

# Insert NSGA-III code after Section 11 header (index 28)
cells.insert(29, nsga3_code)

# Insert Pareto code after Section 12 header (now at index 30)
cells.insert(31, pareto_code)

# Update notebook
nb['cells'] = cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Fixed sections 11 and 12")
print(f"[OK] Removed 1 duplicate cell")
print(f"[OK] Added 2 new code cells")
print(f"[OK] Notebook updated with {len(cells)} cells")
print(f"[OK] Backup: {backup_path}")
