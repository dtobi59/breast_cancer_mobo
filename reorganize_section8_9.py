"""
Reorganize sections 8 and 9:
- Section 8 should have: augmentation test + training setup + training demo
- Section 9 should have: breast-level aggregation test
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
backup_path = notebook_path.with_suffix('.ipynb.backup6')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Current structure (from verification):
# 20: ## 8. Test Training Pipeline (MD)
# 21: augmentation code (CODE)
# 22: ## 9. Test Breast-Level Aggregation (MD)
# 23: training setup code (CODE) - from src.training import train_model
# 24: train_model call (CODE) - model, metrics = train_model(...)
# 25: ## 10. Test Robustness Evaluation (MD)
# 26: aggregation code (CODE) - print("\nTesting breast-level aggregation...")

# Target structure:
# 20: ## 8. Test Training Pipeline (MD)
# 21: augmentation code (CODE)
# 22: training setup code (CODE)
# 23: train_model call (CODE)
# 24: ## 9. Test Breast-Level Aggregation (MD)
# 25: aggregation code (CODE)
# 26: ## 10. Test Robustness Evaluation (MD)

# Extract cells
section8_md = cells[20]       # ## 8
augmentation_code = cells[21]  # augmentation test
section9_md = cells[22]        # ## 9
training_setup = cells[23]     # training setup (new)
train_model_code = cells[24]   # train_model call
section10_md = cells[25]       # ## 10
aggregation_code = cells[26]   # aggregation test

# Rebuild in correct order
new_cells = cells[:20] + [
    section8_md,
    augmentation_code,
    training_setup,
    train_model_code,
    section9_md,
    aggregation_code,
    section10_md,
] + cells[27:]

# Update notebook
nb['cells'] = new_cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Reorganized sections 8-10")
print(f"[OK] Notebook has {len(new_cells)} cells")
print(f"[OK] Backup: {backup_path}")
