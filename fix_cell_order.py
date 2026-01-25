"""
Fix the ordering of cells in sections 13-17.
The code cells should come AFTER their markdown headers.
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
backup_path = notebook_path.with_suffix('.ipynb.backup4')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

def get_first_line(cell):
    """Get first line from cell source."""
    source = cell['source']
    if isinstance(source, list):
        source = ''.join(source)
    for line in source.split('\n'):
        if line.strip():
            return line.strip()
    return ""

# Find and reorder cells from index 30 onwards
# Current structure (WRONG):
#   30: ## 13. Test INbreast Dataset Loading (MD)
#   31: from src.datasets import INbreastDataset (CODE for section 13)
#   32: import matplotlib.pyplot as plt (CODE - misplaced)
#   33: from sklearn.metrics... (CODE for section 14)
#   34: print("Current Configuration:") (CODE for section 15)
#   35: from src.domain_adaptation... (CODE for section 16)
#   36: ## 14. Zero-Shot Evaluation (MD)
#   37: ## 15. Entropy-Based Domain Adaptation (MD)
#   38: ## 16. Entropy Statistics (MD)
#   39: ## 17. Final Summary (MD)

# Target structure (CORRECT):
#   30: ## 13. Test INbreast Dataset Loading (MD)
#   31: from src.datasets import INbreastDataset (CODE for section 13)
#   32: ## 14. Zero-Shot Evaluation (MD)
#   33: from sklearn.metrics... (CODE for section 14)
#   34: ## 15. Entropy-Based Domain Adaptation (MD)
#   35: print("Current Configuration:") (CODE for section 15)
#   36: ## 16. Entropy Statistics (MD)
#   37: from src.domain_adaptation... (CODE for section 16)
#   38: ## 17. Final Summary (MD)

# Extract cells 30-39
section13_md = cells[30]
section13_code = cells[31]
misplaced_code = cells[32]  # This is a duplicate from Section 12, remove it
section14_code = cells[33]
section15_code = cells[34]
section16_code = cells[35]
section14_md = cells[36]
section15_md = cells[37]
section16_md = cells[38]
section17_md = cells[39]

# Rebuild in correct order
new_order = [
    section13_md,    # 30: ## 13
    section13_code,  # 31: Section 13 code
    section14_md,    # 32: ## 14
    section14_code,  # 33: Section 14 code
    section15_md,    # 34: ## 15
    section15_code,  # 35: Section 15 code
    section16_md,    # 36: ## 16
    section16_code,  # 37: Section 16 code
    section17_md,    # 38: ## 17
]

# Replace cells 30-39 with new order
cells = cells[:30] + new_order

# Update notebook
nb['cells'] = cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Fixed cell ordering")
print(f"[OK] Removed 1 misplaced cell")
print(f"[OK] Notebook now has {len(cells)} cells")
print(f"[OK] Backup: {backup_path}")
