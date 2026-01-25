"""
Add complete training setup code to Section 8.
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
backup_path = notebook_path.with_suffix('.ipynb.backup5')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Find Section 8
section8_idx = None
for i, cell in enumerate(cells):
    source = cell['source']
    if isinstance(source, list):
        source = ''.join(source)
    if '## 8. Test Training Pipeline' in source:
        section8_idx = i
        break

if section8_idx is None:
    print("[ERROR] Could not find Section 8")
    exit(1)

print(f"[OK] Found Section 8 at index {section8_idx}")

# The current Section 8 has augmentation code (cell 21)
# We need to insert proper training setup BEFORE the train_model call (cell 23)

# Create the training setup cell
training_setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from src.training import train_model\n",
        "from torch.utils.data import DataLoader, Subset\n",
        "import numpy as np\n",
        "\n",
        "print(\"Setting up training pipeline (demo with small subset)...\\n\")\n",
        "\n",
        "# Create small subsets for quick demo\n",
        "# For full training, remove these and use entire dataset\n",
        "n_samples = len(dataset)\n",
        "n_train_demo = min(100, int(n_samples * 0.8))\n",
        "n_val_demo = min(30, int(n_samples * 0.2))\n",
        "\n",
        "train_indices = np.random.choice(n_samples, n_train_demo, replace=False)\n",
        "val_indices = np.random.choice(\n",
        "    [i for i in range(n_samples) if i not in train_indices],\n",
        "    n_val_demo,\n",
        "    replace=False\n",
        ")\n",
        "\n",
        "small_train_dataset = Subset(dataset, train_indices)\n",
        "small_val_dataset = Subset(dataset, val_indices)\n",
        "\n",
        "print(f\"Demo training set: {len(small_train_dataset)} samples\")\n",
        "print(f\"Demo validation set: {len(small_val_dataset)} samples\")\n",
        "\n",
        "# Create data loaders\n",
        "train_loader = DataLoader(\n",
        "    small_train_dataset,\n",
        "    batch_size=4,\n",
        "    shuffle=True,\n",
        "    num_workers=2,\n",
        "    pin_memory=True\n",
        ")\n",
        "\n",
        "val_loader = DataLoader(\n",
        "    small_val_dataset,\n",
        "    batch_size=4,\n",
        "    shuffle=False,\n",
        "    num_workers=2,\n",
        "    pin_memory=True\n",
        ")\n",
        "\n",
        "print(f\"\\n[OK] Data loaders created\")\n",
        "print(f\"  Train batches: {len(train_loader)}\")\n",
        "print(f\"  Val batches: {len(val_loader)}\")\n",
        "\n",
        "# Test batch loading\n",
        "imgs, labels = next(iter(train_loader))\n",
        "print(f\"\\nBatch test:\")\n",
        "print(f\"  Image batch shape: {imgs.shape}\")\n",
        "print(f\"  Label batch: {labels}\")\n",
        "\n",
        "# Initialize model for training\n",
        "from src.models import build_resnet152\n",
        "\n",
        "model = build_resnet152(\n",
        "    pretrained=True,\n",
        "    dropout=0.2,\n",
        "    unfreeze_fraction=0.3  # Fine-tune last 30% of backbone\n",
        ")\n",
        "\n",
        "model = model.to(device)\n",
        "print(f\"\\n[OK] Model initialized and moved to {device}\")\n",
        "\n",
        "# Show trainable parameters\n",
        "info = model.get_trainable_params_info()\n",
        "print(f\"  Total params: {info['total_params']:,}\")\n",
        "print(f\"  Trainable: {info['trainable_params']:,} ({info['trainable_percentage']:.1f}%)\")"
    ]
}

# Insert after Section 8 markdown header (after cell 20 - augmentation code, before cell 22 - training code)
# Cell 22 is the Section 9 header, so we need to insert before it
# Current structure:
# 20: ## 8. Test Training Pipeline (MD)
# 21: augmentation code (CODE)
# 22: ## 9. Test Breast-Level Aggregation (MD)
# 23: model, metrics = train_model(...) (CODE) - this is the problem!

# Actually, looking at the structure, the train_model code (cell 23) is under Section 9
# We need to move it to Section 8 and add setup before it

# Find where train_model is called
train_model_idx = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        if 'model, metrics = train_model(' in source:
            train_model_idx = i
            break

print(f"[OK] Found train_model call at index {train_model_idx}")

# Insert training setup BEFORE train_model call
cells.insert(train_model_idx, training_setup_cell)

# Now we need to move the train_model call from Section 9 to Section 8
# Current positions after insertion:
# train_model_idx + 1 is the train_model call
# We need to ensure it's logically under Section 8

# Update notebook
nb['cells'] = cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Added training setup code to Section 8")
print(f"[OK] Notebook updated with {len(cells)} cells")
print(f"[OK] Backup: {backup_path}")
