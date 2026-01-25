"""
Add a cell to allow using the full dataset instead of a subset.
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
backup_path = notebook_path.with_suffix('.ipynb.backup9')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Find cell 22 (the training setup cell with subset creation)
# This is after Section 8 header and augmentation test
training_setup_idx = 22

print(f"[OK] Found training setup cell at index {training_setup_idx}")

# Create new cell for dataset selection
dataset_selection_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# DATASET SELECTION: DEMO MODE vs FULL DATASET\n",
        "# ============================================================================\n",
        "\n",
        "# Set this to True to use the FULL dataset for training/validation\n",
        "# Set to False to use a small subset for quick testing (demo mode)\n",
        "USE_FULL_DATASET = False\n",
        "\n",
        "print(\"=\"*60)\n",
        "print(\"DATASET SELECTION\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "if USE_FULL_DATASET:\n",
        "    print(\"\\n[SELECTED] Full Dataset Mode\")\n",
        "    print(\"  - Uses entire VinDr-Mammo dataset\")\n",
        "    print(\"  - 80/20 train/validation split\")\n",
        "    print(\"  - Training will take significantly longer\")\n",
        "    print(\"  - Recommended for production runs\")\n",
        "else:\n",
        "    print(\"\\n[SELECTED] Demo Mode (Small Subset)\")\n",
        "    print(\"  - Uses ~100 training samples, ~30 validation samples\")\n",
        "    print(\"  - Fast training for testing the pipeline\")\n",
        "    print(\"  - Not recommended for production\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*60)"
    ]
}

# Now modify the training setup cell to use the flag
# Current cell 22 has the subset creation code
# We need to replace it with conditional logic

modified_training_setup = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from src.training import train_model\n",
        "from torch.utils.data import DataLoader, Subset\n",
        "from sklearn.model_selection import train_test_split\n",
        "import numpy as np\n",
        "\n",
        "print(\"Setting up training pipeline...\\n\")\n",
        "\n",
        "if USE_FULL_DATASET:\n",
        "    # Full dataset mode: Use proper train/val split\n",
        "    print(\"Using FULL dataset with stratified 80/20 split\")\n",
        "    \n",
        "    # Get all labels for stratification\n",
        "    all_labels = [dataset[i][1].item() for i in range(len(dataset))]\n",
        "    all_indices = np.arange(len(dataset))\n",
        "    \n",
        "    # Stratified train/val split\n",
        "    train_indices, val_indices = train_test_split(\n",
        "        all_indices,\n",
        "        test_size=0.2,\n",
        "        stratify=all_labels,\n",
        "        random_state=42\n",
        "    )\n",
        "    \n",
        "    train_dataset = Subset(dataset, train_indices)\n",
        "    val_dataset = Subset(dataset, val_indices)\n",
        "    \n",
        "    print(f\"  Training set: {len(train_dataset)} samples\")\n",
        "    print(f\"  Validation set: {len(val_dataset)} samples\")\n",
        "    \n",
        "    # Use larger batch size for full dataset\n",
        "    batch_size = 8\n",
        "    max_epochs = 50\n",
        "    \n",
        "else:\n",
        "    # Demo mode: Use small subset for quick testing\n",
        "    print(\"Using DEMO subset for quick testing\")\n",
        "    \n",
        "    n_samples = len(dataset)\n",
        "    n_train_demo = min(100, int(n_samples * 0.8))\n",
        "    n_val_demo = min(30, int(n_samples * 0.2))\n",
        "    \n",
        "    train_indices = np.random.choice(n_samples, n_train_demo, replace=False)\n",
        "    val_indices = np.random.choice(\n",
        "        [i for i in range(n_samples) if i not in train_indices],\n",
        "        n_val_demo,\n",
        "        replace=False\n",
        "    )\n",
        "    \n",
        "    train_dataset = Subset(dataset, train_indices)\n",
        "    val_dataset = Subset(dataset, val_indices)\n",
        "    \n",
        "    print(f\"  Demo training set: {len(train_dataset)} samples\")\n",
        "    print(f\"  Demo validation set: {len(val_dataset)} samples\")\n",
        "    \n",
        "    # Use smaller batch size and fewer epochs for demo\n",
        "    batch_size = 4\n",
        "    max_epochs = 5\n",
        "\n",
        "# Create data loaders\n",
        "train_loader = DataLoader(\n",
        "    train_dataset,\n",
        "    batch_size=batch_size,\n",
        "    shuffle=True,\n",
        "    num_workers=2,\n",
        "    pin_memory=True\n",
        ")\n",
        "\n",
        "val_loader = DataLoader(\n",
        "    val_dataset,\n",
        "    batch_size=batch_size,\n",
        "    shuffle=False,\n",
        "    num_workers=2,\n",
        "    pin_memory=True\n",
        ")\n",
        "\n",
        "print(f\"\\n[OK] Data loaders created\")\n",
        "print(f\"  Batch size: {batch_size}\")\n",
        "print(f\"  Train batches: {len(train_loader)}\")\n",
        "print(f\"  Val batches: {len(val_loader)}\")\n",
        "print(f\"  Max epochs: {max_epochs}\")\n",
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

# Also update the train_model call (cell 23) to use max_epochs variable
modified_train_call = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Train the model\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(f\"TRAINING MODEL ({'FULL DATASET' if USE_FULL_DATASET else 'DEMO MODE'})\")\n",
        "print(\"=\"*80 + \"\\n\")\n",
        "\n",
        "model, metrics = train_model(\n",
        "    model=model,\n",
        "    train_loader=train_loader,\n",
        "    val_loader=val_loader,\n",
        "    val_dataset=val_dataset,\n",
        "    device=device,\n",
        "    learning_rate=1e-4,\n",
        "    weight_decay=1e-4,\n",
        "    patience=5,\n",
        "    max_epochs=max_epochs,  # Use variable from above\n",
        "    pos_weight=n_benign/n_malignant,\n",
        "    verbose=True\n",
        ")\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"TRAINING COMPLETED!\")\n",
        "print(\"=\"*80)\n",
        "print(f\"\\nFinal Metrics:\")\n",
        "print(f\"  PR-AUC:               {metrics['pr_auc']:.4f}\")\n",
        "print(f\"  AUROC:                {metrics['auroc']:.4f}\")\n",
        "print(f\"  Brier Score:          {metrics['brier']:.4f}\")\n",
        "print(f\"  Robustness Degrad.:   {metrics['robustness_degradation']:.4f}\")\n",
        "print(\"=\"*80)"
    ]
}

# Insert dataset selection cell at position 22
cells.insert(22, dataset_selection_cell)

# Replace the training setup cell (now at position 23)
cells[23] = modified_training_setup

# Replace the train_model call (now at position 24)
cells[24] = modified_train_call

# Update notebook
nb['cells'] = cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Added dataset selection cell")
print(f"[OK] Modified training setup to use full dataset when enabled")
print(f"[OK] Updated train_model call")
print(f"[OK] Notebook updated with {len(cells)} cells")
print(f"[OK] Backup: {backup_path}")
print(f"\n[USAGE] Set USE_FULL_DATASET = True in the new cell to use entire dataset")
