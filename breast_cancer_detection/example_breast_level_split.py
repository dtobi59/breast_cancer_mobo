"""
Example: Creating breast-level train/val splits

This ensures complete breast groups (CC + MLO views) stay together.
"""

import sys
sys.path.insert(0, 'src')

from datasets import VinDRMammoBinaryDataset, create_breast_level_splits
from preprocessing import MammographyPreprocessor
from torch.utils.data import DataLoader

# Initialize preprocessor
preprocessor = MammographyPreprocessor(target_size=(720, 480))

# Load full dataset
full_dataset = VinDRMammoBinaryDataset(
    images_root='../images',  # Adjust path as needed
    csv_file='../stratified_selection.csv',
    preprocessor=preprocessor
)

# Create breast-level splits (80/20 stratified)
train_subset, val_subset = create_breast_level_splits(
    dataset=full_dataset,
    train_ratio=0.8,
    random_state=42,
    stratify=True  # Stratify by breast-level labels
)

# Create data loaders
train_loader = DataLoader(
    train_subset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_subset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"\nDataLoaders created:")
print(f"  Train loader: {len(train_loader)} batches")
print(f"  Val loader: {len(val_loader)} batches")

# For evaluation with Noisy-OR aggregation, pass val_subset to the trainer
# The aggregate_breast_level_predictions function will handle the Subset properly
print(f"\nFor training, pass val_subset to the trainer:")
print(f"  trainer = Trainer(..., val_dataset=val_subset, ...)")
