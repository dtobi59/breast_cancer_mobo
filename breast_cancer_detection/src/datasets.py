"""
Dataset classes for VinDr-Mammo and INbreast.

Supports:
- Image-level classification (training)
- Breast-level aggregation (evaluation)
"""

import os
import re
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


def map_birads_to_binary(birads, benign_list=[1, 2, 3], malignant_list=[5, 6]):
    """
    Map BI-RADS score to binary label.

    Args:
        birads: BI-RADS score (int, float, or string)
        benign_list: BI-RADS values mapped to 0 (benign)
        malignant_list: BI-RADS values mapped to 1 (malignant)

    Returns:
        0 (benign), 1 (malignant), or None (exclude)
    """
    if pd.isna(birads):
        return None

    # Extract number if string like "BI-RADS 2"
    if isinstance(birads, str):
        match = re.search(r"\d+", birads)
        if not match:
            return None
        birads = int(match.group())

    birads = int(birads)

    if birads in benign_list:
        return 0  # benign
    if birads in malignant_list:
        return 1  # malignant

    # BI-RADS 4 or others → exclude
    return None


class VinDRMammoBinaryDataset(Dataset):
    """
    VinDr-Mammo binary classification dataset.

    Supports:
    - Image-level labels for training
    - Metadata for breast-level aggregation during evaluation
    """

    def __init__(
        self,
        images_root,
        csv_file,
        preprocessor,
        transform=None,
        benign_birads=[1, 2, 3],
        malignant_birads=[5, 6]
    ):
        """
        Args:
            images_root: Path to images directory
            csv_file: Path to metadata CSV
            preprocessor: MammographyPreprocessor instance
            transform: Optional augmentation transform
            benign_birads: BI-RADS values for benign class
            malignant_birads: BI-RADS values for malignant class
        """
        self.images_root = Path(images_root)
        self.preprocessor = preprocessor
        self.transform = transform

        # Load and filter data
        df = pd.read_csv(csv_file)

        # Map BI-RADS to binary labels
        df["label"] = df["breast_birads"].apply(
            lambda x: map_birads_to_binary(x, benign_birads, malignant_birads)
        )
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        # Store samples with metadata for breast-level aggregation
        # Keep: study_id, image_id, laterality, view_position, label
        self.samples = df[
            ["study_id", "image_id", "laterality", "view_position", "label"]
        ].drop_duplicates().values

        print(f"[VinDr-Mammo] Loaded {len(self.samples)} image-level samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            img: Tensor of shape (3, H, W) with values in [0, 1]
            label: Tensor of shape () with value 0 or 1
        """
        study_id, image_id, laterality, view_position, label = self.samples[idx]

        # Construct DICOM path
        dicom_path = self.images_root / str(study_id) / f"{image_id}.dicom"

        # Preprocess
        img = self.preprocessor(str(dicom_path))
        if img is None:
            raise RuntimeError(f"Failed preprocessing: {dicom_path}")

        # Convert to tensor [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(label, dtype=torch.long)

        # Apply augmentation (training only)
        if self.transform:
            img = self.transform(img)

        return img, label

    def get_breast_groups(self):
        """
        Group samples by breast for Noisy-OR aggregation.

        Returns:
            List of dicts, each containing:
                - breast_id: (study_id, laterality)
                - image_indices: List of indices in this dataset
                - label: Breast-level label
        """
        breast_groups = {}

        for idx, (study_id, image_id, laterality, view_pos, label) in enumerate(self.samples):
            breast_id = (study_id, laterality)

            if breast_id not in breast_groups:
                breast_groups[breast_id] = {
                    "breast_id": breast_id,
                    "image_indices": [],
                    "label": label
                }

            breast_groups[breast_id]["image_indices"].append(idx)

        return list(breast_groups.values())


class INbreastDataset(Dataset):
    """
    INbreast dataset for zero-shot evaluation.

    Supports:
    - Image-level predictions
    - Breast-level aggregation
    """

    def __init__(
        self,
        dicom_dir,
        csv_file,
        preprocessor,
        benign_birads=[2, 3],
        malignant_birads=[5, 6]
    ):
        """
        Args:
            dicom_dir: Path to AllDICOMs/ directory
            csv_file: Path to INbreast.csv
            preprocessor: MammographyPreprocessor instance
            benign_birads: BI-RADS values for benign class
            malignant_birads: BI-RADS values for malignant class
        """
        self.dicom_dir = Path(dicom_dir)
        self.preprocessor = preprocessor

        # Load CSV
        df = pd.read_csv(csv_file)

        samples = []

        for _, row in df.iterrows():
            file_id = str(row["File Name"]).strip()
            birads_raw = str(row["Bi-Rads"]).lower().strip()

            # Extract BI-RADS number
            match = re.search(r"\b([1-6])\b", birads_raw)
            if match is None:
                continue

            birads_num = int(match.group(1))

            # Map to binary label
            label = map_birads_to_binary(birads_num, benign_birads, malignant_birads)
            if label is None:
                continue

            # Find DICOM file
            path = self._find_inbreast_dicom(file_id)
            if path is None or not os.path.exists(path):
                continue

            # Extract metadata from filename
            # Format: {file_id}_{hash}_{laterality}_{view}.dcm
            filename = os.path.basename(path)
            parts = filename.split("_")

            laterality = "UNKNOWN"
            view_position = "UNKNOWN"

            if len(parts) >= 4:
                # parts[2] should be laterality (L/R)
                # parts[3] should be view (CC/ML)
                lat_part = parts[2].upper()
                view_part = parts[3].upper()

                if lat_part in ["L", "R", "MG"]:
                    # Handle "MG_L_CC" format
                    if lat_part == "MG" and len(parts) >= 5:
                        laterality = parts[3].upper()
                        view_position = parts[4].split(".")[0].upper()
                    else:
                        laterality = lat_part
                        view_position = view_part.split(".")[0].upper()

            samples.append((path, label, file_id, laterality, view_position))

        self.samples = samples
        print(f"[INbreast] Loaded {len(self.samples)} image-level samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            img: Tensor of shape (3, H, W) with values in [0, 1]
            label: Tensor of shape () with value 0 or 1
        """
        dicom_path, label, file_id, laterality, view_pos = self.samples[idx]

        # Preprocess
        img = self.preprocessor(dicom_path)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def _find_inbreast_dicom(self, file_id):
        """Find DICOM file matching the given file ID."""
        file_id_str = str(file_id).strip()

        try:
            # Convert "22678646.0" to "22678646"
            file_id_int_part = str(int(float(file_id_str)))
        except ValueError:
            return None

        if not self.dicom_dir.exists():
            return None

        for f in self.dicom_dir.iterdir():
            # Check for patterns like "123456_" or just "123456"
            if f.name.startswith(f"{file_id_int_part}_") or f.stem == file_id_int_part:
                return str(f)

        return None

    def get_breast_groups(self):
        """
        Group samples by breast for Noisy-OR aggregation.

        Note: INbreast file IDs don't directly encode patient/breast ID,
        so we use heuristics based on consecutive file IDs and laterality.

        Returns:
            List of dicts, each containing:
                - breast_id: (patient_group, laterality)
                - image_indices: List of indices in this dataset
                - label: Breast-level label
        """
        breast_groups = {}

        for idx, (path, label, file_id, laterality, view_pos) in enumerate(self.samples):
            # Extract base file ID
            try:
                file_num = int(float(file_id))
            except:
                file_num = idx  # Fallback

            # Group by approximate patient (every 4-8 images)
            patient_group = file_num // 10

            breast_id = (patient_group, laterality)

            if breast_id not in breast_groups:
                breast_groups[breast_id] = {
                    "breast_id": breast_id,
                    "image_indices": [],
                    "label": label
                }

            breast_groups[breast_id]["image_indices"].append(idx)

        return list(breast_groups.values())
