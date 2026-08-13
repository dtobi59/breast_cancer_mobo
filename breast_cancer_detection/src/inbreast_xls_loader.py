"""
Enhanced INbreast Dataset Loader using .xls file

Uses File Name + Laterality for proper breast-level grouping.
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


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

    # Extract number if string like "4a", "4b", "4c"
    if isinstance(birads, str):
        match = re.search(r"\b([1-6])", birads)
        if not match:
            return None
        birads = int(match.group(1))

    try:
        birads = int(birads)
    except:
        return None

    if birads in benign_list:
        return 0  # benign
    if birads in malignant_list:
        return 1  # malignant

    # BI-RADS 4 or others → exclude
    return None


class INbreastDatasetFromXLS(Dataset):
    """
    INbreast dataset loader using .xls file with proper breast grouping.

    Uses File Name + Laterality to group images into breasts.
    """

    def __init__(
        self,
        dicom_dir,
        xls_file,
        preprocessor,
        benign_birads=[1, 2, 3],
        malignant_birads=[5, 6]
    ):
        """
        Args:
            dicom_dir: Path to AllDICOMs/ directory
            xls_file: Path to INbreast.xls
            preprocessor: MammographyPreprocessor instance
            benign_birads: BI-RADS values for benign class
            malignant_birads: BI-RADS values for malignant class
        """
        self.dicom_dir = Path(dicom_dir)
        self.preprocessor = preprocessor

        # Load .xls file
        df = pd.read_excel(xls_file)

        print(f"[INbreast XLS] Loaded {len(df)} rows from {xls_file}")

        samples = []

        for _, row in df.iterrows():
            # Get file name
            file_id = str(row["File Name"]).strip()
            if file_id in ['nan', '']:
                continue

            # Get BI-RADS
            birads_raw = row["Bi-Rads"]
            label = map_birads_to_binary(birads_raw, benign_birads, malignant_birads)
            if label is None:
                continue

            # Get laterality (L/R)
            laterality = str(row["Laterality"]).strip().upper()
            if laterality not in ['L', 'R']:
                laterality = "UNKNOWN"

            # Get view (CC/MLO/FB)
            view = str(row["View"]).strip().upper()
            if view not in ['CC', 'MLO', 'FB']:
                view = "UNKNOWN"

            # Find DICOM file
            path = self._find_inbreast_dicom(file_id)
            if path is None or not os.path.exists(path):
                continue

            # Extract patient group from file ID
            # File IDs are like: 22678622, 22678646, etc.
            # Group by consecutive numbers
            try:
                file_num = int(float(file_id))
                # Group into patients (every ~4-8 images)
                patient_group = file_num // 100  # Rough grouping
            except:
                patient_group = len(samples) // 4  # Fallback

            samples.append({
                'path': path,
                'label': label,
                'file_id': file_id,
                'laterality': laterality,
                'view': view,
                'patient_group': patient_group
            })

        self.samples = samples
        print(f"[INbreast XLS] Loaded {len(self.samples)} valid image-level samples")

        # Show distribution
        labels = [s['label'] for s in samples]
        print(f"[INbreast XLS] Label distribution: Benign={labels.count(0)}, Malignant={labels.count(1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            img: Tensor of shape (3, H, W) with values in [0, 1]
            label: Tensor of shape () with value 0 or 1
        """
        sample = self.samples[idx]

        # Preprocess
        img = self.preprocessor(sample['path'])
        if img is None:
            raise RuntimeError(f"Failed preprocessing: {sample['path']}")

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(sample['label'], dtype=torch.long)

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
        Group samples by breast using patient_group + laterality.

        Returns:
            List of dicts, each containing:
                - breast_id: (patient_group, laterality)
                - image_indices: List of indices in this dataset
                - label: Breast-level label (majority vote)
        """
        breast_groups = {}

        for idx, sample in enumerate(self.samples):
            # Use patient_group + laterality as breast identifier
            breast_id = (sample['patient_group'], sample['laterality'])

            if breast_id not in breast_groups:
                breast_groups[breast_id] = {
                    "breast_id": breast_id,
                    "image_indices": [],
                    "labels": []
                }

            breast_groups[breast_id]["image_indices"].append(idx)
            breast_groups[breast_id]["labels"].append(sample['label'])

        # Convert to list and assign consensus label
        result = []
        for breast_id, group in breast_groups.items():
            # Use majority vote for breast-level label
            labels = group["labels"]
            breast_label = int(np.round(np.mean(labels)))  # 0 or 1

            result.append({
                "breast_id": breast_id,
                "image_indices": group["image_indices"],
                "label": breast_label
            })

        print(f"[INbreast XLS] Created {len(result)} breast groups from {len(self.samples)} images")
        return result


def load_inbreast_from_xls(dicom_dir, xls_file, preprocessor):
    """
    Convenience function to load INbreast from .xls file.

    Args:
        dicom_dir: Path to AllDICOMs/ directory
        xls_file: Path to INbreast.xls
        preprocessor: MammographyPreprocessor instance

    Returns:
        INbreastDatasetFromXLS instance
    """
    return INbreastDatasetFromXLS(
        dicom_dir=dicom_dir,
        xls_file=xls_file,
        preprocessor=preprocessor
    )
