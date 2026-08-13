"""
Dataset classes for VinDr-Mammo and INbreast.

Supports:
- Image-level classification (training)
- Breast-level aggregation (evaluation)
"""

import os
import re
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from sklearn.model_selection import train_test_split


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


def create_breast_level_splits(dataset, train_ratio=0.8, random_state=42, stratify=True):
    """
    Create train/val splits at the breast level (preserves complete breast groups).

    This ensures that both CC and MLO views from the same breast stay together
    in either train or validation set, which is crucial for proper Noisy-OR evaluation.

    Args:
        dataset: VinDRMammoBinaryDataset instance
        train_ratio: Fraction of data for training (default 0.8)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by breast-level labels

    Returns:
        train_subset: Subset for training
        val_subset: Subset for validation
    """
    # Get all breast groups
    breast_groups = dataset.get_breast_groups()

    # Collect breast-level info
    breast_indices_list = []  # List of image indices for each breast
    breast_labels = []  # Label for each breast

    for group in breast_groups:
        breast_indices_list.append(group["image_indices"])
        breast_labels.append(group["label"])

    breast_labels = np.array(breast_labels)

    # Split at breast level
    n_breasts = len(breast_groups)
    breast_idx = np.arange(n_breasts)

    if stratify:
        train_breast_idx, val_breast_idx = train_test_split(
            breast_idx,
            train_size=train_ratio,
            random_state=random_state,
            stratify=breast_labels
        )
    else:
        train_breast_idx, val_breast_idx = train_test_split(
            breast_idx,
            train_size=train_ratio,
            random_state=random_state
        )

    # Collect all image indices for train and val
    train_image_indices = []
    for idx in train_breast_idx:
        train_image_indices.extend(breast_indices_list[idx])

    val_image_indices = []
    for idx in val_breast_idx:
        val_image_indices.extend(breast_indices_list[idx])

    # Create subsets
    train_subset = Subset(dataset, train_image_indices)
    val_subset = Subset(dataset, val_image_indices)

    # Print statistics
    print(f"\n[Breast-level split]")
    print(f"  Total breasts: {n_breasts}")
    print(f"  Train breasts: {len(train_breast_idx)} ({len(train_image_indices)} images)")
    print(f"  Val breasts: {len(val_breast_idx)} ({len(val_image_indices)} images)")

    if stratify:
        train_labels = breast_labels[train_breast_idx]
        val_labels = breast_labels[val_breast_idx]
        print(f"  Train label distribution: {np.bincount(train_labels)}")
        print(f"  Val label distribution: {np.bincount(val_labels)}")

    return train_subset, val_subset

def create_inbreast_calibration_test_splits(dataset,calibration_ratio=0.2,random_state=42,stratify=True):
    """
    Create IMAGE-LEVEL calibration/test splits for INbreast.

    Patient/breast identifiers cannot be reliably recovered from the
    available INbreast metadata, so no heuristic breast grouping is used.

    Calibration:
        Used only for Objective 4 during HPO.

    Test:
        Held out until final cross-dataset evaluation.

    IMPORTANT:
        Because verified patient identifiers are unavailable, patient-level
        independence between calibration and test cannot be guaranteed.
        This must be reported as a study limitation.
    """

    # ------------------------------------------------------
    # 1. Image indices
    # ------------------------------------------------------
    n_images = len(dataset)
    image_indices = np.arange(n_images)

    # INbreast sample format:
    # (path, label, file_id, laterality, view_position)
    labels = np.array([
        int(sample[1])
        for sample in dataset.samples
    ])

    # ------------------------------------------------------
    # 2. Stratified image-level split
    # ------------------------------------------------------
    if stratify:
        calibration_indices, test_indices = train_test_split(
            image_indices,
            train_size=calibration_ratio,
            random_state=random_state,
            stratify=labels
        )
    else:
        calibration_indices, test_indices = train_test_split(
            image_indices,
            train_size=calibration_ratio,
            random_state=random_state
        )

    # ------------------------------------------------------
    # 3. Create PyTorch subsets
    # ------------------------------------------------------
    calibration_subset = Subset(
        dataset,
        calibration_indices.tolist()
    )

    test_subset = Subset(
        dataset,
        test_indices.tolist()
    )

    # ------------------------------------------------------
    # 4. Statistics
    # ------------------------------------------------------
    calibration_labels = labels[calibration_indices]
    test_labels = labels[test_indices]

    print("\n[INbreast IMAGE-LEVEL Calibration/Test split]")
    print(f"  Total images: {n_images}")
    print(
        f"  Calibration images: {len(calibration_indices)} "
        f"({len(calibration_indices) / n_images:.1%})"
    )
    print(
        f"  Test images: {len(test_indices)} "
        f"({len(test_indices) / n_images:.1%})"
    )

    print(
        f"  Calibration label distribution: "
        f"{np.bincount(calibration_labels)}"
    )

    print(
        f"  Test label distribution: "
        f"{np.bincount(test_labels)}"
    )

    # ------------------------------------------------------
    # 5. Reproducibility metadata
    # ------------------------------------------------------
    split_info = {
        "split_level": "image",
        "random_state": random_state,
        "calibration_ratio": calibration_ratio,
        "stratify": stratify,

        "n_images_total": n_images,
        "n_images_calibration": len(calibration_indices),
        "n_images_test": len(test_indices),

        "calibration_image_indices":
            calibration_indices.tolist(),

        "test_image_indices":
            test_indices.tolist(),

        "patient_independence_guaranteed": False
    }

    return (
        calibration_subset,
        test_subset,
        split_info
    )


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
        csv_file=None,
        preprocessor=None,
        transform=None,
        benign_birads=[1, 2, 3],
        malignant_birads=[5, 6],
        samples=None
    ):
        """
        Args:
            images_root: Path to images directory
            csv_file: Path to metadata CSV (optional if samples provided)
            preprocessor: MammographyPreprocessor instance
            transform: Optional augmentation transform
            benign_birads: BI-RADS values for benign class
            malignant_birads: BI-RADS values for malignant class
            samples: Pre-loaded samples array (optional, avoids CSV re-reading)
        """
        self.images_root = Path(images_root)
        self.preprocessor = preprocessor
        self.transform = transform

        # Load from CSV or use provided samples
        if samples is not None:
            # Use pre-loaded samples (avoids CSV re-reading during optimization)
            self.samples = samples
            print(f"[VinDr-Mammo] Using {len(self.samples)} provided samples")
        elif csv_file is not None:
            # Load and filter data from CSV
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
        else:
            raise ValueError("Either csv_file or samples must be provided")

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

            case_id, laterality, view_position = (
                extract_inbreast_dicom_metadata(path)
            )

            # laterality = "UNKNOWN"
            # view_position = "UNKNOWN"

            # if len(parts) >= 4:
            #     # parts[2] should be laterality (L/R)
            #     # parts[3] should be view (CC/ML)
            #     lat_part = parts[2].upper()
            #     view_part = parts[3].upper()

            #     if lat_part in ["L", "R", "MG"]:
            #         # Handle "MG_L_CC" format
            #         if lat_part == "MG" and len(parts) >= 5:
            #             laterality = parts[3].upper()
            #             view_position = parts[4].split(".")[0].upper()
            #         else:
            #             laterality = lat_part
            #             view_position = view_part.split(".")[0].upper()

            # samples.append((path, label, file_id, laterality, view_position))

            samples.append((
                path,
                label,
                file_id,
                case_id,
                laterality,
                view_position
            ))

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
        Group CC/MLO images belonging to the same breast.

        Breast identity is defined as:
            (case_id, laterality)
        """

        breast_groups = {}

        for idx, sample in enumerate(self.samples):

            (
                path,
                label,
                file_id,
                case_id,
                laterality,
                view_position
            ) = sample

            breast_id = (case_id, laterality)

            if breast_id not in breast_groups:
                breast_groups[breast_id] = {
                    "breast_id": breast_id,
                    "patient_id": case_id,
                    "laterality": laterality,
                    "image_indices": [],
                    "views": [],
                    "label": label,
                }

            breast_groups[breast_id]["image_indices"].append(idx)
            breast_groups[breast_id]["views"].append(view_position)

            # Safety check: images grouped into one breast
            # should not have conflicting labels.
            if breast_groups[breast_id]["label"] != label:
                raise ValueError(
                    f"Conflicting labels for breast {breast_id}"
                )

        return list(breast_groups.values())


# ============================================================================
# FACTORY FUNCTIONS FOR ENTROPY ADAPTATION
# ============================================================================

def create_vindr_dataset_with_adaptation(
    images_root: str,
    csv_file: str,
    split: str,  # 'train', 'val', 'test'
    entropy_stats_path: Optional[str] = None,
    apply_adaptation: bool = False,
    transform=None,
    benign_birads=[1, 2, 3],
    malignant_birads=[5, 6]
) -> VinDRMammoBinaryDataset:
    """
    Create VinDr dataset with optional entropy adaptation.

    This factory function enables easy creation of datasets with or without
    entropy adaptation for domain shift mitigation.

    Args:
        images_root: Path to images directory
        csv_file: Path to metadata CSV
        split: Dataset split identifier ('train', 'val', 'test')
        entropy_stats_path: Path to cached entropy statistics JSON
        apply_adaptation: Whether to apply entropy adaptation
        transform: Optional augmentation transform (for training)
        benign_birads: BI-RADS values for benign class
        malignant_birads: BI-RADS values for malignant class

    Returns:
        VinDRMammoBinaryDataset instance

    Usage:
        # Training: NO adaptation (preserve clean training data)
        >>> train_ds = create_vindr_dataset_with_adaptation(
        ...     images_root=VINDR_IMAGES_ROOT,
        ...     csv_file=VINDR_CSV,
        ...     split='train',
        ...     apply_adaptation=False
        ... )

        # Validation: WITH adaptation (experimental per user requirement)
        >>> val_ds = create_vindr_dataset_with_adaptation(
        ...     images_root=VINDR_IMAGES_ROOT,
        ...     csv_file=VINDR_CSV,
        ...     split='val',
        ...     entropy_stats_path='cache/entropy_stats_vindr_train.json',
        ...     apply_adaptation=True
        ... )

    Note:
        - Training data should NEVER have adaptation applied
        - Validation/test data can optionally have adaptation
        - Adaptation requires entropy_stats_path to be provided
    """
    if apply_adaptation and entropy_stats_path:
        from .domain_adaptation import EntropyStatistics
        from .adaptive_preprocessing import AdaptiveMammographyPreprocessor

        # Load entropy statistics from cache
        entropy_stats = EntropyStatistics.load(entropy_stats_path)

        # Create adaptive preprocessor
        preprocessor = AdaptiveMammographyPreprocessor(
            target_size=(720, 480),
            aspect_ratio=1.5,
            entropy_stats=entropy_stats,
            apply_adaptation=True
        )
    else:
        # Use regular preprocessor (no adaptation)
        from .preprocessing import MammographyPreprocessor
        preprocessor = MammographyPreprocessor(
            target_size=(720, 480),
            aspect_ratio=1.5
        )

    # Create dataset with appropriate preprocessor
    dataset = VinDRMammoBinaryDataset(
        images_root=images_root,
        csv_file=csv_file,
        preprocessor=preprocessor,
        transform=transform,
        benign_birads=benign_birads,
        malignant_birads=malignant_birads
    )

    return dataset


def create_inbreast_dataset_with_adaptation(
    dicom_dir: str,
    csv_file: str,
    entropy_stats_path: str,
    benign_birads=[2, 3],
    malignant_birads=[5, 6]
) -> INbreastDataset:
    """
    Create INbreast dataset with mandatory entropy adaptation.

    For zero-shot evaluation on INbreast (target domain), entropy adaptation
    is ALWAYS applied to mitigate domain shift.

    Args:
        dicom_dir: Path to AllDICOMs directory
        csv_file: Path to INbreast.csv
        entropy_stats_path: Path to cached entropy statistics (REQUIRED)
        benign_birads: BI-RADS values for benign class
        malignant_birads: BI-RADS values for malignant class

    Returns:
        INbreastDataset instance with adaptation enabled

    Usage:
        >>> inbreast_ds = create_inbreast_dataset_with_adaptation(
        ...     dicom_dir=INBREAST_DICOM_DIR,
        ...     csv_file=INBREAST_CSV,
        ...     entropy_stats_path='cache/entropy_stats_vindr_train.json'
        ... )

    Note:
        Adaptation is always applied for INbreast to align with VinDr-Mammo
        intensity distribution.
    """
    from .domain_adaptation import EntropyStatistics
    from .adaptive_preprocessing import AdaptiveMammographyPreprocessor

    # Load entropy statistics
    entropy_stats = EntropyStatistics.load(entropy_stats_path)

    # Create adaptive preprocessor (always enabled for INbreast)
    preprocessor = AdaptiveMammographyPreprocessor(
        target_size=(720, 480),
        aspect_ratio=1.5,
        entropy_stats=entropy_stats,
        apply_adaptation=True
    )

    # Create dataset
    dataset = INbreastDataset(
        dicom_dir=dicom_dir,
        csv_file=csv_file,
        preprocessor=preprocessor,
        benign_birads=benign_birads,
        malignant_birads=malignant_birads
    )

    return dataset

def extract_inbreast_dicom_metadata(path):
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=True)

    patient_id = str(getattr(ds, "PatientID", "")).strip()
    study_uid = str(getattr(ds, "StudyInstanceUID", "")).strip()

    laterality = str(
        getattr(
            ds,
            "ImageLaterality",
            getattr(ds, "Laterality", "")
        )
    ).strip().upper()

    view_position = str(
        getattr(ds, "ViewPosition", "")
    ).strip().upper()

    # Prefer PatientID.
    # If unavailable, StudyInstanceUID can serve as study/case identity.
    if patient_id:
        case_id = patient_id
    elif study_uid:
        case_id = study_uid
    else:
        raise ValueError(
            f"No PatientID or StudyInstanceUID found in {path}"
        )

    if laterality not in {"L", "R"}:
        raise ValueError(
            f"Invalid/missing laterality for {path}: {laterality}"
        )

    return case_id, laterality, view_position