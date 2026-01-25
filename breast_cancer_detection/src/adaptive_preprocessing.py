"""
Adaptive preprocessing wrapper with entropy-based domain adaptation.

This module wraps MammographyPreprocessor to enable entropy adaptation
at the grayscale stage (before colormap application).

Code duplication is intentional to:
1. Access grayscale intermediate before colormap
2. Respect "DO NOT MODIFY" constraint on preprocessing.py
3. Ensure identical preprocessing behavior when adaptation disabled
"""

import cv2
import pydicom
import numpy as np
from typing import Optional
from skimage.filters import threshold_otsu

from .domain_adaptation import EntropyStatistics, adaptive_entropy_transform


class AdaptiveMammographyPreprocessor:
    """
    Preprocessing pipeline with optional entropy adaptation.

    Replicates MammographyPreprocessor but enables entropy adaptation
    on grayscale image before colormap application.

    Pipeline:
    1. load_dicom()                # DICOM → grayscale uint8 [0,255]
    2. ensure_left()               # Orientation normalization
    3. extract_single_breast()     # Breast region extraction
    4. remove_inferior_fold_adaptive()  # Remove inferior fold
    5. suppress_nipple()           # Nipple suppression
    6. resize_grayscale()          # Resize (grayscale only)
    7. ENTROPY ADAPTATION          # ← NEW STEP (if enabled)
    8. apply_colormap()            # Magma colormap → RGB

    Usage:
        # Without adaptation (identical to MammographyPreprocessor)
        >>> preprocessor = AdaptiveMammographyPreprocessor(
        ...     apply_adaptation=False
        ... )
        >>> img = preprocessor(dicom_path)

        # With adaptation
        >>> from domain_adaptation import EntropyStatistics
        >>> stats = EntropyStatistics.load('cache/entropy_stats.json')
        >>> preprocessor = AdaptiveMammographyPreprocessor(
        ...     entropy_stats=stats,
        ...     apply_adaptation=True
        ... )
        >>> img = preprocessor(dicom_path)
        >>> # Access adaptation metrics
        >>> summary = preprocessor.get_adaptation_summary()
    """

    def __init__(
        self,
        target_size=(720, 480),
        aspect_ratio=1.5,
        entropy_stats: Optional[EntropyStatistics] = None,
        apply_adaptation: bool = False,
        max_iterations: int = 50,
        tolerance: float = 0.1
    ):
        """
        Initialize adaptive preprocessor.

        Args:
            target_size: (width, height) for final output
            aspect_ratio: Maximum allowed height/width ratio
            entropy_stats: EntropyStatistics from source domain
            apply_adaptation: Whether to apply entropy adaptation
            max_iterations: Max iterations for entropy convergence
            tolerance: Convergence tolerance in bits
        """
        # Preprocessing parameters (same as MammographyPreprocessor)
        self.target_width, self.target_height = target_size
        self.aspect_ratio = aspect_ratio

        # Morphological kernel for nipple suppression
        self.nipple_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (15, 15)
        )

        # Adaptation parameters
        self.entropy_stats = entropy_stats
        self.apply_adaptation = apply_adaptation
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Metrics tracking
        self.adaptation_metrics = []

    # ========================================================================
    # REPLICATED METHODS FROM preprocessing.py
    # (Lines 40-144 from MammographyPreprocessor)
    # ========================================================================

    def load_dicom(self, path):
        """
        Load DICOM and apply rescale transformations.

        Replicated from preprocessing.py lines 40-57
        """
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)

        # Apply rescale slope/intercept if present
        if hasattr(ds, "RescaleSlope"):
            img = img * ds.RescaleSlope + ds.RescaleIntercept

        # Handle photometric interpretation
        if ds.PhotometricInterpretation == "MONOCHROME1":
            img = np.max(img) - img

        # Normalize to [0, 255]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

        return img

    def ensure_left(self, img):
        """
        Flip image if breast is on right side.

        Replicated from preprocessing.py lines 59-67
        """
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        cx = np.mean(np.where(binary > 0)[1])

        if cx > img.shape[1] / 2:
            img = cv2.flip(img, 1)

        return img

    def breast_mask(self, img):
        """
        Generate binary mask of breast tissue.

        Replicated from preprocessing.py lines 69-72
        """
        mask = img > threshold_otsu(img)
        return mask

    def extract_single_breast(self, img):
        """
        Extract largest connected component (breast region).

        Replicated from preprocessing.py lines 74-91
        """
        mask = self.breast_mask(img)

        # Find connected components
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8
        )

        # Get largest component (excluding background)
        label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        component = labels == label

        # Extract bounding box
        ys, xs = np.where(component)
        roi = img[ys.min():ys.max(), xs.min():xs.max()]

        return roi

    def remove_inferior_fold_adaptive(self, roi):
        """
        Remove inferior fold based on row density analysis.

        Replicated from preprocessing.py lines 93-106
        """
        binary = roi > threshold_otsu(roi)
        row_density = binary.mean(axis=1)

        # Find cutoff point (starting from 40% down)
        cutoff = len(row_density)
        for i in range(int(0.4 * len(row_density)), len(row_density)):
            if row_density[i] < 0.15:
                cutoff = i
                break

        roi = roi[:cutoff]
        return roi

    def suppress_nipple(self, roi):
        """
        Suppress nipple artifact using convex hull method.

        Replicated from preprocessing.py lines 108-144
        """
        binary = (roi > threshold_otsu(roi)).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return roi

        # Use largest contour
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)

        # Create masks
        mask_hull = np.zeros(binary.shape, dtype=np.uint8)
        mask_cnt = np.zeros(binary.shape, dtype=np.uint8)

        cv2.drawContours(mask_hull, [hull], -1, 1, -1)
        cv2.drawContours(mask_cnt, [cnt], -1, 1, -1)

        # Identify nipple region
        nipple = (mask_hull - mask_cnt) > 0
        nipple = cv2.dilate(
            nipple.astype(np.uint8),
            self.nipple_kernel
        )

        # Suppress nipple
        roi = roi.copy()
        roi[nipple > 0] = 0

        return roi

    # ========================================================================
    # NEW METHODS FOR ENTROPY ADAPTATION
    # ========================================================================

    def resize_grayscale(self, img):
        """
        Resize grayscale image WITHOUT applying colormap.

        This is the first half of resize_and_color() from preprocessing.py.
        Replicates lines 157-178 but STOPS before colormap.

        Args:
            img: Grayscale image, uint8 [0, 255]

        Returns:
            Resized grayscale image, uint8 [0, 255], shape (target_height, target_width)
        """
        h, w = img.shape

        # Aspect ratio check (same logic as original)
        if h / w > self.aspect_ratio:
            # Compute required width to satisfy h/w = aspect_ratio
            new_w = int(np.ceil(h / self.aspect_ratio))
            pad = new_w - w

            # Pad zero columns on the RIGHT
            img = np.pad(
                img,
                pad_width=((0, 0), (0, pad)),
                mode="constant",
                constant_values=0
            )

        # Resize to target size
        img = cv2.resize(
            img,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR
        )

        return img

    def apply_colormap(self, img):
        """
        Apply magma colormap and convert to RGB.

        This is the second half of resize_and_color() from preprocessing.py.
        Replicates lines 181-184.

        Args:
            img: Grayscale image, uint8 [0, 255]

        Returns:
            RGB image, uint8 [0, 255], shape (H, W, 3)
        """
        # Apply magma colormap
        img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __call__(self, dicom_path):
        """
        Execute preprocessing pipeline with optional entropy adaptation.

        This replicates MammographyPreprocessor.__call__ but adds
        entropy adaptation step between resize and colormap.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            RGB image, numpy array of shape (H, W, 3) with values in [0, 255]
        """
        # Steps 1-5: Same as original (lines 198-202)
        img = self.load_dicom(dicom_path)
        img = self.ensure_left(img)
        roi = self.extract_single_breast(img)
        roi = self.remove_inferior_fold_adaptive(roi)
        roi = self.suppress_nipple(roi)

        # Step 6: Resize (grayscale only) - MODIFIED
        roi = self.resize_grayscale(roi)

        # Step 7: ENTROPY ADAPTATION - NEW STEP
        if self.apply_adaptation and self.entropy_stats is not None:
            roi, metrics = adaptive_entropy_transform(
                image=roi,
                target_mean=self.entropy_stats.mean,
                target_std=self.entropy_stats.std,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance
            )
            self.adaptation_metrics.append(metrics)

        # Step 8: Apply colormap - MODIFIED (separated from resize)
        roi = self.apply_colormap(roi)

        return roi

    def get_adaptation_summary(self):
        """
        Get summary of adaptation metrics across all processed images.

        Returns:
            Dictionary with aggregated statistics or None if no adaptation applied:
            - mean_initial_entropy: Average entropy before adaptation
            - mean_final_entropy: Average entropy after adaptation
            - mean_iterations: Average iterations needed
            - convergence_rate: Fraction of images that converged
            - n_samples: Number of images processed
        """
        if not self.adaptation_metrics:
            return None

        return {
            'mean_initial_entropy': np.mean([m['initial_entropy'] for m in self.adaptation_metrics]),
            'mean_final_entropy': np.mean([m['final_entropy'] for m in self.adaptation_metrics]),
            'mean_iterations': np.mean([m['iterations'] for m in self.adaptation_metrics]),
            'convergence_rate': np.mean([m['converged'] for m in self.adaptation_metrics]),
            'n_samples': len(self.adaptation_metrics)
        }

    def reset_metrics(self):
        """Reset adaptation metrics tracking."""
        self.adaptation_metrics = []
