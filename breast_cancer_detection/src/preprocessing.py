"""
Mammography preprocessing pipeline
Extracted from baseline notebook - DO NOT MODIFY
"""

import cv2
import pydicom
import numpy as np
from skimage.filters import threshold_otsu


class MammographyPreprocessor:
    """
    Fixed preprocessing pipeline for mammography DICOM images.

    Pipeline steps:
    1. Load DICOM with rescale slope/intercept
    2. Orientation normalization (left-oriented)
    3. Breast region extraction (connected components)
    4. Inferior fold removal (row density analysis)
    5. Nipple suppression (convex hull + morphology)
    6. Aspect-ratio-safe resize to 720×480
    7. Magma colormap → RGB
    """

    def __init__(self, target_size=(720, 480), aspect_ratio=1.5):
        """
        Args:
            target_size: (width, height) for final output
            aspect_ratio: maximum allowed height/width ratio
        """
        self.target_width, self.target_height = target_size
        self.aspect_ratio = aspect_ratio

        # Morphological kernel for nipple suppression
        self.nipple_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (15, 15)
        )

    def load_dicom(self, path):
        """Load DICOM and apply rescale transformations."""
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
        """Flip image if breast is on right side."""
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        cx = np.mean(np.where(binary > 0)[1])

        if cx > img.shape[1] / 2:
            img = cv2.flip(img, 1)

        return img

    def breast_mask(self, img):
        """Generate binary mask of breast tissue."""
        mask = img > threshold_otsu(img)
        return mask

    def extract_single_breast(self, img):
        """Extract largest connected component (breast region)."""
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
        """Remove inferior fold based on row density analysis."""
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
        """Suppress nipple artifact using convex hull method."""
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

    def resize_and_color(self, img):
        """
        Aspect-ratio-safe resizing and magma color mapping.

        Steps:
        1. Check aspect ratio (h/w > 1.5)
        2. Pad with zeros if needed to satisfy ratio
        3. Resize to target size
        4. Apply magma colormap
        5. Convert BGR to RGB
        """
        h, w = img.shape

        # Aspect ratio check
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

        # Apply magma colormap
        img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __call__(self, dicom_path):
        """
        Execute full preprocessing pipeline.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            numpy array of shape (H, W, 3) with values in [0, 255]
        """
        img = self.load_dicom(dicom_path)
        img = self.ensure_left(img)
        roi = self.extract_single_breast(img)
        roi = self.remove_inferior_fold_adaptive(roi)
        roi = self.suppress_nipple(roi)
        roi = self.resize_and_color(roi)

        return roi
