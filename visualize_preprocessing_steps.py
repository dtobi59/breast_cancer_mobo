"""
Visualize each step of the adaptive preprocessing pipeline.

Shows the transformation at each stage:
1. Original DICOM
2. After orientation normalization
3. After breast extraction
4. After inferior fold removal
5. After nipple suppression
6. After resize (grayscale)
7. After entropy adaptation (optional)
8. Final with colormap
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pydicom
from skimage.filters import threshold_otsu

# Add breast_cancer_detection to path
sys.path.insert(0, str(Path(__file__).parent / "breast_cancer_detection"))

from src.domain_adaptation import calculate_shannon_entropy


class VisualizingAdaptivePreprocessor:
    """
    Adaptive preprocessor that captures and displays each step.
    """

    def __init__(self, target_size=(720, 480), aspect_ratio=1.5,
                 entropy_stats=None, apply_adaptation=False):
        self.target_width, self.target_height = target_size
        self.aspect_ratio = aspect_ratio
        self.entropy_stats = entropy_stats
        self.apply_adaptation = apply_adaptation

        self.nipple_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        # Storage for intermediate steps
        self.steps = {}

    def load_dicom(self, path):
        """Step 1: Load DICOM and normalize."""
        print("\n" + "="*70)
        print("STEP 1: LOAD DICOM")
        print("="*70)

        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)

        print(f"Original DICOM:")
        print(f"  Shape: {img.shape}")
        print(f"  Dtype: {ds.pixel_array.dtype}")
        print(f"  Value range: [{img.min():.1f}, {img.max():.1f}]")

        # Apply rescale
        if hasattr(ds, "RescaleSlope"):
            print(f"  Rescale slope: {ds.RescaleSlope}")
            print(f"  Rescale intercept: {ds.RescaleIntercept}")
            img = img * ds.RescaleSlope + ds.RescaleIntercept

        # Handle photometric interpretation
        if ds.PhotometricInterpretation == "MONOCHROME1":
            print(f"  Photometric: {ds.PhotometricInterpretation} (inverting)")
            img = np.max(img) - img
        else:
            print(f"  Photometric: {ds.PhotometricInterpretation}")

        # Normalize to [0, 255]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

        print(f"After normalization:")
        print(f"  Shape: {img.shape}")
        print(f"  Dtype: {img.dtype}")
        print(f"  Value range: [{img.min()}, {img.max()}]")

        self.steps['01_original'] = img.copy()
        return img

    def ensure_left(self, img):
        """Step 2: Ensure left-oriented."""
        print("\n" + "="*70)
        print("STEP 2: ORIENTATION NORMALIZATION (Ensure Left)")
        print("="*70)

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        cx = np.mean(np.where(binary > 0)[1])
        img_center = img.shape[1] / 2

        print(f"Breast centroid X: {cx:.1f}")
        print(f"Image center X: {img_center:.1f}")

        if cx > img_center:
            print(f"  => Breast on RIGHT side, flipping to LEFT")
            img = cv2.flip(img, 1)
        else:
            print(f"  => Breast already on LEFT side, no flip needed")

        self.steps['02_oriented'] = img.copy()
        return img

    def breast_mask(self, img):
        """Generate breast mask."""
        mask = img > threshold_otsu(img)
        return mask

    def extract_single_breast(self, img):
        """Step 3: Extract breast region."""
        print("\n" + "="*70)
        print("STEP 3: BREAST EXTRACTION (Largest Connected Component)")
        print("="*70)

        mask = self.breast_mask(img)

        # Find connected components
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8
        )

        print(f"Found {n-1} connected components (excluding background)")

        # Get largest component
        label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        component = labels == label

        largest_area = stats[label, cv2.CC_STAT_AREA]
        print(f"Largest component area: {largest_area} pixels")

        # Extract bounding box
        ys, xs = np.where(component)
        roi = img[ys.min():ys.max(), xs.min():xs.max()]

        print(f"Bounding box:")
        print(f"  Y: [{ys.min()}, {ys.max()}]")
        print(f"  X: [{xs.min()}, {xs.max()}]")
        print(f"  ROI shape: {roi.shape}")

        self.steps['03_extracted'] = roi.copy()
        return roi

    def remove_inferior_fold_adaptive(self, roi):
        """Step 4: Remove inferior fold."""
        print("\n" + "="*70)
        print("STEP 4: INFERIOR FOLD REMOVAL (Row Density Analysis)")
        print("="*70)

        binary = roi > threshold_otsu(roi)
        row_density = binary.mean(axis=1)

        print(f"Original ROI height: {len(row_density)} rows")

        # Find cutoff point
        cutoff = len(row_density)
        start_row = int(0.4 * len(row_density))
        print(f"Searching from row {start_row} (40% down)")

        for i in range(start_row, len(row_density)):
            if row_density[i] < 0.15:
                cutoff = i
                print(f"Found low density row at {i} (density: {row_density[i]:.3f})")
                break

        if cutoff == len(row_density):
            print(f"No low-density region found, keeping full height")
        else:
            print(f"Cutting at row {cutoff}")
            print(f"  Rows removed: {len(row_density) - cutoff}")
            print(f"  New height: {cutoff} rows")

        roi = roi[:cutoff]
        self.steps['04_fold_removed'] = roi.copy()
        return roi

    def suppress_nipple(self, roi):
        """Step 5: Nipple suppression."""
        print("\n" + "="*70)
        print("STEP 5: NIPPLE SUPPRESSION (Convex Hull Method)")
        print("="*70)

        binary = (roi > threshold_otsu(roi)).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print("No contours found, skipping nipple suppression")
            self.steps['05_nipple_suppressed'] = roi.copy()
            return roi

        print(f"Found {len(contours)} contours")

        # Use largest contour
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)

        contour_area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)

        print(f"Contour area: {contour_area:.0f} pixels")
        print(f"Convex hull area: {hull_area:.0f} pixels")
        print(f"Difference (nipple region): {hull_area - contour_area:.0f} pixels")

        # Create masks
        mask_hull = np.zeros(binary.shape, dtype=np.uint8)
        mask_cnt = np.zeros(binary.shape, dtype=np.uint8)

        cv2.drawContours(mask_hull, [hull], -1, 1, -1)
        cv2.drawContours(mask_cnt, [cnt], -1, 1, -1)

        # Identify nipple region
        nipple = (mask_hull - mask_cnt) > 0
        nipple = cv2.dilate(nipple.astype(np.uint8), self.nipple_kernel)

        nipple_pixels = np.sum(nipple > 0)
        print(f"Nipple mask (after dilation): {nipple_pixels} pixels")

        # Suppress nipple
        roi = roi.copy()
        roi[nipple > 0] = 0

        print(f"Pixels set to zero: {nipple_pixels}")

        self.steps['05_nipple_suppressed'] = roi.copy()
        return roi

    def resize_grayscale(self, img):
        """Step 6: Resize (grayscale only)."""
        print("\n" + "="*70)
        print("STEP 6: RESIZE (Grayscale, Aspect-Ratio Safe)")
        print("="*70)

        h, w = img.shape
        aspect = h / w

        print(f"Current shape: {img.shape}")
        print(f"Current aspect ratio (H/W): {aspect:.3f}")
        print(f"Max allowed aspect ratio: {self.aspect_ratio:.3f}")

        # Aspect ratio check
        if aspect > self.aspect_ratio:
            new_w = int(np.ceil(h / self.aspect_ratio))
            pad = new_w - w

            print(f"Aspect ratio exceeds maximum!")
            print(f"  Required width: {new_w} (current: {w})")
            print(f"  Padding {pad} columns on RIGHT")

            img = np.pad(
                img,
                pad_width=((0, 0), (0, pad)),
                mode="constant",
                constant_values=0
            )

            print(f"  After padding: {img.shape}")
        else:
            print(f"Aspect ratio OK, no padding needed")

        # Resize to target
        print(f"\nResizing to target: ({self.target_width}, {self.target_height})")
        img = cv2.resize(
            img,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR
        )

        print(f"Final shape: {img.shape}")

        self.steps['06_resized'] = img.copy()
        return img

    def apply_entropy_adaptation(self, img):
        """Step 7: Entropy adaptation (optional)."""
        print("\n" + "="*70)
        print("STEP 7: ENTROPY ADAPTATION")
        print("="*70)

        if not self.apply_adaptation or self.entropy_stats is None:
            print("SKIPPED (adaptation disabled)")
            self.steps['07_adapted'] = img.copy()
            return img, None

        from src.domain_adaptation import adaptive_entropy_transform

        initial_entropy = calculate_shannon_entropy(img)
        target_mean = self.entropy_stats.mean
        target_std = self.entropy_stats.std

        print(f"Initial entropy: {initial_entropy:.4f} bits")
        print(f"Target range: [{target_mean - target_std:.4f}, {target_mean + target_std:.4f}] bits")
        print(f"  (mean: {target_mean:.4f}, std: {target_std:.4f})")

        if target_mean - target_std <= initial_entropy <= target_mean + target_std:
            print(f"\n[OK] Already in target range, no adaptation needed")
            self.steps['07_adapted'] = img.copy()
            return img, {'initial_entropy': initial_entropy, 'final_entropy': initial_entropy,
                        'iterations': 0, 'converged': True}

        print(f"\nApplying iterative adaptation...")
        adapted_img, metrics = adaptive_entropy_transform(
            img.copy(),
            target_mean=target_mean,
            target_std=target_std,
            max_iterations=50,
            tolerance=0.1
        )

        print(f"\nResults:")
        print(f"  Final entropy: {metrics['final_entropy']:.4f} bits")
        print(f"  Iterations: {metrics['iterations']}")
        print(f"  Converged: {metrics['converged']}")
        print(f"  Entropy change: {metrics['final_entropy'] - initial_entropy:+.4f} bits")

        self.steps['07_adapted'] = adapted_img.copy()
        return adapted_img, metrics

    def apply_colormap(self, img):
        """Step 8: Apply magma colormap."""
        print("\n" + "="*70)
        print("STEP 8: COLORMAP APPLICATION (Magma => RGB)")
        print("="*70)

        print(f"Input: Grayscale uint8 {img.shape}")

        # Apply magma colormap
        img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
        print(f"After colormap: BGR uint8 {img.shape}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"After BGR=>RGB: RGB uint8 {img.shape}")
        print(f"Final value range: [{img.min()}, {img.max()}]")

        self.steps['08_final'] = img.copy()
        return img

    def __call__(self, dicom_path):
        """Execute full pipeline with step tracking."""
        print("\n" + "="*70)
        print("ADAPTIVE PREPROCESSING PIPELINE")
        print("="*70)
        print(f"Input: {dicom_path}")
        print(f"Target size: {self.target_width}x{self.target_height}")
        print(f"Adaptation: {'ENABLED' if self.apply_adaptation else 'DISABLED'}")
        print("="*70)

        # Execute pipeline
        img = self.load_dicom(dicom_path)
        img = self.ensure_left(img)
        roi = self.extract_single_breast(img)
        roi = self.remove_inferior_fold_adaptive(roi)
        roi = self.suppress_nipple(roi)
        roi = self.resize_grayscale(roi)
        roi, metrics = self.apply_entropy_adaptation(roi)
        roi = self.apply_colormap(roi)

        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)

        return roi, metrics


def visualize_all_steps(steps, entropy_metrics=None):
    """Create comprehensive visualization of all steps."""

    # Determine grid layout
    n_steps = len(steps)
    ncols = 3
    nrows = (n_steps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6*nrows))
    axes = axes.flatten() if n_steps > 1 else [axes]

    # Define step titles
    step_titles = {
        '01_original': 'Step 1: Original DICOM\n(After normalization to [0,255])',
        '02_oriented': 'Step 2: Orientation Normalized\n(Breast positioned left)',
        '03_extracted': 'Step 3: Breast Extracted\n(Largest connected component)',
        '04_fold_removed': 'Step 4: Inferior Fold Removed\n(Row density analysis)',
        '05_nipple_suppressed': 'Step 5: Nipple Suppressed\n(Convex hull method)',
        '06_resized': 'Step 6: Resized\n(720x480, aspect-ratio safe)',
        '07_adapted': 'Step 7: Entropy Adapted\n(Iterative transform)',
        '08_final': 'Step 8: Final Output\n(Magma colormap => RGB)'
    }

    for idx, (step_key, img) in enumerate(sorted(steps.items())):
        ax = axes[idx]

        # Display image
        if len(img.shape) == 3:
            # RGB image
            ax.imshow(img)
        else:
            # Grayscale
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)

        # Title
        title = step_titles.get(step_key, step_key)

        # Add entropy info for certain steps
        if step_key in ['06_resized', '07_adapted']:
            entropy = calculate_shannon_entropy(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
            title += f'\nEntropy: {entropy:.4f} bits'

        # Add shape info
        title += f'\nShape: {img.shape}'

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    # Hide extra subplots
    for idx in range(n_steps, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    return fig


def main():
    """Main visualization script."""

    # Find DICOM file
    dicom_path = "./4a06e41e38450e4c6e743b8e8da1ec60.dicom"

    if not os.path.exists(dicom_path):
        print(f"ERROR: DICOM file not found at {dicom_path}")
        return 1

    print("\n" + "#"*70)
    print("# ADAPTIVE PREPROCESSING VISUALIZATION")
    print("#"*70)
    print(f"# DICOM file: {dicom_path}")
    print("#"*70 + "\n")

    # Check if entropy cache exists
    cache_path = Path("breast_cancer_detection/cache/entropy_stats_vindr_train.json")

    if cache_path.exists():
        print("[OK] Entropy cache found, loading...")
        from src.domain_adaptation import EntropyStatistics
        entropy_stats = EntropyStatistics.load(str(cache_path))
        print(f"  Mean: {entropy_stats.mean:.4f} bits")
        print(f"  Std: {entropy_stats.std:.4f} bits")
        apply_adaptation = True
    else:
        print("[WARNING] Entropy cache not found")
        print(f"  Expected location: {cache_path}")
        print("  Running without adaptation (baseline mode)")
        entropy_stats = None
        apply_adaptation = False

    # Create visualizing preprocessor
    preprocessor = VisualizingAdaptivePreprocessor(
        target_size=(720, 480),
        aspect_ratio=1.5,
        entropy_stats=entropy_stats,
        apply_adaptation=apply_adaptation
    )

    # Run preprocessing
    final_img, metrics = preprocessor(dicom_path)

    # Create visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATION...")
    print("="*70)

    fig = visualize_all_steps(preprocessor.steps, metrics)

    # Save figure
    output_path = "preprocessing_steps_visualization.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] Saved visualization to: {output_path}")

    print("\n" + "#"*70)
    print("# VISUALIZATION COMPLETE!")
    print("#"*70)
    print(f"# View the visualization: {os.path.abspath(output_path)}")
    print("#"*70)

    return 0


if __name__ == "__main__":
    exit(main())
