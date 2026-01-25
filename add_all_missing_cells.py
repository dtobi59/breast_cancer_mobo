"""
Add all missing code cells to sections 13-16.
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
backup_path = notebook_path.with_suffix('.ipynb.backup3')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Find section indices
sections = {}
for i, cell in enumerate(cells):
    source = cell['source']
    if isinstance(source, list):
        source = ''.join(source)

    if '## 13. Test INbreast' in source:
        sections[13] = i
    elif '## 14. Zero-Shot Evaluation' in source:
        sections[14] = i
    elif '## 15. Entropy-Based' in source:
        sections[15] = i
    elif '## 16. Entropy Statistics' in source:
        sections[16] = i

print(f"[OK] Found sections: {list(sections.keys())}")

# Section 13: INbreast Dataset Loading
section13_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from src.datasets import INbreastDataset\n",
        "\n",
        "# Load INbreast dataset with configured preprocessor\n",
        "print(\"Loading INbreast dataset...\")\n",
        "print(f\"Preprocessor: {'Adaptive' if USE_ADAPTIVE_PREPROCESSING else 'Standard'}\\n\")\n",
        "\n",
        "inbreast_dataset = INbreastDataset(\n",
        "    dicom_dir=INBREAST_DICOM_DIR,\n",
        "    csv_file=INBREAST_CSV,\n",
        "    preprocessor=preprocessor\n",
        ")\n",
        "\n",
        "print(f\"[OK] INbreast dataset loaded: {len(inbreast_dataset)} samples\")\n",
        "\n",
        "# Test sample\n",
        "img, label = inbreast_dataset[0]\n",
        "print(f\"\\nSample shape: {img.shape}\")\n",
        "print(f\"Sample label: {label.item()} ({'Malignant' if label.item() == 1 else 'Benign'})\")\n",
        "\n",
        "# Visualize\n",
        "plt.figure(figsize=(8, 6))\n",
        "plt.imshow(img.permute(1, 2, 0))\n",
        "plt.title(f\"INbreast Sample - {'Malignant' if label.item() == 1 else 'Benign'}\")\n",
        "plt.axis('off')\n",
        "plt.show()"
    ]
}

# Section 14: Zero-Shot Evaluation
section14_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss\n",
        "from src.evaluation import aggregate_breast_level_predictions, compute_metrics\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"ZERO-SHOT EVALUATION ON INBREAST\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "# Ensure model is in eval mode\n",
        "model.eval()\n",
        "\n",
        "# Evaluate on INbreast (zero-shot)\n",
        "print(\"\\n[1/2] Aggregating breast-level predictions...\")\n",
        "y_true_inbreast, y_probs_inbreast = aggregate_breast_level_predictions(\n",
        "    model, inbreast_dataset, device\n",
        ")\n",
        "\n",
        "print(f\"Number of breasts: {len(y_true_inbreast)}\")\n",
        "print(f\"  Benign: {sum(y_true_inbreast == 0)}\")\n",
        "print(f\"  Malignant: {sum(y_true_inbreast == 1)}\")\n",
        "\n",
        "# Compute metrics\n",
        "print(\"\\n[2/2] Computing metrics...\")\n",
        "\n",
        "# Find optimal threshold from VinDr validation set\n",
        "# (Normally you'd compute this from validation predictions)\n",
        "# For demo, using a default\n",
        "optimal_threshold = 0.5\n",
        "\n",
        "inbreast_metrics = compute_metrics(\n",
        "    y_true_inbreast,\n",
        "    y_probs_inbreast,\n",
        "    threshold=optimal_threshold\n",
        ")\n",
        "\n",
        "# Display results\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"ZERO-SHOT RESULTS\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "print(f\"\\nThreshold-Independent Metrics:\")\n",
        "print(f\"  PR-AUC:       {inbreast_metrics['pr_auc']:.4f}\")\n",
        "print(f\"  AUROC:        {inbreast_metrics['auroc']:.4f}\")\n",
        "print(f\"  Brier Score:  {inbreast_metrics['brier']:.4f}\")\n",
        "\n",
        "print(f\"\\nPrediction Distribution:\")\n",
        "print(f\"  Min probability:  {y_probs_inbreast.min():.4f}\")\n",
        "print(f\"  Max probability:  {y_probs_inbreast.max():.4f}\")\n",
        "print(f\"  Spread:           {y_probs_inbreast.max() - y_probs_inbreast.min():.4f} ({(y_probs_inbreast.max() - y_probs_inbreast.min())*100:.1f}% of [0,1] range)\")\n",
        "\n",
        "print(f\"\\nThreshold-Dependent (t={optimal_threshold:.4f}):\")\n",
        "print(f\"  Sensitivity:  {inbreast_metrics['sensitivity']:.4f}\")\n",
        "print(f\"  Specificity:  {inbreast_metrics['specificity']:.4f}\")\n",
        "\n",
        "print(f\"\\nConfusion Matrix:\")\n",
        "print(f\"  TN={inbreast_metrics['tn']:<4} FP={inbreast_metrics['fp']:<4}\")\n",
        "print(f\"  FN={inbreast_metrics['fn']:<4} TP={inbreast_metrics['tp']:<4}\")\n",
        "\n",
        "# Check for domain shift\n",
        "prob_spread = y_probs_inbreast.max() - y_probs_inbreast.min()\n",
        "if prob_spread < 0.20:\n",
        "    print(\"\\n[WARNING] Severe domain shift detected!\")\n",
        "    print(f\"  - Predictions collapsed to narrow range: {prob_spread*100:.1f}% of [0,1]\")\n",
        "    print(f\"  - Model has minimal discriminative power on INbreast\")\n",
        "    print(f\"  - Entropy adaptation may help (see Section 15-16)\")\n",
        "elif inbreast_metrics['auroc'] < 0.60:\n",
        "    print(\"\\n[WARNING] Poor generalization detected!\")\n",
        "    print(f\"  - AUROC barely above random (0.5)\")\n",
        "    print(f\"  - Domain shift likely present\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)"
    ]
}

# Section 15: Add code for entropy-based adaptation explanation
section15_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# This section demonstrates entropy-based domain adaptation\n",
        "# To use adaptation:\n",
        "#   1. Set USE_ADAPTIVE_PREPROCESSING = True in Section 3\n",
        "#   2. Run Section 16 to compute/load entropy statistics\n",
        "#   3. Re-run from Section 4 onwards with adapted preprocessor\n",
        "\n",
        "print(\"Current Configuration:\")\n",
        "print(f\"  Adaptive Preprocessing: {USE_ADAPTIVE_PREPROCESSING}\")\n",
        "print(f\"  Entropy Stats Available: {ENTROPY_STATS_AVAILABLE}\")\n",
        "\n",
        "if USE_ADAPTIVE_PREPROCESSING and ENTROPY_STATS_AVAILABLE:\n",
        "    print(\"\\n[OK] Adaptive preprocessing is ENABLED\")\n",
        "    print(\"     All evaluations above used entropy-adapted images\")\n",
        "else:\n",
        "    print(\"\\n[INFO] Adaptive preprocessing is DISABLED\")\n",
        "    print(\"       Run Section 16 to enable domain adaptation\")"
    ]
}

# Section 16: Entropy Statistics Computation and Comparative Analysis
section16_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from src.domain_adaptation import compute_dataset_entropy_stats, EntropyStatistics\n",
        "from src.cache_manager import EntropyCache\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"ENTROPY STATISTICS COMPUTATION\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "# Check cache\n",
        "cache = EntropyCache(cache_dir=os.path.join(PROJECT_PATH, \"cache\"))\n",
        "\n",
        "if cache.exists(\"vindr_train\"):\n",
        "    print(\"\\n[OK] Loading cached entropy statistics...\")\n",
        "    entropy_stats = cache.load(\"vindr_train\")\n",
        "    \n",
        "    print(f\"\\nEntropy Statistics (from cache):\")\n",
        "    print(f\"  Mean:     {entropy_stats.mean:.4f} bits\")\n",
        "    print(f\"  Std:      {entropy_stats.std:.4f} bits\")\n",
        "    print(f\"  Min:      {entropy_stats.min_entropy:.4f} bits\")\n",
        "    print(f\"  Max:      {entropy_stats.max_entropy:.4f} bits\")\n",
        "    print(f\"  Samples:  {entropy_stats.n_samples}\")\n",
        "    if hasattr(entropy_stats, 'computed_date'):\n",
        "        print(f\"  Computed: {entropy_stats.computed_date}\")\n",
        "    print(f\"\\nTarget range for adaptation: [{entropy_stats.mean - entropy_stats.std:.4f}, {entropy_stats.mean + entropy_stats.std:.4f}] bits\")\n",
        "else:\n",
        "    print(\"\\n[INFO] Cache not found. Computing entropy statistics...\")\n",
        "    print(\"This may take 1-2 hours for full VinDr-Mammo training set.\\n\")\n",
        "    \n",
        "    from src.adaptive_preprocessing import AdaptiveMammographyPreprocessor\n",
        "    \n",
        "    # Create adaptive preprocessor (no adaptation, just for grayscale access)\n",
        "    adaptive_preprocessor = AdaptiveMammographyPreprocessor(\n",
        "        apply_adaptation=False\n",
        "    )\n",
        "    \n",
        "    print(\"Computing entropy statistics on training set...\")\n",
        "    print(\"Progress will be shown below:\\n\")\n",
        "    \n",
        "    # Note: This requires train_dataset to be defined\n",
        "    # For demo purposes, computing on full dataset (should be training only)\n",
        "    entropy_stats = compute_dataset_entropy_stats(\n",
        "        dataset=dataset,\n",
        "        preprocessor=adaptive_preprocessor,\n",
        "        save_path=None,\n",
        "        verbose=True\n",
        "    )\n",
        "    \n",
        "    # Set dataset name\n",
        "    entropy_stats.dataset_name = \"VinDr-Mammo-Train\"\n",
        "    \n",
        "    # Save to cache\n",
        "    print(f\"\\nSaving statistics to cache...\")\n",
        "    cache.save(entropy_stats, \"vindr_train\")\n",
        "    \n",
        "    print(f\"\\n[OK] Computation complete!\")\n",
        "    print(f\"\\nEntropy Statistics:\")\n",
        "    print(f\"  Mean:     {entropy_stats.mean:.4f} bits\")\n",
        "    print(f\"  Std:      {entropy_stats.std:.4f} bits\")\n",
        "    print(f\"  Min:      {entropy_stats.min_entropy:.4f} bits\")\n",
        "    print(f\"  Max:      {entropy_stats.max_entropy:.4f} bits\")\n",
        "    print(f\"  Samples:  {entropy_stats.n_samples}\")\n",
        "    print(f\"\\nTarget range for adaptation: [{entropy_stats.mean - entropy_stats.std:.4f}, {entropy_stats.mean + entropy_stats.std:.4f}] bits\")\n",
        "    print(f\"\\nCache saved to: {cache.get_cache_path('vindr_train')}\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"\\nTo enable adaptive preprocessing:\")\n",
        "print(\"  1. Set USE_ADAPTIVE_PREPROCESSING = True in Section 3\")\n",
        "print(\"  2. Restart notebook runtime (or re-run from Section 3)\")\n",
        "print(\"  3. All subsequent sections will use adapted images\")\n",
        "print(\"\\n\" + \"=\"*80)"
    ]
}

# Insert cells at appropriate positions
# Work backwards to avoid index shifting
insertions = [
    (sections[16] + 1, section16_code),
    (sections[15] + 1, section15_code),
    (sections[14] + 1, section14_code),
    (sections[13] + 1, section13_code),
]

for idx, cell in reversed(insertions):
    cells.insert(idx, cell)
    print(f"[OK] Inserted code cell at index {idx}")

# Update notebook
nb['cells'] = cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Added all missing code cells")
print(f"[OK] Notebook updated with {len(cells)} cells")
print(f"[OK] Backup: {backup_path}")
