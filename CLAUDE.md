# Project: Breast Cancer Detection under Dataset Shift
**Domain:** Medical Imaging · Deep Learning · Multi-Objective Optimization

---

## Role & Operating Rules (MANDATORY)

You are a **stateful research collaborator** assisting with a **research-grade medical imaging study**.

- Treat this file as **authoritative context**
- Maintain **full state across turns**
- Do **not simplify, replace, or invent methods**
- Ask **clarifying questions before implementing anything**
- Always align new code with the **existing working notebook**

If any requirement is unclear, **pause and ask**.

---

## Existing Baseline (Ground Truth)

A **working Jupyter notebook** already exists and is the **single source of truth**:

**Notebook:** `Based project.ipynb`

This notebook:
- Implements a complete preprocessing pipeline for mammography DICOM images
- Trains a ResNet152 binary classifier on VinDr-Mammo
- Achieves ~0.95 ROC-AUC on validation
- Demonstrates zero-shot generalization to INbreast

Your task is to **extend and formalize this notebook**, not rewrite it.

---

## Current project direcory
the breast_cancer_detection directory contain the current implement. All bugs fixes will be apply inside the breast_cancer_detection directory

## Datasets

### Source Dataset
- **VinDr-Mammo**
- Patient-wise split: **80% train / 20% validation**
- **Single fixed split**
- No cross-validation
- Same split reused for all experiments

**Label Mapping (fixed):**
- Benign: BI-RADS 1–3
- Malignant: BI-RADS 5–6
- Exclude BI-RADS 4

### Target Dataset
- **INbreast**
- Used **only for zero-shot evaluation**
- No fine-tuning
- No threshold tuning

---

## Preprocessing (DO NOT MODIFY)

Use the existing pipeline implemented in the notebook:

**MammographyPreprocessor**
- DICOM loading with rescale slope/intercept
- Orientation normalization (left-oriented)
- Breast region extraction (connected components)
- Inferior fold removal (row density analysis)
- Nipple suppression (convex hull + morphology)
- Aspect-ratio-safe resize to **720×480**
- Magma colormap → RGB

No alternative preprocessing is allowed.

---

## Prediction Units

### Training
- Image-level classification

### Evaluation
- Breast-level aggregation
- Each breast has **CC + MLO views**

**Mandatory Noisy-OR aggregation:**

\[
p_{breast} = 1 - (1 - p_{CC})(1 - p_{MLO})
\]

Used for:
- Validation evaluation
- Zero-shot INbreast evaluation

---

## Model

- Architecture: **ResNet152**
- Initialization: ImageNet pretrained
- Binary head: single logit + sigmoid
- Partial fine-tuning controlled by a **continuous hyperparameter ∈ [0, 1]**
  - Defines fraction of unfrozen backbone layers

---

## Training Protocol (FIXED)

- Optimizer: **AdamW** (not optimized)
- Loss: Binary Cross-Entropy
- Batch size: Fixed
- Class imbalance handling allowed (as in notebook)
- Early stopping:
  - Monitor **validation PR-AUC**
  - Restore best checkpoint
- Fix all random seeds

---

## Hyperparameters to Optimize (Continuous)

1. Learning rate (log-scale)
2. Weight decay (log-scale)
3. Dropout rate ∈ [0, 0.5]
4. Augmentation strength ∈ [0, 1]
5. Fraction of unfrozen backbone layers ∈ [0, 1]

---

## Augmentation Strength Definition

Augmentation strength is a **scalar controller** for **intensity-only augmentations**:

- Brightness adjustment
- Contrast scaling
- Additive Gaussian noise

Rules:
- No geometric transforms
- Applied **only during training**
- Same base augmentations as baseline notebook

---

## Objectives (Validation Set Only)

Four objectives:

1. Maximize PR-AUC
2. Maximize AUROC
3. Minimize Brier score
4. Minimize robustness degradation

### Robustness Degradation

- Apply **mild intensity perturbations at inference**
- Compute:
  - PR-AUC (standard inference)
  - PR-AUC (perturbed inference)

\[
R = PR\text{-}AUC_{standard} - PR\text{-}AUC_{perturbed}
\]

---

## Optimization Formulation

Convert maximization to minimization:

\[
F(h) = [-PR\text{-}AUC,\ -AUROC,\ Brier,\ R]
\]

---

## Optimization Algorithm

- **NSGA-III** via `pymoo`
- 4-objective many-objective optimization
- Use reference directions
- Fixed population size and generations
- **Each evaluation trains a full CNN**

---

## Zero-Shot Evaluation (INbreast)

Performed only for **non-dominated Pareto solutions**:

- No retraining
- No threshold tuning
- Same preprocessing
- Same Noisy-OR aggregation
- Decision thresholds:
  - Selected on VinDr-Mammo validation
  - Transferred unchanged

---

## Metrics to Report

For both validation and INbreast:

- PR-AUC
- AUROC
- Brier score
- Sensitivity & specificity at transferred thresholds

---

## Required Deliverables

- `pymoo` problem definition class
- PyTorch / Lightning training & validation pipeline
- Utilities:
  - Data loading
  - Augmentation
  - Noisy-OR aggregation
  - Robustness evaluation
- NSGA-III optimization script with logging
- Evaluation scripts:
  - Source validation
  - Zero-shot INbreast transfer

---

## Hard Constraints

- Code must remain **simple and readable**
- No invented methods
- No relaxed objectives
- Always reuse existing notebook logic
- Ask questions **before implementation**

