"""
NSGA-III optimization script for breast cancer detection.

Optimizes 5 continuous hyperparameters across 4 objectives:
- Hyperparameters: learning_rate, weight_decay, dropout, augmentation_strength, unfreeze_fraction
- Objectives: PR-AUC, AUROC, Brier score, Robustness degradation
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import MammographyPreprocessor
from src.datasets import VinDRMammoBinaryDataset
from src.optimization import BreastCancerOptimizationProblem, HyperparameterLogger
from configs.config import (
    VINDR_IMAGES_ROOT,
    VINDR_CSV,
    RANDOM_SEED,
    TRAIN_VAL_SPLIT,
    BATCH_SIZE,
    NUM_WORKERS,
    EARLY_STOPPING_PATIENCE,
    NSGA3_POPULATION_SIZE,
    NSGA3_N_GENERATIONS,
    N_OBJECTIVES,
    LOG_DIR,
    CHECKPOINT_DIR
)

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions


def setup_datasets():
    """
    Setup training and validation datasets.

    Returns:
        train_dataset, val_dataset, pos_weight
    """
    print("Setting up datasets...")

    # Create preprocessor
    preprocessor = MammographyPreprocessor()

    # Load full dataset
    full_dataset = VinDRMammoBinaryDataset(
        images_root=VINDR_IMAGES_ROOT,
        csv_file=VINDR_CSV,
        preprocessor=preprocessor,
        transform=None  # Will be set during optimization
    )

    # Get labels for stratified split
    labels = [full_dataset.samples[i][-1] for i in range(len(full_dataset))]

    # Fixed train/val split
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=TRAIN_VAL_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Compute pos_weight for class imbalance
    train_labels = [labels[i] for i in train_idx]
    n_benign = sum(1 for l in train_labels if l == 0)
    n_malignant = sum(1 for l in train_labels if l == 1)
    pos_weight = n_benign / n_malignant

    print(f"Train samples: {len(train_dataset)} (Benign: {n_benign}, Malignant: {n_malignant})")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Pos weight: {pos_weight:.3f}")

    return train_dataset, val_dataset, pos_weight


def run_optimization(args):
    """
    Run NSGA-III optimization.

    Args:
        args: Command line arguments
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup datasets
    train_dataset, val_dataset, pos_weight = setup_datasets()

    # Create problem
    print("Creating optimization problem...")
    problem = BreastCancerOptimizationProblem(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        patience=EARLY_STOPPING_PATIENCE,
        max_epochs=args.max_epochs,
        pos_weight=pos_weight,
        random_seed=RANDOM_SEED
    )

    # Setup reference directions for NSGA-III
    print(f"Generating reference directions for {N_OBJECTIVES} objectives...")
    ref_dirs = get_reference_directions(
        "das-dennis",
        N_OBJECTIVES,
        n_partitions=args.n_partitions
    )
    print(f"Number of reference directions: {len(ref_dirs)}")

    # Create NSGA-III algorithm
    print("Initializing NSGA-III...")
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=args.pop_size
    )

    # Setup logging
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"nsga3_run_{args.run_id}.csv")
    logger = HyperparameterLogger(log_file)

    # Run optimization
    print(f"Starting optimization with {args.pop_size} population size for {args.n_gen} generations...")
    print(f"Total evaluations: {args.pop_size * args.n_gen}")
    print("-" * 80)

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", args.n_gen),
        seed=RANDOM_SEED,
        verbose=True,
        save_history=True
    )

    # Save results
    print("\nOptimization complete!")
    print(f"Number of Pareto solutions: {len(res.F)}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    result_file = os.path.join(CHECKPOINT_DIR, f"nsga3_results_{args.run_id}.pkl")

    with open(result_file, "wb") as f:
        pickle.dump({
            "X": res.X,  # Decision variables (hyperparameters)
            "F": res.F,  # Objectives
            "history": res.history,
            "algorithm": res.algorithm
        }, f)

    print(f"Results saved to: {result_file}")

    # Print Pareto front summary
    print("\nPareto Front Summary:")
    print("=" * 80)
    print(f"{'ID':<5} {'PR-AUC':>8} {'AUROC':>8} {'Brier':>8} {'Robust':>8}")
    print("-" * 80)

    for i, (x, f) in enumerate(zip(res.X, res.F)):
        pr_auc = -f[0]
        auroc = -f[1]
        brier = f[2]
        robust = f[3]

        print(f"{i:<5} {pr_auc:>8.4f} {auroc:>8.4f} {brier:>8.4f} {robust:>8.4f}")

        # Log to file
        hyperparams = problem._decode_hyperparameters(x)
        logger.log(i, hyperparams, f)

    print("=" * 80)

    return res


def main():
    parser = argparse.ArgumentParser(description="Run NSGA-III optimization for breast cancer detection")

    # Optimization settings
    parser.add_argument("--pop_size", type=int, default=NSGA3_POPULATION_SIZE,
                       help="Population size")
    parser.add_argument("--n_gen", type=int, default=NSGA3_N_GENERATIONS,
                       help="Number of generations")
    parser.add_argument("--n_partitions", type=int, default=12,
                       help="Number of partitions for reference directions")
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum training epochs per evaluation")
    parser.add_argument("--run_id", type=str, default="001",
                       help="Run identifier for logging")

    args = parser.parse_args()

    # Run optimization
    res = run_optimization(args)


if __name__ == "__main__":
    main()
