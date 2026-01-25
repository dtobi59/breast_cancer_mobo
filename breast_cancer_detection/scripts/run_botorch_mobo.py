"""
BoTorch-based Multi-Objective Bayesian Optimization for Breast Cancer Detection.

Replaces NSGA-III with qNEHVI acquisition for efficient exploration of the
hyperparameter-objective space.

Expected evaluations: ~200 (vs 1000 for full NSGA-III)
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import pickle
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation_functions import create_evaluation_function, decode_hyperparameters
from src.botorch_mobo import (
    MultiObjectiveGPModel,
    qNEHVIAcquisition,
    compute_reference_point,
    initial_sobol_sampling
)
from src.botorch_utils import HyperparameterTransform, BoTorchCheckpoint
from src.preprocessing import MammographyPreprocessor
from src.datasets import VinDRMammoBinaryDataset, create_breast_level_splits


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='BoTorch Multi-Objective Bayesian Optimization'
    )

    # Data paths
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory for VinDr-Mammo images')
    parser.add_argument('--csv_file', type=str, required=True,
                       help='Path to VinDr-Mammo CSV file')

    # BO parameters
    parser.add_argument('--n_initial', type=int, default=10,
                       help='Initial random evaluations (default: 10 = 2*dim)')
    parser.add_argument('--n_iterations', type=int, default=38,
                       help='BO iterations after initial sampling (default: 38)')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Candidates per BO iteration (default: 5)')
    parser.add_argument('--total_budget', type=int, default=None,
                       help='Total evaluation budget (overrides n_iterations)')

    # Acquisition parameters
    parser.add_argument('--acq_samples', type=int, default=128,
                       help='MC samples for qNEHVI (default: 128)')
    parser.add_argument('--acq_restarts', type=int, default=20,
                       help='Optimization restarts for acquisition (default: 20)')
    parser.add_argument('--ref_point_offset', type=float, default=0.1,
                       help='Reference point offset (default: 0.1)')

    # Training parameters
    parser.add_argument('--train_batch_size', type=int, default=4,
                       help='CNN training batch size (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Max CNN training epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')

    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='results/botorch_mobo',
                       help='Output directory')
    parser.add_argument('--run_id', type=str, default=None,
                       help='Run ID (default: timestamp)')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                       help='Checkpoint every N iterations (default: 5)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def setup_dataset(data_root, csv_file, seed=42):
    """Setup VinDr-Mammo dataset with breast-level split."""
    print("Loading VinDr-Mammo dataset...")

    preprocessor = MammographyPreprocessor(
        target_size=(720, 480),
        aspect_ratio=1.5
    )

    dataset = VinDRMammoBinaryDataset(
        images_root=data_root,
        csv_file=csv_file,
        preprocessor=preprocessor
    )

    print(f"Total samples: {len(dataset)}")

    # Class counts
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    n_benign = sum([1 for l in all_labels if l == 0])
    n_malignant = sum([1 for l in all_labels if l == 1])

    print(f"  Benign: {n_benign}")
    print(f"  Malignant: {n_malignant}")
    print(f"  Ratio: {n_benign/n_malignant:.2f}:1")

    # Breast-level split
    print("\nCreating breast-level train/val split...")
    train_dataset, val_dataset = create_breast_level_splits(
        dataset=dataset,
        train_ratio=0.8,
        random_state=seed,
        stratify=True
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    return train_dataset, val_dataset, n_benign, n_malignant


def evaluate_batch(evaluate_fn, X_batch, transform):
    """
    Evaluate a batch of hyperparameter configurations.

    Args:
        evaluate_fn: Evaluation function from create_evaluation_function()
        X_batch: (batch_size, n_vars) tensor in linear space
        transform: HyperparameterTransform instance

    Returns:
        Y_batch: (batch_size, n_objs) tensor of objectives
        hyperparams_list: List of hyperparameter dicts
    """
    batch_size = X_batch.shape[0]
    Y_batch = []
    hyperparams_list = []

    for i in range(batch_size):
        x = X_batch[i].numpy()

        # Decode to real hyperparameters
        hyperparams = transform.to_real_hyperparams(x)

        # Evaluate (expensive!)
        metrics = evaluate_fn(hyperparams)

        # Convert to minimization objectives
        objectives = [
            -metrics["pr_auc"],
            -metrics["auroc"],
            metrics["brier"],
            metrics["robustness_degradation"]
        ]

        Y_batch.append(objectives)
        hyperparams_list.append(hyperparams)

        print(f"  [{i+1}/{batch_size}] Evaluated")
        print(f"    PR-AUC: {metrics['pr_auc']:.4f}, "
              f"AUROC: {metrics['auroc']:.4f}, "
              f"Brier: {metrics['brier']:.4f}, "
              f"Robust: {metrics['robustness_degradation']:.4f}")

    Y_batch = torch.tensor(Y_batch, dtype=torch.float64)

    return Y_batch, hyperparams_list


def run_botorch_optimization(args):
    """Main BoTorch optimization loop."""

    # Setup
    if args.run_id is None:
        args.run_id = f"botorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    device_torch = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("="*80)
    print("BoTorch MULTI-OBJECTIVE BAYESIAN OPTIMIZATION")
    print("="*80)
    print(f"\nRun ID: {args.run_id}")
    print(f"Output: {output_dir}")

    # Budget calculation
    if args.total_budget is not None:
        total_evals = args.total_budget
        n_bo_evals = total_evals - args.n_initial
        n_iterations = n_bo_evals // args.batch_size
    else:
        n_iterations = args.n_iterations
        total_evals = args.n_initial + n_iterations * args.batch_size

    print(f"\nBudget:")
    print(f"  Initial (Sobol): {args.n_initial} evaluations")
    print(f"  BO iterations: {n_iterations}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total: {total_evals} evaluations")

    # Setup dataset
    print("\n" + "="*80)
    train_dataset, val_dataset, n_benign, n_malignant = setup_dataset(
        args.data_root, args.csv_file, args.seed
    )

    # Create evaluation function
    print("\n" + "="*80)
    print("CREATING EVALUATION FUNCTION")
    print("="*80)

    evaluate_fn = create_evaluation_function(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device_torch,
        batch_size=args.train_batch_size,
        num_workers=2,
        patience=args.patience,
        max_epochs=args.max_epochs,
        pos_weight=n_benign / n_malignant,
        random_seed=args.seed
    )

    # Setup transforms
    transform = HyperparameterTransform()
    bounds_linear = transform.bounds_linear

    # Initialize GP model
    gp_model = MultiObjectiveGPModel(
        n_objectives=4,
        n_vars=5,
        bounds=bounds_linear
    )

    # Resume or start fresh
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        gp_model, X_train, Y_train, start_iter, extras = \
            BoTorchCheckpoint.resume_from_checkpoint(args.resume, MultiObjectiveGPModel)
        all_hyperparams = extras.get('all_hyperparams', [])
    else:
        # Phase 1: Initial Sobol sampling
        print("\n" + "="*80)
        print("PHASE 1: INITIAL SOBOL SAMPLING")
        print("="*80)

        X_init = initial_sobol_sampling(
            n_vars=5,
            n_samples=args.n_initial,
            bounds=bounds_linear
        )

        print(f"\nEvaluating {args.n_initial} initial Sobol samples...")
        Y_init, hyperparams_init = evaluate_batch(evaluate_fn, X_init, transform)

        # Initialize GP
        X_train = X_init
        Y_train = Y_init
        all_hyperparams = hyperparams_init
        start_iter = 0

        print(f"\n✓ Initial sampling complete")
        print(f"  Evaluations: {len(X_train)}")

    # Save initial results
    df = pd.DataFrame(all_hyperparams)
    df['pr_auc'] = -Y_train[:, 0].numpy()
    df['auroc'] = -Y_train[:, 1].numpy()
    df['brier'] = Y_train[:, 2].numpy()
    df['robustness'] = Y_train[:, 3].numpy()
    df.to_csv(output_dir / 'evaluations.csv', index=False)

    # Phase 2: BO iterations
    print("\n" + "="*80)
    print("PHASE 2: BAYESIAN OPTIMIZATION")
    print("="*80)

    for iteration in range(start_iter, n_iterations):
        print(f"\n{'='*80}")
        print(f"BO Iteration {iteration+1}/{n_iterations}")
        print(f"{'='*80}")

        # Update GP model
        print("\n[1/4] Fitting GP models...")
        gp_model.update_data(X_train, Y_train)
        gp_model.fit_models()

        # Compute reference point
        ref_point = compute_reference_point(Y_train, offset=args.ref_point_offset)
        print(f"[2/4] Reference point: {ref_point.numpy()}")

        # Create acquisition function
        print(f"[3/4] Optimizing qNEHVI acquisition...")
        acq = qNEHVIAcquisition(
            model=gp_model.models,
            ref_point=ref_point,
            bounds=bounds_linear,
            X_baseline=X_train
        )

        # Generate candidates
        X_next = acq.optimize(
            q=args.batch_size,
            num_restarts=args.acq_restarts,
            raw_samples=512
        )

        print(f"      Generated {len(X_next)} candidates")

        # Evaluate candidates
        print(f"[4/4] Evaluating {args.batch_size} candidates...")
        Y_next, hyperparams_next = evaluate_batch(evaluate_fn, X_next, transform)

        # Update training data
        X_train = torch.cat([X_train, X_next], dim=0)
        Y_train = torch.cat([Y_train, Y_next], dim=0)
        all_hyperparams.extend(hyperparams_next)

        # Statistics
        print(f"\nIteration {iteration+1} complete:")
        print(f"  Total evaluations: {len(X_train)}/{total_evals}")

        # Find current Pareto front
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nds = NonDominatedSorting()
        fronts = nds.do(Y_train.numpy(), only_non_dominated_front=True)
        print(f"  Pareto front size: {len(fronts[0])}")

        # Save results
        df = pd.DataFrame(all_hyperparams)
        df['pr_auc'] = -Y_train[:, 0].numpy()
        df['auroc'] = -Y_train[:, 1].numpy()
        df['brier'] = Y_train[:, 2].numpy()
        df['robustness'] = Y_train[:, 3].numpy()
        df.to_csv(output_dir / 'evaluations.csv', index=False)

        # Checkpoint
        if (iteration + 1) % args.checkpoint_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_iter{iteration+1}.pt'
            BoTorchCheckpoint.save(
                checkpoint_path,
                gp_model,
                X_train,
                Y_train,
                iteration+1,
                all_hyperparams=all_hyperparams,
                config=vars(args)
            )
            print(f"  Checkpoint saved: {checkpoint_path.name}")

    # Final results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

    # Compute Pareto front
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds = NonDominatedSorting()
    fronts = nds.do(Y_train.numpy(), only_non_dominated_front=True)
    pareto_idx = fronts[0]

    X_pareto = X_train[pareto_idx]
    Y_pareto = Y_train[pareto_idx]

    print(f"\nFinal Statistics:")
    print(f"  Total evaluations: {len(X_train)}")
    print(f"  Pareto solutions: {len(pareto_idx)}")

    # Save final results
    final_results = {
        'X_all': X_train,
        'Y_all': Y_train,
        'X_pareto': X_pareto,
        'Y_pareto': Y_pareto,
        'pareto_indices': pareto_idx,
        'all_hyperparams': all_hyperparams,
        'gp_model': gp_model,
        'config': vars(args)
    }

    results_path = output_dir / 'final_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(final_results, f)

    print(f"\n✓ Results saved to: {results_path}")

    # Display Pareto front
    print(f"\nPareto Front Solutions:")
    print(f"{'ID':<5} {'PR-AUC':>8} {'AUROC':>8} {'Brier':>8} {'Robust':>8}")
    print("-" * 45)

    for i, y in enumerate(Y_pareto):
        pr_auc = -y[0].item()
        auroc = -y[1].item()
        brier = y[2].item()
        robust = y[3].item()
        print(f"{i:<5} {pr_auc:>8.4f} {auroc:>8.4f} {brier:>8.4f} {robust:>8.4f}")

    return final_results


def main():
    args = parse_args()

    try:
        results = run_botorch_optimization(args)
        print("\n✓ Optimization completed successfully!")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Optimization stopped by user")

    except Exception as e:
        print(f"\n\n[ERROR] Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
