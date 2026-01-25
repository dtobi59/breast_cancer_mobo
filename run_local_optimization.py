"""
Local Optimization Script for Breast Cancer Detection

This script runs NSGA-III optimization locally (without Colab/Jupyter).
It automatically configures paths and supports both standard and surrogate-assisted modes.

Usage:
    # Standard NSGA-III (demo)
    python run_local_optimization.py --mode standard --scale demo

    # Surrogate-assisted (recommended)
    python run_local_optimization.py --mode surrogate --scale recommended

    # Full configuration
    python run_local_optimization.py \
        --mode surrogate \
        --data_root /path/to/vindr/images \
        --csv_file /path/to/vindr/metadata.csv \
        --pop_size 20 \
        --n_gen 50 \
        --output_dir results/nsga3
"""

import sys
import os
from pathlib import Path
import argparse

# Add project to path
PROJECT_ROOT = Path(__file__).parent.resolve()
BREAST_CANCER_PATH = PROJECT_ROOT / "breast_cancer_detection"

if str(BREAST_CANCER_PATH) not in sys.path:
    sys.path.insert(0, str(BREAST_CANCER_PATH))

# Now imports should work
import numpy as np
import torch
from src.preprocessing import MammographyPreprocessor
from src.datasets import VinDRMammoBinaryDataset, create_breast_level_splits
from src.optimization import BreastCancerOptimizationProblem


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NSGA-III Optimization (Local)')

    # Mode selection
    parser.add_argument('--mode', type=str, default='surrogate',
                       choices=['standard', 'surrogate'],
                       help='Optimization mode (default: surrogate)')

    parser.add_argument('--scale', type=str, default='demo',
                       choices=['demo', 'recommended', 'full'],
                       help='Optimization scale (default: demo)')

    # Data paths
    parser.add_argument('--data_root', type=str, default=None,
                       help='Root directory for VinDr-Mammo images')
    parser.add_argument('--csv_file', type=str, default=None,
                       help='Path to VinDr-Mammo CSV file')

    # Output
    parser.add_argument('--output_dir', type=str, default='results/nsga3_local',
                       help='Output directory for results')

    # Overrides (optional)
    parser.add_argument('--pop_size', type=int, default=None,
                       help='Population size (overrides scale default)')
    parser.add_argument('--n_gen', type=int, default=None,
                       help='Number of generations (overrides scale default)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Max epochs per model (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device (default: cuda)')

    return parser.parse_args()


def get_config(args):
    """Generate configuration based on mode and scale."""

    configs = {
        'standard': {
            'demo': {'population': 5, 'generations': 3, 'max_epochs': 5},
            'recommended': {'population': 8, 'generations': 10, 'max_epochs': 50},
            'full': {'population': 20, 'generations': 50, 'max_epochs': 50},
        },
        'surrogate': {
            'demo': {
                'population': 10,
                'generations': 10,
                'n_gen_init': 2,
                'n_true_per_gen': 5,
                'max_epochs': 50,
            },
            'recommended': {
                'population': 20,
                'generations': 50,
                'n_gen_init': 3,
                'n_true_per_gen': 5,
                'max_epochs': 50,
            },
        }
    }

    config = configs[args.mode][args.scale].copy()

    # Apply overrides
    if args.pop_size is not None:
        config['population'] = args.pop_size
    if args.n_gen is not None:
        config['generations'] = args.n_gen

    config['batch_size'] = args.batch_size
    config['max_epochs'] = args.max_epochs
    config['mode'] = args.mode
    config['scale'] = args.scale

    return config


def run_standard_nsga3(config, train_dataset, val_dataset, n_benign, n_malignant, device, output_dir):
    """Run standard NSGA-III optimization."""

    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    import pickle

    print("="*80)
    print("STANDARD NSGA-III OPTIMIZATION")
    print("="*80)

    # Create problem
    problem = BreastCancerOptimizationProblem(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        batch_size=config['batch_size'],
        num_workers=2,
        patience=10,
        max_epochs=config['max_epochs'],
        pos_weight=n_benign/n_malignant,
        random_seed=42
    )

    print(f"\nProblem: {problem.n_var} vars, {problem.n_obj} objectives")

    # Create algorithm
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=3)
    algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=config['population'])

    print(f"Algorithm: {config['population']} pop × {config['generations']} gen")
    print(f"Expected evaluations: {config['population'] * config['generations']}")

    # Run optimization
    print(f"\nStarting optimization...\n")

    res = minimize(
        problem,
        algorithm,
        ('n_gen', config['generations']),
        verbose=True,
        save_history=True,
        seed=42
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"results_{config['mode']}_{config['scale']}.pkl"

    with open(results_path, 'wb') as f:
        pickle.dump(res, f)

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nTotal evaluations: {res.algorithm.n_eval}")
    print(f"Pareto solutions: {len(res.F)}")
    print(f"\nResults saved to: {results_path}")

    return res


def run_surrogate_nsga3(config, train_dataset, val_dataset, n_benign, n_malignant, device, output_dir):
    """Run surrogate-assisted NSGA-III optimization."""

    from src.surrogate_optimizer import MultiObjectiveGPSurrogate
    from src.acquisition import get_acquisition_function
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    import pickle

    print("="*80)
    print("SURROGATE-ASSISTED NSGA-III OPTIMIZATION")
    print("="*80)

    # Create problem
    problem = BreastCancerOptimizationProblem(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        batch_size=config['batch_size'],
        num_workers=2,
        patience=10,
        max_epochs=config['max_epochs'],
        pos_weight=n_benign/n_malignant,
        random_seed=42,
        use_surrogate=False,
        surrogate_manager=None
    )

    print(f"\nProblem: {problem.n_var} vars, {problem.n_obj} objectives")

    # Initialize
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=3)
    algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=config['population'])
    surrogate = MultiObjectiveGPSurrogate(n_objectives=4, n_vars=5)
    acquisition_fn = get_acquisition_function('uncertainty')

    # Budget calculation
    phase1_evals = config['population'] * config['n_gen_init']
    phase2_evals = (config['generations'] - config['n_gen_init']) * config['n_true_per_gen']
    total_expected = phase1_evals + phase2_evals
    baseline = config['population'] * config['generations']

    print(f"\nBudget:")
    print(f"  Phase 1: {phase1_evals} evals")
    print(f"  Phase 2: {phase2_evals} evals")
    print(f"  Total: {total_expected} / {baseline} ({100*total_expected/baseline:.1f}%)")
    print(f"  Savings: {baseline - total_expected} evals ({100*(baseline-total_expected)/baseline:.1f}%)")

    # Phase 1
    print(f"\n{'='*80}")
    print("PHASE 1: INITIAL SAMPLING")
    print("="*80)

    res_init = minimize(
        problem,
        algorithm,
        termination=("n_gen", config['n_gen_init']),
        seed=42,
        verbose=True,
        save_history=True
    )

    X_init, F_init = problem.get_evaluation_history()
    surrogate.update(X_init, F_init)
    surrogate.fit()

    print(f"\nPhase 1 complete: {len(X_init)} evaluations")

    # Phase 2
    print(f"\n{'='*80}")
    print("PHASE 2: SURROGATE-ASSISTED")
    print("="*80)

    current_gen = config['n_gen_init']

    while current_gen < config['generations']:
        current_gen += 1
        print(f"\nGeneration {current_gen}/{config['generations']}")

        offspring = algorithm.infill()
        X_offspring = offspring.get("X")

        F_mean, F_std = surrogate.predict(X_offspring, return_std=True)
        selected_idx = acquisition_fn(F_mean, F_std, config['n_true_per_gen'])

        print(f"  Evaluating {len(selected_idx)} selected candidates...")

        problem.use_surrogate = False
        X_selected = X_offspring[selected_idx]
        F_selected = []

        for i, x in enumerate(X_selected):
            out_dict = {}
            problem._evaluate(np.array([x]), out_dict)
            F_selected.append(out_dict["F"][0])

        F_selected = np.array(F_selected)
        surrogate.update(X_selected, F_selected)
        surrogate.fit()

        problem.use_surrogate = True
        problem.surrogate_manager = surrogate

        F_offspring = []
        for i, x in enumerate(X_offspring):
            if i in selected_idx:
                idx_in_selected = np.where(selected_idx == i)[0][0]
                F_offspring.append(F_selected[idx_in_selected])
            else:
                out_dict = {}
                problem._evaluate(np.array([x]), out_dict)
                F_offspring.append(out_dict["F"][0])

        offspring.set("F", np.array(F_offspring))
        algorithm.advance(infills=offspring)

        total_true = len(problem.eval_history_X)
        print(f"  Total true evals: {total_true}/{baseline} ({100*total_true/baseline:.1f}%)")

    # Save results
    X_all, F_all = problem.get_evaluation_history()

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"results_{config['mode']}_{config['scale']}.pkl"

    final_results = {
        'X_history': X_all,
        'F_history': F_all,
        'algorithm': algorithm,
        'surrogate': surrogate,
        'pop_X': algorithm.pop.get("X"),
        'pop_F': algorithm.pop.get("F"),
        'n_true_evals': len(X_all),
        'config': config
    }

    with open(results_path, 'wb') as f:
        pickle.dump(final_results, f)

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nTrue evaluations: {len(X_all)}/{baseline}")
    print(f"Savings: {baseline - len(X_all)} ({100*(baseline-len(X_all))/baseline:.1f}%)")
    print(f"Pareto solutions: {len(algorithm.pop.get('F'))}")
    print(f"\nResults saved to: {results_path}")

    # Create compatible res object
    class SurrogateResult:
        def __init__(self, algorithm):
            self.algorithm = algorithm
            self.X = algorithm.pop.get("X")
            self.F = algorithm.pop.get("F")

    return SurrogateResult(algorithm)


def main():
    """Main execution."""

    args = parse_args()
    config = get_config(args)

    print("="*80)
    print("NSGA-III OPTIMIZATION - LOCAL EXECUTION")
    print("="*80)
    print(f"\nMode: {config['mode']}")
    print(f"Scale: {config['scale']}")
    print(f"Population: {config['population']}")
    print(f"Generations: {config['generations']}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check data paths
    if args.data_root is None or args.csv_file is None:
        print("\n[WARNING] Data paths not specified")
        print("Using demo mode with mock data (if available)")
        print("\nFor real optimization, specify:")
        print("  --data_root /path/to/vindr/images")
        print("  --csv_file /path/to/vindr/metadata.csv")
        return

    # Load dataset
    print(f"\n{'='*80}")
    print("LOADING DATASET")
    print("="*80)

    preprocessor = MammographyPreprocessor(
        target_size=(720, 480),
        aspect_ratio=1.5
    )

    dataset = VinDRMammoBinaryDataset(
        images_root=args.data_root,
        csv_file=args.csv_file,
        preprocessor=preprocessor
    )

    print(f"\nTotal samples: {len(dataset)}")

    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    n_benign = sum([1 for l in all_labels if l == 0])
    n_malignant = sum([1 for l in all_labels if l == 1])

    print(f"  Benign: {n_benign}")
    print(f"  Malignant: {n_malignant}")

    train_dataset, val_dataset = create_breast_level_splits(
        dataset=dataset,
        train_ratio=0.8,
        random_state=42,
        stratify=True
    )

    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")

    # Setup output
    output_dir = Path(args.output_dir)

    # Run optimization
    if config['mode'] == 'standard':
        res = run_standard_nsga3(config, train_dataset, val_dataset,
                                 n_benign, n_malignant, device, output_dir)
    else:
        res = run_surrogate_nsga3(config, train_dataset, val_dataset,
                                  n_benign, n_malignant, device, output_dir)

    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
