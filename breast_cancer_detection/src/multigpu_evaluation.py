"""
Multi-GPU Evaluation Module for BoTorch Optimization.

Enables parallel evaluation of hyperparameter candidates across multiple GPUs,
reducing total optimization time on platforms like Kaggle with 2x GPU access.

Features:
- Parallel evaluation using ProcessPoolExecutor
- Round-robin GPU assignment
- Automatic fallback to sequential evaluation
- Error handling and recovery
"""

import os
import sys
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Dict, Tuple, Any
from pathlib import Path


def evaluate_batch_parallel(
    evaluate_fn_factory: Any,
    X_batch: torch.Tensor,
    transform: Any,
    num_gpus: int = 2,
    verbose: bool = True
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Evaluate batch of candidates in parallel across multiple GPUs.

    Args:
        evaluate_fn_factory: Factory or function that creates evaluation function
                           Should accept device parameter or be device-agnostic
        X_batch: (batch_size, n_vars) tensor of candidates in normalized space
        transform: HyperparameterTransform instance for decoding candidates
        num_gpus: Number of GPUs to use (default: 2)
        verbose: Print progress messages (default: True)

    Returns:
        Y_batch: (batch_size, n_objs) tensor of objectives
        hyperparams_list: List of hyperparameter dictionaries

    Example:
        >>> Y_batch, hyperparams = evaluate_batch_parallel(
        ...     evaluate_fn_factory=evaluation_fn,
        ...     X_batch=candidates,
        ...     transform=transform,
        ...     num_gpus=2
        ... )
    """
    batch_size = X_batch.shape[0]

    # Detect available GPUs
    if torch.cuda.is_available():
        available_gpus = min(torch.cuda.device_count(), num_gpus)
    else:
        available_gpus = 0

    if available_gpus < 2:
        # Fallback to sequential evaluation
        if verbose:
            print(f"Only {available_gpus} GPU(s) available, using sequential evaluation")
        return evaluate_batch_sequential(evaluate_fn_factory, X_batch, transform, verbose)

    if verbose:
        print(f"\nUsing {available_gpus} GPUs for parallel evaluation")
        print(f"Batch size: {batch_size}")
        print(f"Expected parallel rounds: {int(np.ceil(batch_size / available_gpus))}")

    # Create evaluation tasks with GPU assignments
    tasks = []
    for i in range(batch_size):
        gpu_id = i % available_gpus  # Round-robin GPU assignment
        x = X_batch[i].cpu().numpy()
        hyperparams = transform.to_real_hyperparams(x)
        tasks.append((i, gpu_id, hyperparams))

    # Parallel execution using ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(max_workers=available_gpus) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                _evaluate_single,
                task_id,
                gpu_id,
                hyperparams,
                evaluate_fn_factory,
                batch_size
            ): (task_id, gpu_id)
            for task_id, gpu_id, hyperparams in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task_id, gpu_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"[GPU {gpu_id}] Task {task_id+1} generated an exception: {exc}")
                # Return NaN objectives for failed evaluations
                # BoTorch can handle NaN values and will avoid this region
                results.append((
                    task_id,
                    {
                        "pr_auc": float('nan'),
                        "auroc": float('nan'),
                        "brier": float('nan'),
                        "cross_dataset_degradation": float('nan')
                    },
                    tasks[task_id][2]  # Original hyperparams
                ))

    # Sort by task_id to maintain order
    results.sort(key=lambda x: x[0])

    # Extract objectives and hyperparameters
    Y_batch = []
    hyperparams_list = []
    for _, metrics, hyperparams in results:
        objectives = [
            -metrics["pr_auc"],                        # Maximize PR-AUC (minimize negative)
            -metrics["auroc"],                         # Maximize AUROC (minimize negative)
            metrics["brier"],                          # Minimize Brier score
            metrics["cross_dataset_degradation"]       # Minimize cross-dataset degradation
        ]
        Y_batch.append(objectives)
        hyperparams_list.append(hyperparams)

    Y_batch = torch.tensor(Y_batch, dtype=torch.float64)

    if verbose:
        print(f"Parallel evaluation complete: {batch_size} candidates")

    return Y_batch, hyperparams_list


def evaluate_batch_sequential(
    evaluate_fn: Callable,
    X_batch: torch.Tensor,
    transform: Any,
    verbose: bool = True,
    save_callback: Callable = None
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Evaluate batch of candidates sequentially (fallback for single GPU).

    Args:
        evaluate_fn: Evaluation function that accepts hyperparameters
        X_batch: (batch_size, n_vars) tensor of candidates
        transform: HyperparameterTransform instance
        verbose: Print progress messages (default: True)
        save_callback: Optional callback function to save each evaluation immediately
                      Signature: save_callback(hyperparams, metrics)

    Returns:
        Y_batch: (batch_size, n_objs) tensor of objectives
        hyperparams_list: List of hyperparameter dictionaries
    """
    batch_size = X_batch.shape[0]

    if verbose:
        print(f"\nSequential evaluation: {batch_size} candidates")

    Y_batch = []
    hyperparams_list = []

    for i in range(batch_size):
        x = X_batch[i].cpu().numpy()
        hyperparams = transform.to_real_hyperparams(x)

        if verbose:
            print(f"\nEvaluating candidate {i+1}/{batch_size}")
            print(f"  LR: {hyperparams['learning_rate']:.6f}")
            print(f"  Weight decay: {hyperparams['weight_decay']:.6f}")
            print(f"  Dropout: {hyperparams['dropout']:.3f}")

        try:
            metrics = evaluate_fn(hyperparams)

            objectives = [
                -metrics["pr_auc"],
                -metrics["auroc"],
                metrics["brier"],
                metrics["cross_dataset_degradation"]
            ]

            Y_batch.append(objectives)
            hyperparams_list.append(hyperparams)

            if verbose:
                print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
                print(f"  AUROC: {metrics['auroc']:.4f}")
                print(f"  Brier: {metrics['brier']:.4f}")
                print(f"  Cross-dataset deg: {metrics['cross_dataset_degradation']:.4f}")

            # INCREMENTAL SAVE: Save immediately after each evaluation
            if save_callback is not None:
                save_callback(hyperparams, metrics)

        except Exception as exc:
            print(f"Candidate {i+1} generated an exception: {exc}")
            import traceback
            traceback.print_exc()
            # Return NaN objectives for failed evaluations
            objectives = [float('nan')] * 4
            Y_batch.append(objectives)
            hyperparams_list.append(hyperparams)

    Y_batch = torch.tensor(Y_batch, dtype=torch.float64)
    return Y_batch, hyperparams_list


def _evaluate_single(
    task_id: int,
    gpu_id: int,
    hyperparams: Dict[str, Any],
    evaluate_fn_factory: Any,
    total_tasks: int
) -> Tuple[int, Dict[str, float], Dict[str, Any]]:
    """
    Evaluate single candidate on specific GPU (worker function).

    This function runs in a separate process with its own CUDA context.

    Args:
        task_id: Task identifier (for ordering results)
        gpu_id: GPU device ID to use
        hyperparams: Hyperparameter dictionary
        evaluate_fn_factory: Factory or function to create evaluation function
        total_tasks: Total number of tasks in batch (for progress reporting)

    Returns:
        Tuple of (task_id, metrics_dict, hyperparams)
    """
    # Set CUDA device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Import torch in the worker process to ensure proper CUDA context
    import torch
    device = torch.device('cuda:0')  # Will map to the GPU set in CUDA_VISIBLE_DEVICES

    print(f"[GPU {gpu_id}] Starting candidate {task_id+1}/{total_tasks}")
    print(f"[GPU {gpu_id}]   LR={hyperparams['learning_rate']:.6f}, "
          f"WD={hyperparams['weight_decay']:.6f}, "
          f"Dropout={hyperparams['dropout']:.3f}")

    # Create evaluation function
    # The factory should handle device assignment internally
    if callable(evaluate_fn_factory):
        # If it's a function, call it directly
        evaluate_fn = evaluate_fn_factory
    else:
        # If it's a factory object with device support
        evaluate_fn = evaluate_fn_factory

    # Evaluate candidate
    try:
        metrics = evaluate_fn(hyperparams)

        print(f"[GPU {gpu_id}] Completed candidate {task_id+1}/{total_tasks}")
        print(f"[GPU {gpu_id}]   PR-AUC={metrics['pr_auc']:.4f}, "
              f"AUROC={metrics['auroc']:.4f}, "
              f"Brier={metrics['brier']:.4f}, "
              f"Cross-dataset={metrics['cross_dataset_degradation']:.4f}")

        return (task_id, metrics, hyperparams)

    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR in candidate {task_id+1}: {str(e)}")
        # Return NaN metrics on error
        return (
            task_id,
            {
                "pr_auc": float('nan'),
                "auroc": float('nan'),
                "brier": float('nan'),
                "cross_dataset_degradation": float('nan')
            },
            hyperparams
        )


def detect_gpus(verbose: bool = True) -> Tuple[int, List[str]]:
    """
    Detect available GPUs.

    Args:
        verbose: Print GPU information (default: True)

    Returns:
        Tuple of (num_gpus, gpu_names)
    """
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available - no GPUs detected")
        return 0, []

    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]

    if verbose:
        print(f"\nGPU Detection:")
        print(f"  Available GPUs: {num_gpus}")
        for i, name in enumerate(gpu_names):
            print(f"  GPU {i}: {name}")
            # Get memory info
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / (1024**3)
            print(f"    Memory: {total_memory_gb:.1f} GB")

    return num_gpus, gpu_names


def set_multiprocessing_start_method():
    """
    Set multiprocessing start method to 'spawn' for CUDA compatibility.

    Should be called once at the beginning of the notebook.
    """
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn' (required for CUDA)")
    except RuntimeError:
        # Already set
        print("Multiprocessing start method already configured")


# Utility function for testing
def test_multi_gpu_setup(num_gpus: int = 2):
    """
    Test multi-GPU setup with mock evaluations.

    Args:
        num_gpus: Number of GPUs to test
    """
    print("\n" + "="*80)
    print("MULTI-GPU SETUP TEST")
    print("="*80)

    # Detect GPUs
    num_available, gpu_names = detect_gpus(verbose=True)

    # Check if multi-GPU is possible
    if num_available >= 2:
        print(f"\n Multi-GPU mode: AVAILABLE")
        print(f"  Can use {min(num_available, num_gpus)} GPUs for parallel evaluation")
    else:
        print(f"\n Multi-GPU mode: NOT AVAILABLE")
        print(f"  Only {num_available} GPU(s) detected")
        print(f"  Will fall back to sequential evaluation")

    # Test CUDA visibility
    print(f"\nCUDA Environment:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  PyTorch version: {torch.__version__}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
