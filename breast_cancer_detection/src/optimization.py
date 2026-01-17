"""
pymoo problem definition for multi-objective optimization with NSGA-III.

4 objectives:
1. Maximize PR-AUC (minimize negative PR-AUC)
2. Maximize AUROC (minimize negative AUROC)
3. Minimize Brier score
4. Minimize robustness degradation
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pymoo.core.problem import Problem

from .models import build_resnet152
from .augmentations import get_augmentation
from .training import train_model
from .datasets import VinDRMammoBinaryDataset


class BreastCancerOptimizationProblem(Problem):
    """
    pymoo Problem for breast cancer detection optimization.

    Decision variables (5 continuous hyperparameters):
    1. learning_rate: [1e-5, 1e-3] (log-scale)
    2. weight_decay: [1e-6, 1e-2] (log-scale)
    3. dropout: [0.0, 0.5]
    4. augmentation_strength: [0.0, 1.0]
    5. unfreeze_fraction: [0.0, 1.0]

    Objectives (4):
    1. -PR-AUC (minimize negative = maximize PR-AUC)
    2. -AUROC (minimize negative = maximize AUROC)
    3. Brier score (minimize)
    4. Robustness degradation (minimize)
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        device,
        batch_size=4,
        num_workers=2,
        patience=10,
        max_epochs=100,
        pos_weight=None,
        random_seed=42
    ):
        """
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            device: torch.device
            batch_size: Batch size for training
            num_workers: Number of dataloader workers
            patience: Early stopping patience
            max_epochs: Maximum training epochs
            pos_weight: Positive class weight for loss
            random_seed: Random seed for reproducibility
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patience = patience
        self.max_epochs = max_epochs
        self.pos_weight = pos_weight
        self.random_seed = random_seed

        # Hyperparameter bounds
        # [learning_rate_log, weight_decay_log, dropout, aug_strength, unfreeze_frac]
        xl = np.array([-5.0, -6.0, 0.0, 0.0, 0.0])  # log10(lr), log10(wd), ...
        xu = np.array([-3.0, -2.0, 0.5, 1.0, 1.0])

        # Initialize pymoo Problem
        # n_var: number of decision variables
        # n_obj: number of objectives
        # n_constr: number of constraints
        super().__init__(
            n_var=5,
            n_obj=4,
            n_constr=0,
            xl=xl,
            xu=xu
        )

        self.evaluation_count = 0

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objective functions for a population.

        Args:
            X: Population array of shape (pop_size, n_var)
            out: Output dictionary to store objectives

        Each row of X contains:
        [lr_log, wd_log, dropout, aug_strength, unfreeze_frac]
        """
        objectives = []

        for x in X:
            # Decode hyperparameters
            hyperparams = self._decode_hyperparameters(x)

            # Train model with these hyperparameters
            metrics = self._train_and_evaluate(hyperparams)

            # Convert to minimization objectives
            obj = [
                -metrics["pr_auc"],              # Maximize PR-AUC
                -metrics["auroc"],               # Maximize AUROC
                metrics["brier"],                # Minimize Brier
                metrics["robustness_degradation"] # Minimize robustness degradation
            ]

            objectives.append(obj)

            self.evaluation_count += 1
            print(f"[Evaluation {self.evaluation_count}] Objectives: {obj}")

        out["F"] = np.array(objectives)

    def _decode_hyperparameters(self, x):
        """
        Decode decision variables to hyperparameters.

        Args:
            x: Array of shape (n_var,)

        Returns:
            Dictionary of hyperparameters
        """
        lr_log, wd_log, dropout, aug_strength, unfreeze_frac = x

        return {
            "learning_rate": 10 ** lr_log,
            "weight_decay": 10 ** wd_log,
            "dropout": dropout,
            "augmentation_strength": aug_strength,
            "unfreeze_fraction": unfreeze_frac
        }

    def _train_and_evaluate(self, hyperparams):
        """
        Train model with given hyperparameters and evaluate.

        Args:
            hyperparams: Dictionary of hyperparameters

        Returns:
            Dictionary of metrics
        """
        # Set random seed
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Create augmentation transform
        augmentation = get_augmentation(hyperparams["augmentation_strength"])

        # Handle augmentation for Subset datasets
        # Create a new dataset instance with augmentation, then wrap with same indices
        if isinstance(self.train_dataset, Subset):
            # Get the base dataset and indices
            base_dataset = self.train_dataset.dataset
            train_indices = self.train_dataset.indices

            # Create new dataset instance with augmentation
            if isinstance(base_dataset, VinDRMammoBinaryDataset):
                # Access the constructor parameters from the original dataset
                train_dataset_aug = VinDRMammoBinaryDataset(
                    images_root=base_dataset.images_root,
                    csv_file=None,  # Won't be used since we copy samples
                    preprocessor=base_dataset.preprocessor,
                    transform=augmentation
                )
                # Copy samples directly to avoid reloading CSV
                train_dataset_aug.samples = base_dataset.samples
                # Wrap with same indices
                train_dataset_aug = Subset(train_dataset_aug, train_indices)
            else:
                # Fallback: try to set transform on base dataset
                base_dataset.transform = augmentation
                train_dataset_aug = self.train_dataset
        else:
            # Not a Subset, set transform directly
            self.train_dataset.transform = augmentation
            train_dataset_aug = self.train_dataset

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset_aug,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        # Build model
        model = build_resnet152(
            pretrained=True,
            dropout=hyperparams["dropout"],
            unfreeze_fraction=hyperparams["unfreeze_fraction"]
        )
        model = model.to(self.device)

        # Train model
        model, metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_dataset=self.val_dataset,
            device=self.device,
            learning_rate=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
            patience=self.patience,
            max_epochs=self.max_epochs,
            pos_weight=self.pos_weight,
            verbose=False  # Suppress training output during optimization
        )

        return metrics


class HyperparameterLogger:
    """
    Log hyperparameters and objectives during optimization.
    """

    def __init__(self, log_file):
        self.log_file = log_file
        self.evaluations = []

        # Write header
        with open(log_file, "w") as f:
            f.write("evaluation,lr,wd,dropout,aug_str,unfreeze,pr_auc,auroc,brier,robust_deg\n")

    def log(self, evaluation_id, hyperparams, objectives):
        """
        Log one evaluation.

        Args:
            evaluation_id: Evaluation number
            hyperparams: Dictionary of hyperparameters
            objectives: List of objective values [neg_pr_auc, neg_auroc, brier, robust_deg]
        """
        self.evaluations.append({
            "id": evaluation_id,
            "hyperparams": hyperparams,
            "objectives": objectives
        })

        # Convert objectives back to metrics
        pr_auc = -objectives[0]
        auroc = -objectives[1]
        brier = objectives[2]
        robust_deg = objectives[3]

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(
                f"{evaluation_id},"
                f"{hyperparams['learning_rate']:.6f},"
                f"{hyperparams['weight_decay']:.6f},"
                f"{hyperparams['dropout']:.4f},"
                f"{hyperparams['augmentation_strength']:.4f},"
                f"{hyperparams['unfreeze_fraction']:.4f},"
                f"{pr_auc:.4f},"
                f"{auroc:.4f},"
                f"{brier:.4f},"
                f"{robust_deg:.4f}\n"
            )
