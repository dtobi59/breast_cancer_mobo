"""
Utility functions for BoTorch-based optimization.

Handles:
- Input/output transformations
- Bounds management
- Log-scale conversions
- Checkpoint saving/loading
"""

import torch
import numpy as np
from pathlib import Path
from botorch.utils.transforms import normalize, unnormalize


class HyperparameterTransform:
    """
    Manages transformations between search space and model space.

    Hyperparameters:
    1. learning_rate: [1e-5, 1e-3] log10-scale → [-5, -3] linear
    2. weight_decay: [1e-6, 1e-2] log10-scale → [-6, -2] linear
    3. dropout: [0.0, 0.5] linear
    4. augmentation_strength: [0.0, 1.0] linear
    5. unfreeze_fraction: [0.0, 1.0] linear
    """

    def __init__(self):
        # Bounds in transformed (linear) space for BoTorch
        # [lr_log, wd_log, dropout, aug_str, unfreeze]
        self.bounds_linear = torch.tensor([
            [-5.0, -6.0, 0.0, 0.0, 0.0],  # Lower
            [-3.0, -2.0, 0.5, 1.0, 1.0]   # Upper
        ], dtype=torch.float64)

        # Which dimensions are log-scaled
        self.log_dims = [0, 1]  # LR and WD

    def to_normalized(self, X_linear):
        """
        Convert from linear space to [0,1] normalized space.

        Args:
            X_linear: (..., 5) tensor in linear bounds

        Returns:
            X_norm: (..., 5) tensor in [0,1]
        """
        return normalize(X_linear, self.bounds_linear)

    def from_normalized(self, X_norm):
        """
        Convert from [0,1] normalized to linear space.

        Args:
            X_norm: (..., 5) tensor in [0,1]

        Returns:
            X_linear: (..., 5) tensor in linear bounds
        """
        return unnormalize(X_norm, self.bounds_linear)

    def to_real_hyperparams(self, X_linear):
        """
        Convert from linear space to actual hyperparameter values.

        Args:
            X_linear: (..., 5) tensor/array with [lr_log, wd_log, dropout, aug, unfreeze]

        Returns:
            dict or list of dicts with real hyperparameters
        """
        is_tensor = torch.is_tensor(X_linear)
        if is_tensor:
            X = X_linear.cpu().numpy()
        else:
            X = np.asarray(X_linear)

        # Handle single point or batch
        single = (X.ndim == 1)
        if single:
            X = X.reshape(1, -1)

        results = []
        for x in X:
            hyperparams = {
                "learning_rate": 10 ** x[0],  # log10 → real
                "weight_decay": 10 ** x[1],   # log10 → real
                "dropout": float(x[2]),
                "augmentation_strength": float(x[3]),
                "unfreeze_fraction": float(x[4])
            }
            results.append(hyperparams)

        return results[0] if single else results

    def from_real_hyperparams(self, hyperparams_dict):
        """
        Convert from real hyperparameters to linear space.

        Args:
            hyperparams_dict: Dict with keys: learning_rate, weight_decay, etc.

        Returns:
            X_linear: (5,) array
        """
        return np.array([
            np.log10(hyperparams_dict["learning_rate"]),
            np.log10(hyperparams_dict["weight_decay"]),
            hyperparams_dict["dropout"],
            hyperparams_dict["augmentation_strength"],
            hyperparams_dict["unfreeze_fraction"]
        ])

    @property
    def normalized_bounds(self):
        """Return bounds for normalized [0,1] space."""
        return torch.tensor([[0.0]*5, [1.0]*5], dtype=torch.float64)


class BoTorchCheckpoint:
    """
    Manages checkpointing for BoTorch optimization.
    """

    @staticmethod
    def save(path, gp_model, X_train, Y_train, iteration, **kwargs):
        """
        Save BoTorch optimization state.

        Works for both single-objective and multi-objective models.

        Args:
            path: Path to save checkpoint
            gp_model: SingleObjectiveGPModel or MultiObjectiveGPModel instance
            X_train: Training inputs
            Y_train: Training objectives
            iteration: Current iteration number
            **kwargs: Additional data to save
        """
        # Detect model type
        is_multi_objective = hasattr(gp_model, 'n_objectives')

        state = {
            'X_train': X_train,
            'Y_train': Y_train,
            'iteration': iteration,
            'is_multi_objective': is_multi_objective,
            **kwargs
        }

        if is_multi_objective:
            # Multi-objective: ModelListGP
            state.update({
                'n_objectives': gp_model.n_objectives,
                'n_vars': gp_model.n_vars,
                'bounds': gp_model.bounds,
                # Save GP state dicts
                'gp_state_dicts': [
                    {
                        'model_state': model.state_dict(),
                        'likelihood_state': model.likelihood.state_dict()
                    }
                    for model in gp_model.models.models
                ] if gp_model.models is not None else None
            })
        else:
            # Single-objective: SingleTaskGP
            state.update({
                'n_vars': gp_model.n_vars,
                'bounds': gp_model.bounds,
                # Save single GP state dict
                'gp_state_dict': {
                    'model_state': gp_model.model.state_dict(),
                    'likelihood_state': gp_model.model.likelihood.state_dict()
                } if gp_model.model is not None else None
            })

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @staticmethod
    def load(path, device='cpu'):
        """
        Load BoTorch optimization state.

        Args:
            path: Path to checkpoint
            device: Device to load to

        Returns:
            dict with checkpoint data
        """
        state = torch.load(path, map_location=device)
        return state

    @staticmethod
    def resume_from_checkpoint(path, gp_model_class, device='cpu'):
        """
        Resume optimization from checkpoint.

        Works for both single-objective and multi-objective models.

        Args:
            path: Path to checkpoint
            gp_model_class: SingleObjectiveGPModel or MultiObjectiveGPModel class
            device: Device

        Returns:
            gp_model, X_train, Y_train, iteration, extras
        """
        state = BoTorchCheckpoint.load(path, device)

        # Check model type
        is_multi_objective = state.get('is_multi_objective', True)  # Default to True for backward compat

        if is_multi_objective:
            # Multi-objective model
            gp_model = gp_model_class(
                n_objectives=state['n_objectives'],
                n_vars=state['n_vars'],
                bounds=state['bounds']
            )

            # Restore training data
            X_train = state['X_train'].to(device)
            Y_train = state['Y_train'].to(device)
            gp_model.update_data(X_train, Y_train)

            # Restore GP states if available
            if state.get('gp_state_dicts') is not None:
                gp_model.fit_models()  # Create models first
                for i, gp_state in enumerate(state['gp_state_dicts']):
                    gp_model.models.models[i].load_state_dict(gp_state['model_state'])
                    gp_model.models.models[i].likelihood.load_state_dict(gp_state['likelihood_state'])

            # Extract extras
            extras = {k: v for k, v in state.items()
                     if k not in ['X_train', 'Y_train', 'iteration', 'n_objectives',
                                 'n_vars', 'bounds', 'gp_state_dicts', 'is_multi_objective']}

        else:
            # Single-objective model
            gp_model = gp_model_class(
                n_vars=state['n_vars'],
                bounds=state['bounds']
            )

            # Restore training data
            X_train = state['X_train'].to(device)
            Y_train = state['Y_train'].to(device)
            gp_model.update_data(X_train, Y_train)

            # Restore GP state if available
            if state.get('gp_state_dict') is not None:
                gp_model.fit_model()  # Create model first
                gp_model.model.load_state_dict(state['gp_state_dict']['model_state'])
                gp_model.model.likelihood.load_state_dict(state['gp_state_dict']['likelihood_state'])

            # Extract extras
            extras = {k: v for k, v in state.items()
                     if k not in ['X_train', 'Y_train', 'iteration', 'n_vars',
                                 'bounds', 'gp_state_dict', 'is_multi_objective']}

        return gp_model, X_train, Y_train, state['iteration'], extras
