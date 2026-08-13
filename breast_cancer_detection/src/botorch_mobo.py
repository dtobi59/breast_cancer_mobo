"""
BoTorch-based Multi-Objective Bayesian Optimization utilities.

Implements:
- GP model initialization and fitting
- qNEHVI acquisition function
- Batch candidate generation
- Hypervolume reference point computation
"""

import torch
import gpytorch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning
)
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine


class MultiObjectiveGPModel:
    """
    Manages multiple GPs for multi-objective optimization.

    Uses independent SingleTaskGP models (one per objective).
    """

    def __init__(self, n_objectives=4, n_vars=5, bounds=None):
        """
        Args:
            n_objectives: Number of objectives (4)
            n_vars: Number of decision variables (5)
            bounds: Tensor of shape (2, n_vars) with [lower, upper] bounds
        """
        self.n_objectives = n_objectives
        self.n_vars = n_vars
        self.bounds = bounds if bounds is not None else torch.tensor([
            [0.0] * n_vars,  # Lower bounds (normalized)
            [1.0] * n_vars   # Upper bounds (normalized)
        ], dtype=torch.float64)

        self.models = None
        self.train_X = None
        self.train_Y = None

    def update_data(self, X_new, Y_new):
        """
        Add new evaluations to training data.

        Args:
            X_new: (n_new, n_vars) tensor of inputs
            Y_new: (n_new, n_objs) tensor of objectives
        """
        if self.train_X is None:
            self.train_X = X_new
            self.train_Y = Y_new
        else:
            self.train_X = torch.cat([self.train_X, X_new], dim=0)
            self.train_Y = torch.cat([self.train_Y, Y_new], dim=0)

    def fit_models(self):
        """
        Fit independent GP models for each objective.

        Uses:
        - Matern 5/2 kernel (good for smooth functions)
        - Automatic relevance determination (ARD) for per-dimension lengthscales
        - Output standardization
        """
        if self.train_X is None or len(self.train_X) < 2:
            raise ValueError("Need at least 2 training points")

        models = []

        for i in range(self.n_objectives):
            # Create GP model for objective i
            gp = SingleTaskGP(
                train_X=self.train_X,
                train_Y=self.train_Y[:, i:i+1],  # (n, 1)
                # Kernel will be Matern 5/2 with ARD by default
            )

            # Fit hyperparameters
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            models.append(gp)

        # Create ModelListGP for multi-objective acquisition
        self.models = ModelListGP(*models)

        return self.models

    def predict(self, X):
        """
        Predict objectives with uncertainty.

        Args:
            X: (n_candidates, n_vars) tensor

        Returns:
            means: (n_candidates, n_objs)
            stds: (n_candidates, n_objs)
        """
        if self.models is None:
            raise ValueError("Models not fitted")

        means = []
        stds = []

        with torch.no_grad():
            for gp in self.models.models:
                posterior = gp.posterior(X)
                mean = posterior.mean.squeeze(-1)
                std = posterior.variance.sqrt().squeeze(-1)
                means.append(mean)
                stds.append(std)

        return torch.stack(means, dim=-1), torch.stack(stds, dim=-1)


class qNEHVIAcquisition:
    """
    Manages qNEHVI acquisition function and optimization.
    """

    def __init__(self, model, ref_point, bounds, X_baseline, sampler=None):
        """
        Args:
            model: ModelListGP instance
            ref_point: Reference point for hypervolume (nadir point)
            bounds: (2, n_vars) bounds tensor
            X_baseline: Training inputs for partitioning
            sampler: MC sampler (default: SobolQMCNormalSampler)
        """
        self.model = model
        self.ref_point = ref_point
        self.bounds = bounds

        # Get training targets for partitioning
        # For ModelListGP, need to combine all objectives
        Y_train = []
        for gp in model.models:
            Y_train.append(gp.train_targets.unsqueeze(-1))
        Y_train = torch.cat(Y_train, dim=-1)

        # Create acquisition function
        from botorch.sampling.normal import SobolQMCNormalSampler
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        self.acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),  # Convert to list
            X_baseline=X_baseline,
            sampler=sampler
        )

    def optimize(self, q=5, num_restarts=20, raw_samples=512):
        """
        Optimize acquisition function to find next batch of candidates.

        Args:
            q: Batch size (number of candidates to generate)
            num_restarts: Number of random restarts for optimization
            raw_samples: Number of initial random samples

        Returns:
            candidates: (q, n_vars) tensor of next evaluation points
        """
        candidates, acq_value = optimize_acqf(
            acq_function=self.acq_func,
            bounds=self.bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={
                "batch_limit": 5,
                "maxiter": 200,
            },
            sequential=True  # Sequentially construct batch (fantasy model)
        )

        return candidates


def compute_reference_point(Y_train, offset=0.1):
    """
    Compute reference point for hypervolume calculation.

    Uses worst observed value per objective + offset.

    Args:
        Y_train: (n, n_objs) tensor of observed objectives
        offset: Offset factor (0.1 = 10% worse than worst)

    Returns:
        ref_point: (n_objs,) tensor
    """
    worst = Y_train.min(dim=0).values  # Worst (maximum) for minimization
    ref_point = worst - offset * worst.abs()
    return ref_point


def initial_sobol_sampling(n_vars, n_samples, bounds):
    """
    Generate initial Sobol samples for exploration.

    Args:
        n_vars: Number of variables
        n_samples: Number of samples
        bounds: (2, n_vars) bounds tensor

    Returns:
        X: (n_samples, n_vars) Sobol samples
    """
    sobol = SobolEngine(dimension=n_vars, scramble=True)
    X_norm = sobol.draw(n_samples, dtype=torch.float64)

    # Unnormalize to actual bounds
    X = unnormalize(X_norm, bounds)

    return X
